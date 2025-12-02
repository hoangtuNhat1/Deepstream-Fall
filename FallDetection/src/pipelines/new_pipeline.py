import os
import sys
import json
from datetime import datetime, timezone

from loguru import logger
import pyds

from src.models.input_data import InputData

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import math

# Import các utils và thư viện khác (giữ nguyên)
from utils.bus_call import bus_call
from utils.FPS import PERF_DATA
import time
from ctypes import sizeof, c_float
from kafka import KafkaProducer

TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
MAX_DISPLAY_LEN = 64
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1
fps_log_interval = 10000


CONFIG_INFER = '/workspace/FallDetection/config_infer_primary_yolo11.txt'
STREAMMUX_WIDTH = 640
STREAMMUX_HEIGHT = 640
GPU_ID = 0

frame_interval = int(os.getenv("FRAME_PROCESS_INTERVAL", default=1))
is_file_sink = int(os.getenv("FILE_SINK", default=0))

keys_to_keep = ["recordingId", "cameraCode", "path", "duration", "startTime", "endTime"]

start_time = time.time()
fps_streams = {}

producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
TOPIC_NAME = 'inference-output'


def send_to_kafka(payload):
    try:
        cam_key = str(payload.get("cameraCode", "unknown")).encode("utf-8")
        producer.send(TOPIC_NAME, key=cam_key, value=payload)
        producer.flush()
    except Exception as e:
        logger.error(f"Failed to send message to Kafka: {e}")

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)
    if (gstname.find("video") != -1):
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.error("Failed to link decoder src pad to source bin ghost pad")
        else:
            logger.warning("Error: Decodebin did not pick nvidia decoder plugin")

def decodebin_child_added(child_proxy, Object, name, user_data):
    """Handle child elements added to decodebin"""
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') is not None:
            Object.set_property("drop-on-latency", True)

# --- Class DSL_Pipeline đã được chỉnh sửa ---
class DSL_Pipeline:
    def __init__(
        self,
        input_srcs: list,
        pipeline_name: str = 'pipeline',
        model_name: str = 'yolov11',
    ) -> None :

        global g_pipeline_name
        logger.info(f"Initializing pipeline: {pipeline_name}")
        g_pipeline_name = pipeline_name
        self.input_srcs_bk = input_srcs
        self.model_name = model_name
        Gst.init(None)

        self.init_pipeline(len(input_srcs))
        self.set_property(len(input_srcs))
        self.link_elements()
        self.init_metrics(len(input_srcs))

    def init_metrics(self, number_sources):
        """Initialize performance metrics for each branch"""
        setattr(self, f"perf_data", PERF_DATA(self.model_name, number_sources))

    def init_pipeline(self, number_sources):
        """Initialize and build the GStreamer pipeline (BỎ TRACKER)"""
        # Create Pipeline
        self.pipeline = Gst.Pipeline.new('ds-pipeline')
        if not self.pipeline:
            raise RuntimeError("Unable to create Pipeline")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        if not self.streammux:
            raise RuntimeError("Unable to create NvStreamMux")
        self.pipeline.add(self.streammux)

        for i in range(number_sources):

            uri_name = self.input_srcs_bk[i]

            source_bin = self.create_source_bin(i, uri_name)
            if not source_bin:
                raise RuntimeError(f"Unable to create source bin for {uri_name}")

            self.pipeline.add(source_bin)
            # Connect source to streammux
            padname = f"sink_{i}"
            sinkpad = self.streammux.request_pad_simple(padname)
            if not sinkpad:
                raise RuntimeError(f"Unable to create sink pad for source {i}")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(f"Unable to create src pad for source {i}")
            srcpad.link(sinkpad)

        self.pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
        if not self.pgie:
            raise RuntimeError("Unable to create nvinfer")
        self.pipeline.add(self.pgie)
        self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", f"nvtiler")
        if not self.tiler:
            raise RuntimeError("Unable to create tiler")
        self.pipeline.add(self.tiler)
        self.converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvideoconvert')
        if not self.converter:
            raise RuntimeError("Unable to create nvvideoconvert")
        self.pipeline.add(self.converter)
        self.osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
        if not self.osd:
            raise RuntimeError("Unable to create nvdsosd")
        self.pipeline.add(self.osd)
        # self.sink = Gst.ElementFactory.make("fakesink", f"fakesink")
        self.sink = Gst.ElementFactory.make('nveglglessink', 'nveglglessink')
        if not self.sink:
            raise RuntimeError(f"Unable to create sink")
        self.pipeline.add(self.sink)

    def set_property(self, number_sources) :
        self.streammux.set_property('width', STREAMMUX_WIDTH)
        self.streammux.set_property('height', STREAMMUX_HEIGHT)
        self.streammux.set_property('batch-size', number_sources)
        self.streammux.set_property('live-source', 0)
        self.pgie.set_property('config-file-path', CONFIG_INFER)
        # self.pgie.set_property('interval', frame_interval)
        self.pgie.set_property("batch-size", number_sources)

        # BỎ setup cho self.tracker

        self.osd.set_property('process-mode', OSD_PROCESS_MODE)
        self.osd.set_property('display-text', OSD_DISPLAY_TEXT)
        self.sink.set_property('enable-last-sample', 0)
        self.sink.set_property('sync', 0)
        self.sink.set_property("qos", 0)

        self.streammux.set_property('nvbuf-memory-type', 0)
        self.streammux.set_property('gpu_id', GPU_ID)
        self.pgie.set_property('gpu_id', GPU_ID)

        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        self.tiler.set_property("rows", tiler_rows)
        self.tiler.set_property("columns", tiler_columns)
        self.tiler.set_property("width", TILED_OUTPUT_WIDTH)
        self.tiler.set_property("height", TILED_OUTPUT_HEIGHT)
        # BỎ GPU ID cho tracker
        self.converter.set_property('nvbuf-memory-type', 0)
        self.converter.set_property('gpu_id', GPU_ID)
        self.osd.set_property('gpu_id', GPU_ID)
        
    def link_elements(self):
        self.streammux.link(self.pgie)
        self.pgie.link(self.tiler)
        self.tiler.link(self.converter)
        self.converter.link(self.osd)
        self.osd.link(self.sink)

    def run_pipeline(self, inputs: InputData):
        global cam_ids, cam_sources, extra_informations, extra_informations_compact
        cam_sources = inputs.get_src()
        cam_ids = inputs.get_cams_id()
        extra_informations = inputs.get_extra_informations()
        loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)
        pgie_src_pad = self.pgie.get_static_pad('src')
        if not pgie_src_pad:
             raise RuntimeError('ERROR: Failed to get pgie src pad')
        else:
            pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.pgie_src_pad_buffer_probe, 0) # Đổi tên hàm probe
        self.timeout_id = GLib.timeout_add(fps_log_interval,
                                                      getattr(self, f"perf_data").perf_print_callback)
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass
        GLib.source_remove(getattr(self, f"timeout_id"))
        self.pipeline.set_state(Gst.State.NULL)

    def set_custom_bbox(self, obj_meta):
        border_width = 6
        font_size = 18
        x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
        y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

        obj_meta.rect_params.border_width = border_width
        obj_meta.rect_params.border_color.red = 0.0
        obj_meta.rect_params.border_color.green = 0.0
        obj_meta.rect_params.border_color.blue = 1.0
        obj_meta.rect_params.border_color.alpha = 1.0
        obj_meta.text_params.font_params.font_name = 'Ubuntu'
        obj_meta.text_params.font_params.font_size = font_size
        obj_meta.text_params.x_offset = x_offset
        obj_meta.text_params.y_offset = y_offset
        obj_meta.text_params.font_params.font_color.red = 1.0
        obj_meta.text_params.font_params.font_color.green = 1.0
        obj_meta.text_params.font_params.font_color.blue = 1.0
        obj_meta.text_params.font_params.font_color.alpha = 1.0
        obj_meta.text_params.set_bg_clr = 1
        obj_meta.text_params.text_bg_clr.red = 0.0
        obj_meta.text_params.text_bg_clr.green = 0.0
        obj_meta.text_params.text_bg_clr.blue = 1.0
        obj_meta.text_params.text_bg_clr.alpha = 1.0

    def pgie_src_pad_buffer_probe(self, pad, info, user_data):
            """Hàm probe mới trên pad nguồn của PGIE"""
            buf = info.get_buffer()
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

            l_frame = batch_meta.frame_meta_list
            while l_frame:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                current_index = frame_meta.source_id
                cam_info = extra_informations[current_index]

                start_time_frame = datetime.now(timezone.utc).isoformat()

                # Khởi tạo biến cờ
                has_detection = False 
                
                # --- Bắt đầu kiểm tra object ---
                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break
                    
                    # Nếu vòng lặp này được chạy, nghĩa là CÓ object
                    has_detection = True 
                    
                    self.set_custom_bbox(obj_meta)

                    # Chuẩn bị Payload Kafka (giữ nguyên logic cũ)
                    payload = {
                        "cameraCode": cam_info.get("meta", {}).get("cam_id", ""),
                        "path": cam_info.get("path", ""),
                        "startTime": start_time_frame,
                        "endTime": datetime.now(timezone.utc).isoformat(),
                        "frame_num": int(frame_meta.frame_num),
                        "detection": obj_meta.class_id,
                        "confidence": float(obj_meta.confidence),
                        "bbox": {
                                        "x": float(obj_meta.rect_params.left),
                                        "y": float(obj_meta.rect_params.top),
                                        "w": float(obj_meta.rect_params.width),
                                        "h": float(obj_meta.rect_params.height)
                                    },
                    }
                    send_to_kafka(payload)

                    stream_index = "stream{0}".format(frame_meta.pad_index)
                    perf_data = getattr(self, f"perf_data")
                    perf_data.update_fps(stream_index)
                    
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

            return Gst.PadProbeReturn.OK
    def create_source_bin(self, index, uri):
        bin_name = "source-bin-%02d" % index

        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, self.pipeline)

        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin