import time
from threading import Lock
start_time=time.time()

from loguru import logger as logging
import json

fps_mutex = Lock()

class GETFPS:
    def __init__(self,stream_id):
        global start_time
        self.start_time=start_time
        self.is_first=True
        self.frame_count=0
        self.stream_id=stream_id

    def update_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        else:
            global fps_mutex
            with fps_mutex:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        end_time = time.time()
        with fps_mutex:
            stream_fps = float(self.frame_count/(end_time - self.start_time))
            self.frame_count = 0
        self.start_time = end_time
        return round(stream_fps, 2)

    def print_data(self):
        print('frame_count=',self.frame_count)
        print('start_time=',self.start_time)


class PERF_DATA:
    def __init__(self, model_name="", num_streams=1):
        self.perf_dict = {}
        self.all_stream_fps = {}
        self.model_name = model_name
        self.num_streams = num_streams
        for i in range(num_streams):
            self.all_stream_fps["stream{0}".format(i)] = GETFPS(i)

    def perf_print_callback(self):
        self.perf_dict = {stream_index: stream.get_fps() for (stream_index, stream) in self.all_stream_fps.items()}
        dict_str = json.dumps(self.perf_dict)
        logging.info(f"\nModel name : {self.model_name} **PERF: {dict_str} \n")
        return True

    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_fps()