from src.models.input_data import InputData
from src.pipelines.new_pipeline import DSL_Pipeline
from kafka import KafkaConsumer
from loguru import logger
from datetime import datetime
import os
import json

input_data = InputData()

consumer = KafkaConsumer(
    'inference-input',
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='inference-group',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

for msg in consumer:
    extra_information = msg.value
    video_url = extra_information.get("path")
    cam_id = extra_information.get("meta", {}).get("cam_id", "unknown")  # 👈 lấy cam_id từ message
    
    logger.info(f"Received from partition={msg.partition}, offset={msg.offset}")
    logger.info(f"Video path: {video_url}, cam_id: {cam_id}")
    if video_url.split(".")[-1] == "tmp":
        continue

    if not os.path.exists(video_url):
        logger.warning(f"File not exist: {video_url}")
        continue

    video_url = "file://" + video_url

    input_data.add_source(uri_source=video_url, cam_id=cam_id, extra_information=extra_information)

    pipeline = DSL_Pipeline(input_srcs=input_data.get_src())
    if input_data.get_size() == 4:
        pipeline.run_pipeline(input_data)
        input_data = InputData()
