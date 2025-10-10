from src.models.input_data import InputData
from src.pipelines.new_pipeline import DSL_Pipeline
from kafka import KafkaConsumer
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
    input_data.add_source(uri_source=extra_information["path"], cam_id=1, extra_information=extra_information)
    pipeline = DSL_Pipeline(input_srcs=input_data.get_src())
    if input_data.get_size() == 1 :
        pipeline.run_pipeline(input_data)
    input_data = InputData()