from src.models.input_data import InputData
from src.pipelines.new_pipeline import DSL_Pipeline

input_data = InputData()


video_url = ["file:///workspace/FallDetection/no.mp4" for _ in range(2)]
cam_id = [0, 1]
extra_information = []
for i in range(1) : 
    input_data.add_source(uri_source=video_url[i], cam_id=cam_id, extra_information=extra_information)
    pipeline = DSL_Pipeline(input_srcs=input_data.get_src())
pipeline.run_pipeline(input_data)

