from kafka import KafkaProducer
import json
import time
import random

def mock_produce_messages():
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8')  # Serialize key thành bytes
    )

    # Tạo vài mock messages (5 camera)
    mock_messages = [
        {
            "path": f"/workspace/FallDetection/streams/yawn.mp4",
            "meta": {
                "cam_id": f"cam_1",  # thêm cam_id vào metadata
                "timestamp": time.time()
            }
            
        },
        {
            "path": f"/workspace/FallDetection/streams/yawn2.mp4",
            "meta": {
                "cam_id": f"cam_2",  # thêm cam_id vào metadata
                "timestamp": time.time()
            }
        },
        {
            "path": f"/workspace/FallDetection/streams/yawn.mp4",
            "meta": {
                "cam_id": f"cam_2",  # thêm cam_id vào metadata
                "timestamp": time.time()
            }
        },
        {
            "path": f"/workspace/FallDetection/streams/yawn2.mp4",
            "meta": {
                "cam_id": f"cam_2",  # thêm cam_id vào metadata
                "timestamp": time.time()
            }
        },
    ]

    # Gửi từng message
    for msg in mock_messages:
        cam_id = msg["meta"]["cam_id"]  # lấy key theo cam_id
        producer.send('inference-input', key=cam_id, value=msg)
        print(f"Sent message with key={cam_id}: {msg}")
        time.sleep(1)  # pause 1s để dễ quan sát

    producer.flush()
    print("All mock messages sent.")

if __name__ == "__main__":
    mock_produce_messages()
