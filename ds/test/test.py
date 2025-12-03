import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_URL = "triton:8001"  # Thay thế bằng IP/hostname thực tế nếu cần

try:
    # 1. Khởi tạo client gRPC
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print(f"✅ Đang cố gắng kết nối tới Triton tại {TRITON_URL}...")

    # 2. Kiểm tra tính sẵn sàng của Server
    if client.is_server_ready():
        print("✅ Server Triton đã SẴN SÀNG.")
    else:
        print("❌ Server Triton KHÔNG SẴN SÀNG.")
        exit()

    # 3. Kiểm tra tính sẵn sàng của Model "sleep_yolo"
    if client.is_model_ready(model_name="sleep_yolo"):
        print("✅ Model 'sleep_yolo' đã SẴN SÀNG để inference.")
        
        # Lấy thông tin metadata model (để xác nhận đầu vào/đầu ra)
        metadata = client.get_model_metadata(model_name="sleep_yolo")
        print("\n--- Metadata Model ---")
        print(f"Input: {[i['name'] for i in metadata['inputs']]}")
        print(f"Output: {[o['name'] for o in metadata['outputs']]}")
        print("----------------------")
        
        print("Model 'sleep_yolo' đã được kiểm tra và sẵn sàng.")
    else:
        print("❌ Model 'sleep_yolo' KHÔNG SẴN SÀNG (chưa load hoặc lỗi).")

except InferenceServerException as e:
    print(f"❌ Lỗi kết nối hoặc tương tác với Triton: {e}")
    print("Kiểm tra lại địa chỉ URL và đảm bảo Triton Server đang chạy.")
except Exception as e:
    print(f"❌ Đã xảy ra lỗi khác: {e}")