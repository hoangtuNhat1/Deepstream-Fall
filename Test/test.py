import requests
import numpy as np
import cv2
import json


# ==========================================================
# Utility functions
# ==========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


def nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area1 + area2 - inter)

        idxs = idxs[1:][iou < iou_threshold]

    return keep


# ==========================================================
# YOLO Post-process
# ==========================================================

def yolo_postprocess(output, img_w, img_h, conf_thres=0.4, iou_thres=0.5):
    """
    output shape: (1, 10, 8400)
    Format: [x, y, w, h, obj, cls1..cls5]
    """
    output = output[0]  # (10, 8400)
    output = output.transpose(1, 0)  # (8400, 10)

    # Apply sigmoid to all values
    output = sigmoid(output)

    # Confidence computation
    obj = output[:, 4]
    cls_scores = output[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]

    final_conf = obj * cls_conf

    # Filter by confidence threshold
    mask = final_conf > conf_thres
    output = output[mask]
    cls_ids = cls_ids[mask]
    final_conf = final_conf[mask]

    if len(output) == 0:
        return [], [], []

    # Convert to xyxy
    boxes = xywh2xyxy(output[:, :4])

    # Scale back to image size
    boxes[:, [0, 2]] *= img_w
    boxes[:, [1, 3]] *= img_h

    # NMS
    keep = nms(boxes, final_conf, iou_threshold=iou_thres)

    return boxes[keep], final_conf[keep], cls_ids[keep]


# ==========================================================
# MAIN
# ==========================================================

def main():
    url = "http://localhost:8000/v2/models/sleep_yolo/infer"

    # Load image
    img = cv2.imread("images.jpeg")
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0

    # Convert to NCHW
    img_np = np.transpose(img_np, (2, 0, 1))[None, ...]

    payload = {
        "inputs": [
            {
                "name": "images",
                "datatype": "FP32",
                "shape": img_np.shape,
                "data": img_np.flatten().tolist()
            }
        ]
    }

    # Send request
    response = requests.post(url, json=payload)
    resp = response.json()

    # Extract output
    raw = np.array(resp["outputs"][0]["data"], dtype=np.float32)
    out = raw.reshape(1, 10, 8400)

    print("Model output shape:", out.shape)

    # Post-process
    h, w = img.shape[:2]
    boxes, scores, classes = yolo_postprocess(out, w, h)

    print("Detected:", len(boxes))

    # Draw boxes
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img, f"{c}:{s:.2f}", (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.imwrite("result.jpg", img)
    print("Saved result.jpg")


if __name__ == "__main__":
    main()
