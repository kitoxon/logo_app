
import cv2
from ultralytics import YOLO
import easyocr
import os
import json
from pathlib import Path
import numpy as np
# 🧠 Load YOLOv8 model (change path if using custom-trained logo model)
yolo_model = YOLO("fc_tokyo_away_best16.pt")  # Replace with your custom logo model path if available
print("✅ Model classes:", yolo_model.names)

# 📖 Load EasyOCR
reader = easyocr.Reader(["ja", "en"])  # Use Japanese and English

# 🎞 Load Video
video_path = "/home/munkhjin/Desktop/video_to_image/fc_tokyo_2-15.mp4"  # Replace with your video path
output_dir = Path("output_frames_fc_tokyo")
output_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  # Sample every 2 seconds
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
results_data = []

frame_idx = 0
saved_idx = 0

print("🚀 Starting video processing...")
def compute_area(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    area_pixels = (x2 - x1) * (y2 - y1)
    total_area = frame_width * frame_height
    return area_pixels / total_area  # normalized area

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        timestamp = frame_idx / fps

        # 🎯 Run YOLO for logo detection
        detections = yolo_model(frame)[0]

        frame_result = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "logos": [],
            "texts": []
        }

        for det in detections.boxes:
            xyxy = det.xyxy[0].tolist()
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            label = yolo_model.names[cls]

            # draw rectangle
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            percent_area = compute_area(xyxy, width, height) * 100  # e.g., 1.5%
            frame_result["logos"].append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy,
                "area": percent_area
            })

        # 🔤 Run EasyOCR for text detection
        ocr_results = reader.readtext(frame)
        for (bbox, text, conf) in ocr_results:
            frame_result["texts"].append({
                "text": text,
                "confidence": conf,
                "bbox": bbox
            })

        # 💾 Save annotated frame
        output_file = output_dir / f"frame_{saved_idx}.jpg"
        cv2.imwrite(str(output_file), frame)

        results_data.append(frame_result)
        saved_idx += 1

    frame_idx += 1

cap.release()

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
# Save results as JSON
with open("logo_text_results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2, default=convert_numpy)

print("✅ Done. Results saved to 'output_frames/' and 'logo_text_results.json'")
