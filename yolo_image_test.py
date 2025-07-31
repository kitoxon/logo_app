import cv2
from ultralytics import YOLO
import easyocr
import os
import json
from pathlib import Path
import numpy as np

# 🧠 Load YOLOv8 model (your trained model)
yolo_model = YOLO("rb_sns.pt")
print("✅ Model classes:", yolo_model.names)

# 📖 Load EasyOCR
reader = easyocr.Reader(["ja", "en"])

# 📁 Input folder with images
image_folder = Path("test_images")  # 🔁 Put your images here
output_dir = Path("output_images_rb_sns")
output_dir.mkdir(exist_ok=True)

results_data = []

image_files = sorted([f for f in image_folder.glob("*.jpeg")])  # Change to *.png if needed

print(f"🚀 Found {len(image_files)} images. Starting processing...")

for idx, image_path in enumerate(image_files):
    frame = cv2.imread(str(image_path))
    if frame is None:
        continue

    # 🎯 YOLO logo detection
    detections = yolo_model(frame)[0]

    frame_result = {
        "filename": image_path.name,
        "logos": [],
        "texts": []
    }

    for det in detections.boxes:
        xyxy = det.xyxy[0].tolist()
        conf = float(det.conf[0])
        cls = int(det.cls[0])
        label = yolo_model.names[cls]

        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        frame_result["logos"].append({
            "label": label,
            "confidence": conf,
            "bbox": xyxy
        })

    # 🔤 EasyOCR text detection
    ocr_results = reader.readtext(frame)
    for (bbox, text, conf) in ocr_results:
        frame_result["texts"].append({
            "text": text,
            "confidence": conf,
            "bbox": bbox
        })

    # 💾 Save annotated image
    output_file = output_dir / image_path.name
    cv2.imwrite(str(output_file), frame)

    results_data.append(frame_result)

# Utility to handle NumPy types
def convert_numpy(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)

# Save results as JSON
with open("logo_text_image_results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2, default=convert_numpy)

print("✅ Done. Results saved to 'output_images_rb/' and 'logo_text_image_results.json'")
