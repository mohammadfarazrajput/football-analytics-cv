from ultralytics import YOLO

# ── Load base model ───────────────────────────────────────────────────────────
# YOLOv8 variants: nano → small → medium → large → x (largest / most accurate)
model = YOLO('yolov8x.pt')

# ── Run inference ─────────────────────────────────────────────────────────────
# save=True writes an annotated video to runs/detect/predict/
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# ── Inspect first frame output ────────────────────────────────────────────────
print(results[0])

for box in results[0].boxes:
    print("---")
    print(box)
    # box.cls   → class ID (e.g. 0 = person)
    # box.conf  → confidence score
    # box.xyxy  → [x1, y1, x2, y2] in pixels
    # box.xywh  → [cx, cy, w, h]

# ── After fine-tuning, switch to custom model ─────────────────────────────────
# model = YOLO('models/best.pt')
# results = model.predict('input_videos/08fd33_4.mp4', save=True)
