# Football Analytics CV

Computer vision pipeline for football match analysis using YOLOv8 + ByteTrack + OpenCV.

## Metrics Tracked
- Player speed (km/h)
- Distance covered (meters)
- Team possession %
- Player heatmaps
- Pass / event detection

## Stack
- YOLOv8 (Ultralytics) — detection + tracking
- OpenCV — perspective transform, heatmaps, annotation
- NumPy — metric calculations

## Setup
pip install ultralytics opencv-python numpy

## Usage
# Step 1 — pick pitch corners for your video
python get_points.py

# Step 2 — run the full pipeline
python main.py