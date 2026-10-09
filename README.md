# Models Directory

This folder will store your trained AI model weights file (for example, `fire_smoke_model.pt` or `best.pt`).

### Important Note on Models and Classes:
- Standard YOLOv8 (`yolov8n.pt`) is trained on the COCO dataset, which detects common everyday objects (person, car, dog, chair), **NOT** fire or smoke.
- For a real detection system, we will use a YOLO model weights file specifically trained to detect:
  1. `fire`
  2. `smoke`
- We will inspect and verify the model's `.names` (classes) in code to guarantee authentic predictions with no fake or hardcoded results.
