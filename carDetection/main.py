from ultralytics import YOLO

model=YOLO("yolov8n.yaml")

model.train(data="data.yaml",epochs=200)