from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # лёгкий бэкбон для edge
model.train(
    data="configs/ttpla.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="lep_anomaly"
)