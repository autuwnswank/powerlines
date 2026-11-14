import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # лёгкий бэкбон для edge

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")
print(torch.cuda.is_available())

def main():
    model.train(
        data="configs/mapping.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/train",
        name="powerlines_anomaly",
    )

if __name__ == "__main__":
    main()