from ultralytics import YOLO

model = YOLO("runs/detect/runs/train/powerlines_anomaly-4/weights/best.pt")
results = model("dataset/images/test/07_1035.jpg")
results[0].save("prediction_example.jpg")
print("Done")