from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.export(format="onnx")