from ultralytics import YOLO
# Load the exported TensorRT model
model = YOLO("yolov8n-seg.onnx")

# Run inference on an image
results = model.predict("th_roads_780x439.jpg", save=True)  # results list

# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks