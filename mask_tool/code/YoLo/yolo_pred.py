from ultralytics import YOLO
# Load the exported TensorRT model
model = YOLO("YoLo\\model\\yolov8n-seg.onnx")

# Run inference on an image
results = model.predict("test_source\\CCTV.mp4")  # results list

# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks