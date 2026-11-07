from ultralytics import YOLO

# Load the second trained model
model = YOLO("runs/broiler_yolo11n-2/weights/best.pt")

# Export to ONNX
model.export(format="onnx", imgsz=640, simplify=True)

# Export to NCNN
model.export(format="ncnn", imgsz=640)

print("Second model exported successfully!")