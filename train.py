"""Train, evaluate and export YOLOv11n for broiler behavior detection.
pip install ultralytics
"""
from ultralytics import YOLO

DATA = "dataset/dataset.yaml"

model = YOLO("yolo11n.pt")          # COCO-pretrained, per proposal
model.train(
    data=DATA,
    epochs=100,
    patience=10,                    # early stopping
    batch=16,
    imgsz=640,
    lr0=0.01,
    cos_lr=True,                    # cosine decay
    fliplr=0.5, flipud=0.5,         # overhead view: flips are safe
    mosaic=1.0, scale=0.2,
    device=0,                       # GPU; use "cpu" if none
    project="runs", name="broiler_yolo11n",
)

best = YOLO("runs/broiler_yolo11n/weights/best.pt")
metrics = best.val(data=DATA, split="test")     # held-out blocks
print("mAP@0.5:", metrics.box.map50)
print("Precision:", metrics.box.mp, "Recall:", metrics.box.mr)

best.export(format="onnx", imgsz=640, simplify=True)
# For the Raspberry Pi, also try: best.export(format="ncnn")  (often faster than ONNX on ARM)