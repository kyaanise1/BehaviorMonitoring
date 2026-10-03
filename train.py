"""Train, evaluate and export YOLOv11n for single-class broiler detection.
Behavior (inactivity/huddling) is derived downstream via ByteTrack + rules.
"""
from ultralytics import YOLO

DATA = "dataset/dataset.yaml"

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA,
        single_cls=True,
        epochs=100,
        patience=10,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        cos_lr=True,
        fliplr=0.5, flipud=0.5,
        mosaic=1.0, scale=0.2,
        seed=0,
        device=0,
        project=r"C:/Acads/4th Year/1st Semester/Thesis/Behavior Monitoring/runs", 
        name="broiler_yolo11n",
    )

    best = YOLO("C:/Acads/4th Year/1st Semester/Thesis/Behavior Monitoring/runs/broiler_yolo11n/weights/best.pt")    
    m = best.val(data=DATA, split="test")
    print("mAP@0.5:", m.box.map50, "mAP@0.5:0.95:", m.box.map)
    print("Precision:", m.box.mp, "Recall:", m.box.mr)

    best.export(format="onnx", imgsz=640, simplify=True)
    best.export(format="ncnn", imgsz=640)

if __name__ == "__main__":
    main()