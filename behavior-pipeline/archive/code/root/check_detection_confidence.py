
from pathlib import Path
from ultralytics import YOLO

model = YOLO("../runs/broiler_yolo11n/weights/best.pt")

images = sorted(Path("tracking_frames").glob("*.jpg"))

print("Frame | Total | 0.10-0.60 | 0.60-0.70 | >=0.70")
print("-" * 54)

for frame in range(4, 10):
    result = model.predict(
        str(images[frame - 1]),
        conf=0.1,
        imgsz=640,
        verbose=False
    )[0]

    scores = result.boxes.conf.cpu().tolist()

    low = sum(0.1 <= s < 0.6 for s in scores)
    middle = sum(0.6 <= s < 0.7 for s in scores)
    high = sum(s >= 0.7 for s in scores)

    print(
        f"{frame:5d} | {len(scores):5d} | "
        f"{low:9d} | {middle:9d} | {high:6d}"
    )
