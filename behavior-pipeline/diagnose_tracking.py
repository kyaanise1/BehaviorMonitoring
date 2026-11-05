
from pathlib import Path
from ultralytics import YOLO
import motmetrics as mm

MODEL = "../runs/broiler_yolo11n/weights/best.pt"
FRAMES = Path("tracking_frames")

gt = mm.io.loadtxt(
    "annotated_tracking_frames/gt/gt.txt",
    fmt="mot15-2D"
)
pred = mm.io.loadtxt(
    "pred_hybrid.txt",
    fmt="mot15-2D"
)

model = YOLO(MODEL)
images = sorted(
    list(FRAMES.glob("*.jpg")) +
    list(FRAMES.glob("*.png"))
)

def count_objects(df, frame):
    return int(
        (df.index.get_level_values(0) == frame).sum()
    )

print("\nFRAME | GT | YOLO | HYBRID")
print("-" * 32)

for frame in range(4, 10):
    image = images[frame - 1]

    result = model.predict(
        str(image),
        conf=0.1,
        imgsz=640,
        verbose=False
    )[0]

    yolo_count = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )

    print(
        f"{frame:5d} | "
        f"{count_objects(gt, frame):2d} | "
        f"{yolo_count:4d} | "
        f"{count_objects(pred, frame):6d}"
    )
