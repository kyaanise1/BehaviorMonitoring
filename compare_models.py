
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

# Project directory
ROOT = Path(__file__).resolve().parent

# Dataset location
DATA = ROOT / "dataset" / "dataset.yaml"


def main():

    models = {
        "First Run": ROOT / "runs" / "broiler_yolo11n" / "weights" / "best.pt",
        "Second Run": ROOT / "runs" / "broiler_yolo11n-2" / "weights" / "best.pt"
    }

    results = {}

    for name, path in models.items():

        print(f"\n{'=' * 45}")
        print(f"Evaluating: {name}")
        print(f"Model: {path}")
        print(f"{'=' * 45}")

        if not path.exists():
            print(f"Model not found: {path}")
            continue

        model = YOLO(str(path))

        metrics = model.val(
            data=str(DATA),
            split="test",
            imgsz=640,
            batch=16,
            workers=0,
            plots=False,
            device=0
        )

        results[name] = {
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map
        }

    print("\n========== MODEL COMPARISON ==========")

    for name, values in results.items():
        print(f"\n{name}")
        for metric, value in values.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()
