"""Try several centroid-tracker settings on the same labeled clip and score each one.

    python sweep_centroid.py --source ".\\tracking_frames" --gt ".\\annotated_tracking_frames\\gt\\gt.txt" ^
        --weights "..\\runs\\broiler_yolo11n\\weights\\best.pt"

Defaults test: wide limit {2, 3, 4, 6} body lengths x max_missed {2, 4, 6} frames.
Find the biggest jump a real bird made between consecutive frames in your clip
and make sure the best --wide value is a bit above it.
"""
import argparse
import os
from types import SimpleNamespace

import eval_tracking as et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of frames")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--weights", default="runs/broiler_yolo11n/weights/best.pt")
    ap.add_argument("--wide", type=float, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--missed", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--tight", type=float, default=0.5)
    ap.add_argument("--start-conf", type=float, default=0.6)
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()

    os.makedirs("configs/sweep", exist_ok=True)
    for wide in a.wide:
        for missed in a.missed:
            tag = f"centroid_w{wide}_m{missed}_t{a.tight}_s{a.start_conf}"
            pred_path = f"configs/sweep/pred_{tag}.txt"
            print(f"\n=== centroid  wide={wide}  max_missed={missed}  "
                  f"tight={a.tight}  start_conf={a.start_conf} ===")
            et.predict(SimpleNamespace(
                source=a.source, out=pred_path, weights=a.weights, tracker="",
                interval=a.interval, conf=0.1, imgsz=640, tracker_type="centroid",
                tight=a.tight, wide=wide, max_missed=missed,
                start_conf=a.start_conf, wide_conf=0.3))
            et.score(SimpleNamespace(gt=a.gt, pred=pred_path, iou_thr=0.5))


if __name__ == "__main__":
    main()