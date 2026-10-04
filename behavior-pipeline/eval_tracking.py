"""Tracking evaluation (MOTA, MOTP, IDF1, ID switches).

pip install motmetrics

Step 1: annotate a short clip with consistent IDs (e.g. in CVAT, export "MOT 1.1").
        Annotate the SAME sampled frames the tracker sees (same --interval,
        1-based frame numbers), so gt.txt frame N == sample N.
Step 2: generate predictions with the exact tracker settings under test:
        python eval_tracking.py predict --source clip.mkv --out pred.txt
        python eval_tracking.py predict --source clip.mkv --tracker-type centroid --out pred_centroid.txt
Step 3: score:
        python eval_tracking.py score --gt gt.txt --pred pred.txt

Ablation (track_buffer): copy configs/bytetrack_broiler.yaml, change track_buffer,
run `predict --tracker <new yaml>` and `score` again for each value.
"""
import argparse

import numpy as np

# motmetrics still calls np.asfarray, which NumPy 2.0 removed
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

    
from tracker import add_tracker_args, make_runner


def predict(args):
    from ultralytics import YOLO
    from common import sample_frames

    model = YOLO(args.weights, task="detect")
    track = make_runner(args, model)          # ByteTrack or centroid, per --tracker-type
    with open(args.out, "w") as f:
        for t, n, frame in sample_frames(args.source, args.interval):
            ids, xywh, scores = track(frame)
            for tid, (cx, cy, w, h), s in zip(ids, xywh, scores):
                # MOT format: frame,id,x,y,w,h,conf,-1,-1,-1  (x,y = top-left)
                f.write(f"{n},{tid},{cx - w / 2:.2f},{cy - h / 2:.2f},{w:.2f},{h:.2f},{s:.3f},-1,-1,-1\n")
    print("Saved", args.out)


def _rows(df, frame):
    try:
        sub = df.loc[frame]
    except KeyError:
        return [], np.empty((0, 4))
    return sub.index.tolist(), sub[["X", "Y", "Width", "Height"]].values


def score(args):
    import motmetrics as mm

    gt = mm.io.loadtxt(args.gt, fmt="mot15-2D")
    pr = mm.io.loadtxt(args.pred, fmt="mot15-2D")

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.index.get_level_values(0)) | set(pr.index.get_level_values(0)))
    for fr in frames:
        gids, gboxes = _rows(gt, fr)
        pids, pboxes = _rows(pr, fr)
        # distance = 1 - IoU; pairs with IoU < iou_thr are not allowed to match
        dist = mm.distances.iou_matrix(gboxes, pboxes, max_iou=1 - args.iou_thr)
        acc.update(gids, pids, dist)

    mh = mm.metrics.create()
    s = mh.compute(acc, name="clip", metrics=[
        "mota", "motp", "idf1", "idp", "idr",
        "num_switches", "num_misses", "num_false_positives", "num_objects"])
    s["motp"] = 1 - s["motp"]        # motmetrics reports mean (1 - IoU); convert to mean IoU
    print(mm.io.render_summary(s, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    print(f"\nTargets from the proposal: MOTA >= 0.70, IDF1 >= 0.75, MOTP >= 0.75 (mean IoU)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="pred.txt")
    p.add_argument("--weights", default="runs/broiler_yolo11n/weights/best.pt")
    p.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=640)
    add_tracker_args(p)
    p.set_defaults(fn=predict)

    s = sub.add_parser("score")
    s.add_argument("--gt", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--iou-thr", type=float, default=0.5)
    s.set_defaults(fn=score)

    a = ap.parse_args()
    a.fn(a)