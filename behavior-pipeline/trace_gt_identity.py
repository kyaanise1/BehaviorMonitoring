"""Compare how two MOT predictions associate one ground-truth object over time.

Usage (from behavior-pipeline):
    python trace_gt_identity.py --gt annotated_tracking_frames/gt/gt.txt \
        --v2 pred_hybrid_v2.txt --v3 pred_hybrid_v3.txt \
        --gt-id 3 --start 10 --end 16

Uses the same IoU-based one-to-one association as eval_hybrid_tracking.py.
Does not alter tracking predictions or annotations.
"""
import argparse
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm


def rows(df, frame):
    try:
        sub = df.loc[frame]
    except KeyError:
        return [], np.empty((0, 4), dtype=float)
    return sub.index.tolist(), sub[["X", "Y", "Width", "Height"]].values


def events_for(gt, pred, iou_thr):
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.index.get_level_values(0)) | set(pred.index.get_level_values(0)))
    for frame in frames:
        gt_ids, gt_boxes = rows(gt, frame)
        pred_ids, pred_boxes = rows(pred, frame)
        cost = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - iou_thr)
        acc.update(gt_ids, pred_ids, cost)
    result = {}
    switches = set()
    for (fr, _), event in acc.events.iterrows():
        kind = str(event["Type"])
        if kind not in ("MATCH", "SWITCH", "MISS"):
            continue
        if not np.isfinite(event["OId"]):
            continue
        obj_id = int(event["OId"])
        predicted_id = int(event["HId"]) if np.isfinite(event["HId"]) else None
        result[(int(fr), obj_id)] = (kind, predicted_id)
        if kind == "SWITCH":
            switches.add((int(fr), obj_id))
    return result, switches


def label(value):
    if value is None:
        return "-"
    kind, pred_id = value
    return f"{kind:<6} P{pred_id}" if pred_id is not None else "MISS"


def main():
    ap = argparse.ArgumentParser(description="Inspect GT track identity around V2/V3 disagreement")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--gt-id", type=int, default=3)
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--end", type=int, default=16)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    a = ap.parse_args()
    if not (0.0 < a.iou_thr <= 1.0 and a.start <= a.end):
        ap.error("Require 0 < --iou-thr <= 1 and --start <= --end")

    gt = mm.io.loadtxt(a.gt, fmt="mot15-2D")
    v2 = mm.io.loadtxt(a.v2, fmt="mot15-2D")
    v3 = mm.io.loadtxt(a.v3, fmt="mot15-2D")
    e2, s2 = events_for(gt, v2, a.iou_thr)
    e3, s3 = events_for(gt, v3, a.iou_thr)

    print(f"\nGround-truth chicken {a.gt_id}, frames {a.start}–{a.end}")
    print("Event = IoU-based MOT match, switch, or miss; P# is a tracker-specific ID.")
    print("Frame | V2 event / predicted ID | V3 event / predicted ID")
    print("-" * 65)
    for fr in range(a.start, a.end + 1):
        v2info = e2.get((fr, a.gt_id))
        v3info = e3.get((fr, a.gt_id))
        print(f"{fr:5d} | {label(v2info):<23} | {label(v3info)}")

    print("\nSwitch events found only in V3 (frame, GT ID):", sorted(s3 - s2))
    print("Switch events found only in V2 (frame, GT ID):", sorted(s2 - s3))
    print(f"Total switch events: V2={len(s2)}, V3={len(s3)}")
    print("\nNote: P IDs from V2 and V3 are independent. Different ID numbers alone do not indicate an error.")


if __name__ == "__main__":
    main()
