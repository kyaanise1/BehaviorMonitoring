
import argparse
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
from eval_tracking import _rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--frames", default="tracking_frames")
    args = parser.parse_args()

    gt = mm.io.loadtxt(args.gt, fmt="mot15-2D")
    pred = mm.io.loadtxt(args.pred, fmt="mot15-2D")

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(
        set(gt.index.get_level_values(0))
        | set(pred.index.get_level_values(0))
    )

    for frame in frames:
        gt_ids, gt_boxes = _rows(gt, frame)
        pred_ids, pred_boxes = _rows(pred, frame)

        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=0.5
        )
        acc.update(gt_ids, pred_ids, distances)

    events = acc.events
    switches = events[events["Type"] == "SWITCH"]

    images = sorted(
        list(Path(args.frames).glob("*.jpg"))
        + list(Path(args.frames).glob("*.png"))
    )

    print(f"\nTotal ID switches: {len(switches)}\n")

    for (frame, event_id), row in switches.iterrows():
        frame = int(frame)
        image = (
            images[frame - 1].name
            if 1 <= frame <= len(images)
            else "Unknown"
        )

        print(
            f"Frame {frame:3d} | "
            f"GT ID: {int(row['OId']):3d} | "
            f"Predicted ID: {int(row['HId']):3d} | "
            f"Image: {image}"
        )


if __name__ == "__main__":
    main()
