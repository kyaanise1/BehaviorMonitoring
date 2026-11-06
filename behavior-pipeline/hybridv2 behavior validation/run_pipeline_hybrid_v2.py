"""V2 tracking -> rule-based inactivity/huddling -> CSV for manual validation.

This is a separate validation runner. It does not change run_pipeline.py.
Uses two independent YOLO model instances, as in eval_hybrid_v2.py.
It is NOT a production/hardware-optimized implementation.
"""
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

from ultralytics import YOLO

from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior import detect_huddles, estimate_body_length, prune_history, update_inactivity


def main():
    ap = argparse.ArgumentParser(description="Hybrid V2 behavior-monitoring validation")
    ap.add_argument("--source", required=True, help="Continuous video or time-aligned sampled image folder")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--window", type=float, default=1800.0, help="Inactivity window in seconds")
    ap.add_argument("--move-frac", type=float, default=0.5)
    ap.add_argument("--min-fill", type=float, default=0.5)
    ap.add_argument("--min-huddle", type=int, default=3)
    ap.add_argument("--log-csv", default="behavior_v2_log.csv")
    args = ap.parse_args()

    if args.interval <= 0 or args.window <= 0:
        ap.error("--interval and --window must be positive")
    if not (0 < args.min_fill <= 1):
        ap.error("--min-fill must be in (0,1]")
    if args.min_huddle < 2:
        ap.error("--min-huddle must be >= 2")
    if not Path(args.source).exists():
        ap.error(f"Source path not found: {args.source}")

    # Two separate models preserve YOLO.track() predictor state, matching the
    # evaluation implementation. Do not replace these with one shared model.
    tracker_model = YOLO(args.weights, task="detect")
    detector_model = YOLO(args.weights, task="detect")
    hybrid = RecoveryHybrid(tight=0.5, wide=4.0, max_missed=4,
                            wide_conf=0.3, size_ratio=2.0,
                            recovery_conf=0.7, recovery_distance=1.5,
                            duplicate_iou=0.5)
    history = defaultdict(deque)
    body_lens = deque(maxlen=500)
    min_samples = max(1, int(args.min_fill * args.window / args.interval))

    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Keep original five columns compatible with eval_behavior.py.
        # Append track-level details to facilitate individual manual review.
        writer.writerow(["t_s", "n_birds", "n_huddles", "n_inactive", "body_len_px",
                         "recovered", "inactive_ids", "huddle_ids"])
        count = 0
        for t, n, frame in sample_frames(args.source, args.interval):
            bt = tracker_model.track(frame, tracker=args.tracker,
                                     conf=args.conf, imgsz=args.imgsz,
                                     persist=True, verbose=False)[0]
            if bt.boxes is None or bt.boxes.id is None:
                byte_ids, byte_boxes, byte_scores = [], [], []
            else:
                byte_ids = bt.boxes.id.int().tolist()
                byte_boxes = bt.boxes.xywh.tolist()
                byte_scores = bt.boxes.conf.tolist()

            raw = detector_model.predict(frame, conf=args.conf,
                                         imgsz=args.imgsz, verbose=False)[0]
            if raw.boxes is None:
                raw_boxes, raw_scores = [], []
            else:
                raw_boxes = raw.boxes.xywh.tolist()
                raw_scores = raw.boxes.conf.tolist()

            ids, boxes, scores, recovered = hybrid.update(
                byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores)
            for tid, (cx, cy, w, h) in zip(ids, boxes):
                history[tid].append((t, cx, cy))
            prune_history(history, t, args.window, args.interval)

            if boxes:
                body_lens.append(estimate_body_length([(b[2], b[3]) for b in boxes]))
            body_len = sorted(body_lens)[len(body_lens) // 2] if body_lens else None
            huddles, inactive = [], []
            if body_len:
                huddles = detect_huddles([(b[0], b[1]) for b in boxes],
                                         body_len, args.min_huddle)
                inactive = update_inactivity(history, t, args.window,
                                             args.move_frac * body_len, min_samples)

            huddle_ids = ["|".join(map(str, sorted(ids[k] for k in cluster)))
                          for cluster in huddles]
            writer.writerow([f"{t:.3f}", len(ids), len(huddles), len(inactive),
                             f"{body_len:.2f}" if body_len else "", recovered,
                             "|".join(map(str, sorted(inactive))), ";".join(huddle_ids)])
            count += 1
            if n % 30 == 0 or inactive:
                print(f"[{t:8.1f}s] frame={n}, birds={len(ids)}, "
                      f"huddles={len(huddles)}, inactive={len(inactive)}, "
                      f"recovered={recovered}")
    print(f"Saved {args.log_csv} ({count} frames, {hybrid.recovered_total} recovered detections)")
    print("NOTE: 'inactive' means centroid immobility under the rule, NOT a health diagnosis.")
    print("Validate each reported episode against independently annotated footage.")


if __name__ == "__main__":
    main()
