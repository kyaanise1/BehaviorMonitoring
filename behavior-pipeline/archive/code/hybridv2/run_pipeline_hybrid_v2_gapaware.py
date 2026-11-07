"""Hybrid V2 offline behavior validation that excludes frozen/missing video.

Place this script and gap_guard.py in 'hybridv2 behavior validation'.
Run from behavior-pipeline with PYTHONPATH set to the project root.
This does not change original Hybrid V2 or its scoring files.
"""
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

from ultralytics import YOLO
from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior import detect_huddles, estimate_body_length, prune_history, update_inactivity
from gap_guard import iter_guarded_samples, parse_exclusions


def new_hybrid():
    return RecoveryHybrid(tight=0.5, wide=4.0, max_missed=4, wide_conf=0.3,
                          size_ratio=2.0, recovery_conf=0.7,
                          recovery_distance=1.5, duplicate_iou=0.5)


def main():
    ap = argparse.ArgumentParser(description="Hybrid V2 behavior monitoring with gap detection")
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--window", type=float, default=1800.0)
    ap.add_argument("--move-frac", type=float, default=0.5)
    ap.add_argument("--min-fill", type=float, default=0.5)
    ap.add_argument("--min-huddle", type=int, default=3)
    ap.add_argument("--log-csv", default="behavior_gapaware.csv")
    ap.add_argument("--freeze-frames", type=int, default=3,
                    help="Consecutive near-identical sampled images that qualify as a freeze")
    ap.add_argument("--freeze-diff", type=float, default=0.25,
                    help="Maximum mean gray difference on downsampled images (0..255)")
    ap.add_argument("--max-time-step", type=float, default=6.0,
                    help="Time jump threshold in seconds between sampled frames")
    ap.add_argument("--warmup-frames", type=int, default=3,
                    help="Samples after a gap to rebuild tracking before behavior is valid")
    ap.add_argument("--min-visible", type=int, default=1,
                    help="If fewer birds are tracked, treat frame as unobservable (0 disables)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Known invalid video-time interval, e.g. 00:41:02-00:46:19; repeatable")
    args = ap.parse_args()

    if args.interval <= 0 or args.window <= 0 or args.freeze_frames < 2:
        ap.error("interval/window must be positive and freeze-frames >= 2")
    if not 0 < args.min_fill <= 1:
        ap.error("min-fill must be in (0,1]")
    if args.max_time_step <= args.interval or args.warmup_frames < 0 or args.min_visible < 0:
        ap.error("Check max-time-step, warmup-frames and min-visible")
    if not Path(args.source).exists():
        ap.error(f"Source not found: {args.source}")
    try:
        exclusions = parse_exclusions(args.exclude)
    except ValueError as exc:
        ap.error(str(exc))

    if Path(args.source).is_dir():
        print("WARNING: Image-folder timestamps are synthetic. Cannot detect missing"
              " periods between files unless manually specified with --exclude.")

    # Separate models reproduce the previously validated Hybrid V2 behavior runner.
    # This remains an offline two-inference prototype.
    tracker_model = YOLO(args.weights, task="detect")
    detector_model = YOLO(args.weights, task="detect")
    hybrid = new_hybrid()
    history = defaultdict(deque)
    body_lengths = deque(maxlen=500)
    min_samples = max(1, int(args.min_fill * args.window / args.interval))
    need_reset = False
    in_gap = False
    warmup_left = 0
    counts = {"VALID":0, "GAP":0, "RECOVERING":0}
    reasons = {}
    recoveries = 0

    def output(writer, t, status, reason="", ids=(), boxes=(), recovered=0,
               huddles=(), inactive=(), body_len=None):
        # Missing / recovering data MUST have blank behavior counts, not zero.
        if status != "VALID":
            writer.writerow([f"{t:.3f}", status, reason, "", "", "", "",
                             "", "", "", ""])
            return
        group_ids = ["|".join(map(str, sorted(ids[k] for k in group)))
                     for group in huddles]
        writer.writerow([f"{t:.3f}", status, reason, len(ids), len(huddles),
                         len(inactive), f"{body_len:.2f}" if body_len else "",
                         recovered, "|".join(map(str, sorted(inactive))),
                         ";".join(group_ids), len(history)])

    samples = sample_frames(args.source, args.interval)
    guarded = iter_guarded_samples(samples, interval=args.interval,
                                   freeze_frames=args.freeze_frames,
                                   freeze_diff=args.freeze_diff,
                                   max_step=args.max_time_step,
                                   exclusions=exclusions)
    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "status", "reason", "n_birds", "n_huddles",
                         "n_inactive", "body_len_px", "recovered", "inactive_ids",
                         "huddle_ids", "tracked_histories"])
        for item in guarded:
            t = item.time
            if item.status == "GAP":
                if not in_gap:
                    print(f"[{t:8.1f}s] GAP begins: {item.reason}; tracking histories invalidated")
                in_gap = True
                need_reset = True
                counts["GAP"] += 1
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
                output(writer, t, "GAP", item.reason)
                continue

            if need_reset:
                # Recreate the tracking model so its internal ByteTrack state
                # cannot silently bridge a missing/frozen video interval.
                del tracker_model
                tracker_model = YOLO(args.weights, task="detect")
                hybrid = new_hybrid()
                history.clear()
                body_lengths.clear()
                warmup_left = args.warmup_frames
                need_reset = False
                print(f"[{t:8.1f}s] Valid footage resumed: tracker/history reset")
            in_gap = False

            bt = tracker_model.track(item.frame, tracker=args.tracker,
                                     conf=args.conf, imgsz=args.imgsz,
                                     persist=True, verbose=False)[0]
            if bt.boxes is None or bt.boxes.id is None:
                byte_ids, byte_boxes, byte_scores = [], [], []
            else:
                byte_ids = bt.boxes.id.int().tolist()
                byte_boxes = bt.boxes.xywh.tolist()
                byte_scores = bt.boxes.conf.tolist()

            raw = detector_model.predict(item.frame, conf=args.conf,
                                         imgsz=args.imgsz, verbose=False)[0]
            if raw.boxes is None:
                raw_boxes, raw_scores = [], []
            else:
                raw_boxes = raw.boxes.xywh.tolist()
                raw_scores = raw.boxes.conf.tolist()
            ids, boxes, scores, recovered = hybrid.update(
                byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores)
            recoveries += recovered

            if len(ids) < args.min_visible:
                counts["GAP"] += 1
                reasons["too_few_tracked_birds"] = reasons.get("too_few_tracked_birds",0)+1
                output(writer, t, "GAP", "too_few_tracked_birds")
                need_reset = True
                in_gap = True
                print(f"[{t:8.1f}s] GAP: only {len(ids)} tracked birds")
                continue

            for tid, (cx,cy,w,h) in zip(ids, boxes):
                history[tid].append((t,cx,cy))
            prune_history(history, t, args.window, args.interval)
            if boxes:
                body_lengths.append(estimate_body_length([(b[2],b[3]) for b in boxes]))
            body_len = sorted(body_lengths)[len(body_lengths)//2] if body_lengths else None

            if warmup_left > 0:
                warmup_left -= 1
                counts["RECOVERING"] += 1
                output(writer, t, "RECOVERING", "tracker_warmup")
                continue

            huddles, inactive = [], []
            if body_len:
                huddles = detect_huddles([(b[0],b[1]) for b in boxes],
                                         body_len, args.min_huddle)
                inactive = update_inactivity(history, t, args.window,
                                             args.move_frac*body_len, min_samples)
            counts["VALID"] += 1
            output(writer, t, "VALID", ids=ids, boxes=boxes,
                   recovered=recovered, huddles=huddles, inactive=inactive,
                   body_len=body_len)
            if item.number is not None and (item.number % 100 == 0 or inactive):
                print(f"[{t:8.1f}s] frame={item.number}, birds={len(ids)}, "
                      f"huddles={len(huddles)}, inactive={len(inactive)}")

    print(f"Saved {args.log_csv}")
    print("Sample counts:", counts, "| Gap reasons:", reasons,
          "| recovered detections:", recoveries)
    print("GAP/RECOVERING means UNKNOWN behavior, not absence of behavior.")
    print("Frozen-frame detection is heuristic; review flagged intervals and camera timestamps.")


if __name__ == "__main__":
    main()
