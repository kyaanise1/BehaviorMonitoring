# REVIEW COPY (original run_pipeline_hybrid_v2.py remains unchanged)

"""Non-gap-aware Hybrid V2 tracking -> inactivity/huddling CSV + optional ID review video.

This is a separate validation runner. It does not change run_pipeline.py.
Uses two independent YOLO model instances, as in eval_hybrid_v2.py.
It is NOT a production/hardware-optimized implementation.

Review output is sampled at the SAME rate as the tracker (e.g. every 2 seconds).
Always rerun with the identical source/weights/tracker/interval/settings that
produced the original CSV if you want matching IDs. The video writer is only a
visualization; it does not change any YOLO/Hybrid/behavior decisions.

Example from behavior-pipeline:
    python ".\\hybridv2 behavior validation\\run_pipeline_hybrid_v2_review.py" \
      --source "../video/Day12_repaired.mp4" \
      --weights "../runs/broiler_yolo11n/weights/best.pt" \
      --tracker "configs/bytetrack_broiler.yaml" \
      --interval 2 --window 1800 \
      --review-id 15 --review-video "day12_id15_review.mp4" \
      --review-start 0 --review-end 2100 --log-csv "behavior_day12test2_replay.csv"

"""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ultralytics import YOLO

from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior import detect_huddles, estimate_body_length, prune_history, update_inactivity


def _hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{(t // 60) % 60:02d}:{t % 60:02d}"


def _draw_review_frame(frame, t, sample_number, review_id, ids, boxes,
                       inactive, body_len, max_width, trail):
    """Add a review header OUTSIDE the source image, never cover the source."""
    import cv2

    ih, iw = frame.shape[:2]
    scale = min(1.0, max_width / iw)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(iw * scale), round(ih * scale)))
    else:
        frame = frame.copy()
    height, width = frame.shape[:2]

    match = next(((cx, cy, bw, bh) for tid, (cx, cy, bw, bh)
                  in zip(ids, boxes) if int(tid) == review_id), None)
    # A missing ID is not evidence that the physical bird has moved.
    # Never bridge long non-observation gaps with an artificial trajectory line.
    if match is None:
        if trail and t - trail[-1][0] > 10:
            trail.clear()
        state = 'NOT DETECTED IN SAMPLED FRAME'
        pos = 'No box/centroid for this ID at this timestamp'
    else:
        cx, cy, bw, bh = (float(v) * scale for v in match)
        x1 = max(0, round(cx - bw / 2))
        y1 = max(0, round(cy - bh / 2))
        x2 = min(width - 1, round(cx + bw / 2))
        y2 = min(height - 1, round(cy + bh / 2))
        color = (0, 255, 255) if review_id in inactive else (0, 210, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.circle(frame, (round(cx), round(cy)), 5, (255, 0, 255), -1)
        if trail and t - trail[-1][0] > 10:
            trail.clear()
        trail.append((t, round(cx), round(cy)))
        # Show up to the last 60 seconds of sampled movement (not future/predicted positions).
        while trail and t - trail[0][0] > 60:
            trail.popleft()
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(frame, (a[1], a[2]), (b[1], b[2]), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'ID {review_id}', (x1, max(18, y1 - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
        state = 'INACTIVE (system)' if review_id in inactive else 'TRACKED / NOT INACTIVE'
        pos = f'Centroid ({match[0]:.1f}, {match[1]:.1f})   Box ({match[2]:.1f} x {match[3]:.1f})'

    hh = 122
    canvas = cv2.copyMakeBorder(frame, hh, 0, 0, 0,
                                cv2.BORDER_CONSTANT, value=(25, 25, 25))
    labels = [
        f'NON-GAP-AWARE HYBRID V2 | ID {review_id} | VIDEO TIME {_hms(t)} | Sample {sample_number}',
        f'State: {state}',
        pos,
        f'30-minute rule: checked from history | Body length: {body_len:.1f}px' if body_len else
        '30-minute rule: checked from history | Body length: unknown',
    ]
    for y, line in zip((24, 50, 78, 105), labels):
        # Keep text legible on resized frames; the 1280px default fits typical lines.
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, match is not None


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
    ap.add_argument("--clusters-csv", default="", help="Export per-cluster bounding boxes to CSV")
    ap.add_argument("--timestamps-csv", default="", help="Annotation CSV with t_s column; export only these timestamps")
    ap.add_argument("--log-csv", default="behavior_v2_log.csv")
    ap.add_argument("--review-id", type=int, default=None,
                    help="Track ID to highlight from THIS non-gap-aware run")
    ap.add_argument("--review-video", default="",
                    help="MP4 output path; requires --review-id")
    ap.add_argument("--review-start", type=float, default=0.0,
                    help="Show only frames after this source video timestamp (seconds); tracking still starts at 0")
    ap.add_argument("--review-end", type=float, default=float('inf'),
                    help="Show frames until this source video timestamp (seconds); tracking still begins at 0")
    ap.add_argument("--review-fps", type=float, default=5.0,
                    help="Review-video playback frames/second; 5 fps with 2-s samples is 10x speed")
    ap.add_argument("--review-width", type=int, default=1280,
                    help="Maximum review-video frame width, without upscaling")
    args = ap.parse_args()

    if args.interval <= 0 or args.window <= 0:
        ap.error("--interval and --window must be positive")
    if not (0 < args.min_fill <= 1):
        ap.error("--min-fill must be in (0,1]")
    if args.min_huddle < 2:
        ap.error("--min-huddle must be >= 2")
    if args.review_video and args.review_id is None:
        ap.error("--review-video requires --review-id")
    if args.review_video and (args.review_fps <= 0 or args.review_width < 480):
        ap.error("--review-fps must be positive and --review-width >= 480")
    if args.review_start < 0 or args.review_end <= args.review_start:
        ap.error("review timestamps must obey 0 <= --review-start < --review-end")
    if args.review_video and Path(args.review_video).resolve() == Path(args.source).resolve():
        ap.error("Review video must not overwrite the source video")
    if args.review_video and Path(args.review_video).resolve() == Path(args.log_csv).resolve():
        ap.error("Review video must not overwrite the log CSV")
    if not Path(args.source).exists():
        ap.error(f"Source path not found: {args.source}")

    selected_times = None
    if args.timestamps_csv:
        with open(args.timestamps_csv, newline="", encoding="utf-8-sig") as tf:
            selected_times = [float(row["t_s"]) for row in csv.DictReader(tf)]
        if not selected_times:
            ap.error("No timestamps found in --timestamps-csv")
    if args.timestamps_csv and not args.clusters_csv:
        ap.error("--timestamps-csv requires --clusters-csv")

    # If using review-video mode, cv2 is imported only for video output.
    cv2 = None
    if args.review_video:
        import cv2
        if Path(args.review_video).suffix.lower() != '.mp4':
            ap.error("--review-video output must end with .mp4")
        Path(args.review_video).parent.mkdir(parents=True, exist_ok=True)

    cluster_file = None
    cluster_writer = None
    if args.clusters_csv:
        cluster_file = open(args.clusters_csv, "w", newline="", encoding="utf-8")
        cluster_writer = csv.writer(cluster_file)
        cluster_writer.writerow(["t_s", "frame_no", "cluster_no", "track_ids",
                                 "member_boxes_xywh_json", "cluster_center_x",
                                 "cluster_center_y", "n_members", "n_clusters_in_frame"])

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

    review_writer = None
    review_frames = 0
    review_visible = 0
    review_trail = deque(maxlen=90)

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
            if cluster_writer is not None and (
                selected_times is None or any(abs(t - target) < 0.01 for target in selected_times)
            ):
                if not huddles:
                    cluster_writer.writerow([f"{t:.3f}", n, "", "", "[]", "", "", 0, 0])
                for cluster_no, cluster in enumerate(huddles, start=1):
                    members = [
                        {"track_id": int(ids[k]),
                         "cx": round(float(boxes[k][0]), 2),
                         "cy": round(float(boxes[k][1]), 2),
                         "w": round(float(boxes[k][2]), 2),
                         "h": round(float(boxes[k][3]), 2)}
                        for k in cluster
                    ]
                    cluster_writer.writerow([
                        f"{t:.3f}", n, cluster_no,
                        "|".join(str(m["track_id"]) for m in members),
                        json.dumps(members, separators=(",", ":")),
                        round(sum(m["cx"] for m in members) / len(members), 2),
                        round(sum(m["cy"] for m in members) / len(members), 2),
                        len(members), len(huddles)
                    ])
            if args.review_video and args.review_start <= t <= args.review_end:
                annotated, found = _draw_review_frame(
                    frame, t, n, args.review_id, ids, boxes, inactive,
                    body_len, args.review_width, review_trail)
                if review_writer is None:
                    ah, aw = annotated.shape[:2]
                    review_writer = cv2.VideoWriter(
                        str(args.review_video), cv2.VideoWriter_fourcc(*'mp4v'),
                        args.review_fps, (aw, ah))
                    if not review_writer.isOpened():
                        raise RuntimeError(f"Cannot write review video: {args.review_video}")
                review_writer.write(annotated)
                review_frames += 1
                review_visible += int(found)
            count += 1
            if n % 30 == 0 or inactive:
                print(f"[{t:8.1f}s] frame={n}, birds={len(ids)}, "
                      f"huddles={len(huddles)}, inactive={len(inactive)}, "
                      f"recovered={recovered}")
    if review_writer is not None:
        review_writer.release()
    if args.review_video:
        if review_frames == 0:
            print('WARNING: No frames inside requested --review-start/--review-end range')
        else:
            print(f'Saved review video: {args.review_video} '
                  f'({review_frames} sampled frames; selected ID present in {review_visible})')
            if review_visible == 0:
                print('WARNING: This track ID was never seen. Check that source and tracker settings '
                      'match the original non-gap-aware run.')
    if cluster_file is not None:
        cluster_file.close()
        print(f"Saved cluster details to {args.clusters_csv}")
    print(f"Saved {args.log_csv} ({count} frames, {hybrid.recovered_total} recovered detections)")
    print("NOTE: 'inactive' means centroid immobility under the rule, NOT a health diagnosis.")
    print("Validate each reported episode against independently annotated footage.")


if __name__ == "__main__":
    main()
