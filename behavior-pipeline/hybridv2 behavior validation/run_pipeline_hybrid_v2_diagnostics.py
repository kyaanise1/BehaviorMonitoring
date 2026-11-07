"""V2 tracking -> rule-based inactivity/huddling -> CSV for manual validation.

This is a separate validation runner. It does not change run_pipeline.py.
Uses two independent YOLO model instances, as in eval_hybrid_v2.py.
It is NOT a production/hardware-optimized implementation.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

# Allow direct execution from the nested validation folder.
# Shared modules (common.py, behavior.py, hybrid_tracker_v2.py) live one level up.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ultralytics import YOLO

from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior_linkage_test import (
    detect_huddles, detect_huddles_complete_linkage, detect_huddles_average_linkage, estimate_body_length,
    prune_history, update_inactivity
)


def load_feeding_zones(path):
    """Load axis-aligned feeding-zone rectangles in reference-image pixels."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "ref_size" not in config or "zones" not in config:
        raise ValueError("Feeding-zone JSON must contain 'ref_size' and 'zones'.")

    ref_width, ref_height = config["ref_size"]
    if ref_width <= 0 or ref_height <= 0:
        raise ValueError("'ref_size' dimensions must be positive.")

    zones = config["zones"]
    for zone in zones:
        if len(zone) != 4:
            raise ValueError(f"Each feeding zone must be [x1, y1, x2, y2]: {zone}")
        x1, y1, x2, y2 = zone
        if not (0 <= x1 < x2 <= ref_width and 0 <= y1 < y2 <= ref_height):
            raise ValueError(f"Feeding zone is invalid or outside ref_size: {zone}")

    return zones, (ref_width, ref_height)


def get_huddling_indices(centroids, zones, ref_size, frame_size):
    """Return indices of birds whose centroids are outside all feeding zones."""
    if not zones:
        return list(range(len(centroids)))

    ref_width, ref_height = ref_size
    frame_width, frame_height = frame_size
    scale_x = frame_width / ref_width
    scale_y = frame_height / ref_height

    scaled_zones = [
        (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        for x1, y1, x2, y2 in zones
    ]

    keep_indices = []
    for i, (cx, cy) in enumerate(centroids):
        inside_zone = any(
            x1 <= cx <= x2 and y1 <= cy <= y2
            for x1, y1, x2, y2 in scaled_zones
        )
        if not inside_zone:
            keep_indices.append(i)

    return keep_indices



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
    ap.add_argument(
        "--zones",
        default="configs/feeding_zones.json",
        help="Feeding-zone JSON. Use an empty string to disable."
    )
    ap.add_argument("--huddle-method", choices=["connected", "complete", "average"],
                    default="complete", help="Experimental proximity rule")
    ap.add_argument("--clusters-csv", default="", help="Export per-cluster bounding boxes to CSV")
    ap.add_argument("--timestamps-csv", default="", help="Annotation CSV with t_s column; export only these timestamps")
    ap.add_argument("--diagnostics-csv", default="", help="Export all raw, ByteTrack, and hybrid boxes at selected timestamps")
    ap.add_argument("--log-csv", default="behavior_v2_log.csv")
    args = ap.parse_args()
    if args.zones:
        if not Path(args.zones).exists():
            ap.error(f"Feeding-zone file not found: {args.zones}")
        try:
            feeding_zones, ref_size = load_feeding_zones(args.zones)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            ap.error(f"Could not load feeding zones: {exc}")
        print(f"Loaded {len(feeding_zones)} feeding zones from {args.zones}")
    else:
        feeding_zones, ref_size = [], None
        print("Feeding-zone exclusion disabled.")

    if args.interval <= 0 or args.window <= 0:
        ap.error("--interval and --window must be positive")
    if not (0 < args.min_fill <= 1):
        ap.error("--min-fill must be in (0,1]")
    if args.min_huddle < 2:
        ap.error("--min-huddle must be >= 2")
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

    cluster_file = None
    cluster_writer = None
    if args.clusters_csv:
        cluster_file = open(args.clusters_csv, "w", newline="", encoding="utf-8")
        cluster_writer = csv.writer(cluster_file)
        cluster_writer.writerow(["t_s", "frame_no", "cluster_no", "track_ids",
                                 "member_boxes_xywh_json", "cluster_center_x",
                                 "cluster_center_y", "n_members", "n_clusters_in_frame"])

    diagnostic_file = None
    diagnostic_writer = None
    if args.diagnostics_csv:
        if selected_times is None:
            ap.error("--diagnostics-csv requires --timestamps-csv to limit diagnostic output")
        diagnostic_file = open(args.diagnostics_csv, "w", newline="", encoding="utf-8")
        diagnostic_writer = csv.writer(diagnostic_file)
        diagnostic_writer.writerow(["t_s", "frame_no", "source", "track_id", "confidence",
                                    "cx", "cy", "w", "h", "feeder_excluded",
                                    "huddle_cluster_no", "body_len_px", "n_hybrid_birds",
                                    "n_feeder_excluded"])

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
                         "recovered", "inactive_ids", "huddle_ids", "n_feeding"])
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
            n_feeding = 0

            # Hybrid V2 boxes are in (center_x, center_y, width, height) format.
            centroids = [(b[0], b[1]) for b in boxes]

            # Exclude feeder-zone birds from huddling only.
            # All birds remain in tracking history and inactivity evaluation.
            keep_indices = list(range(len(centroids)))
            if feeding_zones:
                frame_height, frame_width = frame.shape[:2]
                keep_indices = get_huddling_indices(
                    centroids,
                    feeding_zones,
                    ref_size,
                    (frame_width, frame_height)
                )
                n_feeding = len(centroids) - len(keep_indices)

            if body_len:
                filtered_centroids = [centroids[i] for i in keep_indices]
                huddle_fn = {"connected": detect_huddles,
                             "complete": detect_huddles_complete_linkage,
                             "average": detect_huddles_average_linkage}[args.huddle_method]
                filtered_huddles = huddle_fn(
                    filtered_centroids, body_len, args.min_huddle
                )

                # Convert filtered cluster indices back to original boxes/track IDs.
                huddles = [
                    [keep_indices[i] for i in cluster]
                    for cluster in filtered_huddles
                ]

                # Inactivity continues to use the complete, unfiltered history.
                inactive = update_inactivity(
                    history, t, args.window,
                    args.move_frac * body_len, min_samples
                )

            huddle_ids = ["|".join(map(str, sorted(ids[k] for k in cluster)))
                          for cluster in huddles]
            writer.writerow([
                f"{t:.3f}", len(ids), len(huddles), len(inactive),
                f"{body_len:.2f}" if body_len else "", recovered,
                "|".join(map(str, sorted(inactive))), ";".join(huddle_ids),
                n_feeding
            ])
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
            if diagnostic_writer is not None and any(abs(t - target) < 0.01 for target in selected_times):
                excluded = set(range(len(boxes))) - set(keep_indices)
                membership = {k: cluster_no for cluster_no, cluster in enumerate(huddles, 1)
                              for k in cluster}
                def write_detection(source, track_id, confidence, box, excluded_flag="", cluster_no=""):
                    diagnostic_writer.writerow([
                        f"{t:.3f}", n, source, track_id,
                        f"{float(confidence):.5f}" if confidence is not None else "",
                        *[f"{float(v):.2f}" for v in box],
                        excluded_flag, cluster_no,
                        f"{body_len:.2f}" if body_len else "", len(boxes), n_feeding
                    ])
                for i, (track_id, box) in enumerate(zip(ids, boxes)):
                    write_detection("hybrid", track_id, scores[i] if i < len(scores) else None,
                                    box, int(i in excluded), membership.get(i, ""))
                for track_id, box, score in zip(byte_ids, byte_boxes, byte_scores):
                    write_detection("bytetrack", track_id, score, box)
                for box, score in zip(raw_boxes, raw_scores):
                    write_detection("raw_yolo", "", score, box)

            count += 1
            if n % 30 == 0 or inactive:
                print(f"[{t:8.1f}s] frame={n}, birds={len(ids)}, "
                      f"huddles={len(huddles)}, inactive={len(inactive)}, "
                      f"feeder_excluded={n_feeding}, "
                      f"recovered={recovered}")
    if diagnostic_file is not None:
        diagnostic_file.close()
        print(f"Saved detection diagnostics to {args.diagnostics_csv}")
    if cluster_file is not None:
        cluster_file.close()
        print(f"Saved cluster details to {args.clusters_csv}")
    print(f"Saved {args.log_csv} ({count} frames, {hybrid.recovered_total} recovered detections)")
    print("NOTE: 'inactive' means centroid immobility under the rule, NOT a health diagnosis.")
    print("Validate each reported episode against independently annotated footage.")


if __name__ == "__main__":
    main()
