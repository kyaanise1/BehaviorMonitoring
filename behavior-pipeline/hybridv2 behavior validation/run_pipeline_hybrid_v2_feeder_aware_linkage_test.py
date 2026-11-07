"""V2 tracking -> rule-based inactivity/huddling -> CSV for manual validation.

This is a separate validation runner. It does not change run_pipeline.py.
Uses two independent YOLO model instances, as in eval_hybrid_v2.py.
It is NOT a production/hardware-optimized implementation.
"""
import argparse
import csv
import json
import math
import sys
from itertools import combinations
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



def cluster_zone_overlap_fraction(cluster, boxes, zones, ref_size, frame_size):
    """Fraction of the enclosing cluster rectangle covered by the union of exclusion zones.

    The overlap uses actual member bounding boxes (cx,cy,w,h), scaled feeding zones,
    and counts overlapping zone area only once.
    """
    if not cluster or not zones:
        return 0.0
    x1 = min(boxes[i][0] - boxes[i][2] / 2 for i in cluster)
    y1 = min(boxes[i][1] - boxes[i][3] / 2 for i in cluster)
    x2 = max(boxes[i][0] + boxes[i][2] / 2 for i in cluster)
    y2 = max(boxes[i][1] + boxes[i][3] / 2 for i in cluster)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return 0.0
    sx = frame_size[0] / ref_size[0]
    sy = frame_size[1] / ref_size[1]
    intersections = []
    for zx1, zy1, zx2, zy2 in zones:
        rect = (max(x1, zx1 * sx), max(y1, zy1 * sy),
                min(x2, zx2 * sx), min(y2, zy2 * sy))
        if rect[0] < rect[2] and rect[1] < rect[3]:
            intersections.append(rect)
    if not intersections:
        return 0.0
    # Exact union area of axis-aligned intersection rectangles via x-slabs.
    xs = sorted({x for a, _, b, _ in intersections for x in (a, b)})
    covered = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((a_y, b_y) for a_x, a_y, b_x, b_y in intersections
                           if a_x < right and b_x > left)
        merged_height = 0.0
        bottom = top = None
        for lo, hi in intervals:
            if bottom is None:
                bottom, top = lo, hi
            elif lo <= top:
                top = max(top, hi)
            else:
                merged_height += top - bottom
                bottom, top = lo, hi
        if bottom is not None:
            merged_height += top - bottom
        covered += (right - left) * merged_height
    return min(1.0, covered / area)


def cluster_rectangle(cluster, boxes):
    """Enclosing xyxy rectangle of member boxes in center-based xywh format."""
    return (
        min(boxes[i][0] - boxes[i][2] / 2 for i in cluster),
        min(boxes[i][1] - boxes[i][3] / 2 for i in cluster),
        max(boxes[i][0] + boxes[i][2] / 2 for i in cluster),
        max(boxes[i][1] + boxes[i][3] / 2 for i in cluster),
    )


def rectangle_iou(a, b):
    """Intersection-over-union of two (x1,y1,x2,y2) rectangles."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_huddles_average_no_chaining(centroids, body_len_px, min_birds=3):
    """Average-linkage ranking with a HARD all-pairs body-length limit.

    Every final cluster has maximum centroid-to-centroid member distance strictly
    below one estimated body length. A-B and B-C closeness can never make A-C
    join if A-C violates the limit. One bird belongs to at most one group.
    """
    if not body_len_px or body_len_px <= 0:
        return []
    groups = [[i] for i in range(len(centroids))]
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                distances = [math.dist(centroids[i], centroids[j])
                             for i in groups[a] for j in groups[b]]
                # A hard maximum rule, not a mean/connected-path proximity rule.
                if max(distances) >= body_len_px:
                    continue
                mean_distance = sum(distances) / len(distances)
                candidate = (mean_distance, a, b)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, a, b = best
        groups[a].extend(groups[b])
        del groups[b]
    return [sorted(g) for g in groups if len(g) >= min_birds]


def all_pairs_within_body_length(cluster, boxes, body_len_px):
    """Validate EVERY pair of member centroids, not merely neighbors/averages."""
    return bool(body_len_px and body_len_px > 0 and
                all(math.dist(boxes[i][:2], boxes[j][:2]) < body_len_px
                    for i, j in combinations(cluster, 2)))


def recover_compact_triplets(primary, keep_indices, boxes, body_len_px,
                             zones, ref_size, frame_size, overlap_limit):
    """Conservative second pass on *unassigned* eligible chickens.

    Find exactly three birds for which every center-to-center pair is closer
    than one estimated body length and mean pair distance is <= 0.75 lengths.
    Candidate groups must pass the same zone-overlap exclusion as primary
    clusters. Greedy non-overlap and track-membership checks prevent duplicates.
    All original primary clusters are preserved; no detections are added.
    """
    if body_len_px is None or body_len_px <= 0:
        return []
    assigned = {i for cluster in primary for i in cluster}
    available = sorted(i for i in keep_indices if i not in assigned)
    if len(available) < 3:
        return []

    primary_rects = [cluster_rectangle(cluster, boxes) for cluster in primary]
    candidates = []
    max_pair = body_len_px
    max_mean = 0.75 * body_len_px
    for triple in combinations(available, 3):
        ds = [math.dist(boxes[i][:2], boxes[j][:2]) for i, j in combinations(triple, 2)]
        # Keep all three pairwise distances strictly below one body length.
        if max(ds) >= max_pair or sum(ds) / 3 > max_mean:
            continue
        if zones and overlap_limit > 0:
            zone_fraction = cluster_zone_overlap_fraction(
                triple, boxes, zones, ref_size, frame_size)
            if zone_fraction >= overlap_limit:
                continue
        rect = cluster_rectangle(triple, boxes)
        if any(rectangle_iou(rect, old_rect) > 0.30 for old_rect in primary_rects):
            continue
        candidates.append((sum(ds) / 3, triple, rect))

    # Nearest average pair-distance first, with deterministic tie-breaking.
    candidates.sort(key=lambda item: (item[0], item[1]))
    added = []
    used = set()
    accepted_rects = list(primary_rects)
    for _, triple, rect in candidates:
        if any(i in used for i in triple):
            continue
        if any(rectangle_iou(rect, old_rect) > 0.30 for old_rect in accepted_rects):
            continue
        added.append(list(triple))
        used.update(triple)
        accepted_rects.append(rect)
    return added


def filter_clusters(candidate_huddles, overlap_by_cluster, threshold,
                    enable_recovery, keep_indices, boxes, body_len_px,
                    zones, ref_size, frame_size):
    """Apply a specific zone-overlap threshold, then optional local recovery."""
    if zones and threshold > 0:
        kept = [cluster for cluster in candidate_huddles
                if overlap_by_cluster[tuple(cluster)] < threshold]
    else:
        kept = list(candidate_huddles)
    if enable_recovery:
        kept += recover_compact_triplets(
            kept, keep_indices, boxes, body_len_px,
            zones, ref_size, frame_size, threshold)
    return kept


def temporal_triplet_geometry(track_ids, boxes_by_id, body_length):
    """Silhouette gap and centroid distances, normalized by median body length."""
    gaps, centers = [], []
    for left, right in combinations(track_ids, 2):
        a, b = boxes_by_id[left], boxes_by_id[right]
        centers.append(math.dist(a[:2], b[:2]) / body_length)
        dx = max(0.0, abs(a[0] - b[0]) - (a[2] + b[2]) / 2)
        dy = max(0.0, abs(a[1] - b[1]) - (a[3] + b[3]) / 2)
        gaps.append(math.hypot(dx, dy) / body_length)
    return max(gaps), sum(gaps) / len(gaps), max(centers), sum(centers) / len(centers)


def temporal_recover_huddles(target, window, zones, ref_size):
    """Two-stage temporal proposal recovery. Uses no manual annotations.

    target and window are small in-memory frame records created by main().
    Requires four qualifying neighbors: two prior and two future frames. Results
    for a sample become available when sample t+2 has been processed (~4 seconds).
    Primary average-linkage and local-recovery clusters are always retained.

    IMPORTANT: The thresholds are development-set choices, not independently
    verified operating thresholds. Do not treat Day 6 F1 as a holdout result.
    """
    primary = [(list(cluster), 'primary') for cluster in target['huddles']
               if all_pairs_within_body_length(cluster, target['boxes'], target['body_len'])]
    audit = []
    if len(window) < 5 or not target['body_len'] or not zones:
        return primary, audit
    if sum(f['seq'] < target['seq'] for f in window) < 2:
        return primary, audit
    if sum(f['seq'] > target['seq'] for f in window) < 2:
        return primary, audit

    # Existing all-frame detection export uses two decimal places. Match the
    # offline experiment's numerical precision for reproducible comparisons.
    all_boxes = [tuple(round(float(v), 2) for v in box) for box in target['boxes']]
    mapping = dict(zip(target['ids'], all_boxes))
    length = round(float(target['body_len']), 2)
    if length <= 0:
        return primary, audit
    eligible = sorted(target['ids'][i] for i in target['keep_indices'])
    index_of = {tid: i for i, tid in enumerate(target['ids'])}
    primary_rects = [cluster_rectangle(cluster, all_boxes) for cluster, _ in primary]
    next_frames = []
    for other in window:
        delta = other['seq'] - target['seq']
        if delta not in (-2, -1, 1, 2):
            continue
        next_frames.append((delta,
            dict(zip(other['ids'],
                     [tuple(round(float(v), 2) for v in box) for box in other['boxes']])),
            round(float(other['body_len']), 2) if other['body_len'] else 0))
    if len(next_frames) != 4:
        return primary, audit

    candidates = []
    for track_ids in combinations(eligible, 3):
        mg, ag, mc, ac = temporal_triplet_geometry(track_ids, mapping, length)
        # No chaining: A-C cannot exceed one body length even if A-B/B-C pass.
        if mg > .75 or ag > .40 or mc >= 1.0 or ac > 1.5:
            continue
        indices = [index_of[tid] for tid in track_ids]
        rect = cluster_rectangle(indices, all_boxes)
        zone_fraction = cluster_zone_overlap_fraction(
            indices, all_boxes, zones, ref_size, target['size'])
        if zone_fraction >= .30:
            continue
        old_iou = max((rectangle_iou(rect, old) for old in primary_rects), default=0)
        neighbor_features = {}
        for delta, box_map, neighbor_len in next_frames:
            if neighbor_len > 0 and all(tid in box_map for tid in track_ids):
                neighbor_features[delta] = temporal_triplet_geometry(
                    track_ids, box_map, neighbor_len)
        candidates.append((track_ids, indices, rect, mg, ag, mc, ac,
                           zone_fraction, old_iou, neighbor_features))

    selected = list(primary)
    selected_rects = list(primary_rects)
    used_added = set()

    def select_stage(stage, cap, max_gap, avg_gap, max_center,
                     avg_center, neighbor_max_gap, max_new, score_factor):
        scored = []
        for item in candidates:
            tids, indices, rect, mg, ag, mc, ac, z, old_iou, neighbors = item
            if z >= cap or mg > max_gap or ag > avg_gap or mc > max_center or ac > avg_center:
                continue
            if old_iou > .30 or any(tid in used_added for tid in tids):
                continue
            support = sum(1 for delta, features in neighbors.items()
                          if abs(delta) <= 2 and features[0] <= neighbor_max_gap
                          and features[2] <= 2.1)
            if support < 4:
                continue
            scored.append(((-support, ag + score_factor * ac, mg, tids), item, support))
        scored.sort(key=lambda v: v[0])
        accepted = 0
        for _, item, support in scored:
            if accepted >= max_new:
                break
            tids, indices, rect, mg, ag, mc, ac, z, old_iou, neighbors = item
            if any(tid in used_added for tid in tids):
                continue
            if any(rectangle_iou(rect, old_rect) > .30 for old_rect in selected_rects):
                continue
            selected.append((indices, stage))
            selected_rects.append(rect)
            used_added.update(tids)
            accepted += 1
            audit.append([f"{target['t']:.3f}", stage, '|'.join(map(str, tids)),
                          round(z, 4), round(mg, 4), round(ag, 4),
                          round(mc, 4), support])

    # These exact two stages reproduced the offline development experiment.
    select_stage('temporal_strict', .15, .25, .18, 1.25, 1.50, .25, 3, .15)
    select_stage('temporal_zero_zone', .03, .35, .20, 1.25, 1.50, .60, 1, .20)
    return selected, audit


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
    ap.add_argument("--cluster-zone-overlap", type=float, default=0.15,
                    help="Reject clusters if this fraction of their enclosing box overlaps zones (0 disables; default 0.15)")
    ap.add_argument("--overlap-sweep", default="",
                    help="Comma-separated thresholds to test in one pass, e.g. 0.15,0.20,0.25,0.30. "
                         "Writes a cluster CSV for each threshold.")
    ap.add_argument("--local-recovery", action="store_true",
                    help="Experimental second pass: recover compact groups of 3 unassigned birds; "
                         "primary clustering and other behavior rules stay unchanged")
    ap.add_argument("--clusters-csv", default="", help="Export per-cluster bounding boxes to CSV")
    ap.add_argument("--temporal-recovery", action="store_true",
                    help="Experimental temporal subgroup recovery (two preceding + two future samples; "
                         "~4-second delayed outputs when interval=2).")
    ap.add_argument("--temporal-audit-csv", default="",
                    help="Optional CSV of temporal clusters added by --temporal-recovery")
    ap.add_argument("--timestamps-csv", default="", help="Annotation CSV with t_s column; export only these timestamps")
    ap.add_argument("--all-detections-csv", default="",
                    help="Optional diagnostic export of all raw YOLO, ByteTrack, and Hybrid V2 "
                         "detections at every sampled frame (not only validation timestamps)")
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
    if args.local_recovery and args.min_huddle != 3:
        ap.error("--local-recovery currently requires --min-huddle 3")
    if args.local_recovery and args.huddle_method != "average":
        ap.error("--local-recovery is intended for --huddle-method average")
    if not 0 <= args.cluster_zone_overlap <= 1:
        ap.error("--cluster-zone-overlap must be between 0 and 1")
    if args.temporal_recovery:
        if args.huddle_method != "average" or not args.local_recovery or args.min_huddle != 3:
            ap.error("--temporal-recovery requires --huddle-method average --local-recovery --min-huddle 3")
        if abs(args.cluster_zone_overlap - .30) > 1e-8:
            ap.error("The tested --temporal-recovery configuration requires --cluster-zone-overlap 0.30")
        if not args.zones:
            ap.error("--temporal-recovery requires a feeder/drinker zones JSON")
        if args.overlap_sweep:
            ap.error("--temporal-recovery cannot be combined with --overlap-sweep")
        if len(feeding_zones) != 4:
            print("WARNING: The Day 6 experiment used FOUR feeder/drinker zones; "
                  "a different zone layout changes results.")
    elif args.temporal_audit_csv:
        ap.error("--temporal-audit-csv requires --temporal-recovery")
    thresholds = []
    if args.overlap_sweep:
        try:
            thresholds = [float(x.strip()) for x in args.overlap_sweep.split(",")]
        except ValueError:
            ap.error("--overlap-sweep must be comma-separated decimals, such as 0.15,0.20,0.25,0.30")
        if not thresholds or any(not 0 < x <= 1 for x in thresholds):
            ap.error("Each overlap-sweep threshold must be in (0,1]")
        if len(set(thresholds)) != len(thresholds):
            ap.error("--overlap-sweep contains duplicate thresholds")
        if not args.clusters_csv:
            ap.error("--overlap-sweep requires --clusters-csv as an output filename template")
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

    # Normal mode writes exactly the requested CSV. Sweep mode uses that path as
    # a filename stem and creates independent outputs for all thresholds.
    cluster_files = []
    cluster_writers = {}
    fieldnames = ["t_s", "frame_no", "cluster_no", "track_ids",
                  "member_boxes_xywh_json", "cluster_center_x",
                  "cluster_center_y", "n_members", "n_clusters_in_frame"]
    if args.clusters_csv:
        output = Path(args.clusters_csv)
        for threshold in (thresholds if thresholds else [args.cluster_zone_overlap]):
            if thresholds:
                suffix = f"_overlap_{round(threshold * 100):02d}pct"
                dest = output.with_name(output.stem + suffix + (output.suffix or ".csv"))
            else:
                dest = output
            dest.parent.mkdir(parents=True, exist_ok=True)
            handle = open(dest, "w", newline="", encoding="utf-8")
            cluster_files.append((handle, dest))
            csv_writer = csv.writer(handle)
            csv_writer.writerow(fieldnames + (["origin"] if args.temporal_recovery else []))
            cluster_writers[threshold] = csv_writer

    # Optional all-frame detection audit. This is output-only: it must not
    # modify detections, tracking, clustering, or behavior decisions.
    audit_file = None
    audit_writer = None
    if args.all_detections_csv:
        audit_dest = Path(args.all_detections_csv)
        audit_dest.parent.mkdir(parents=True, exist_ok=True)
        audit_file = open(audit_dest, "w", newline="", encoding="utf-8")
        audit_writer = csv.writer(audit_file)
        audit_writer.writerow([
            "t_s", "frame_no", "source", "track_id", "confidence",
            "cx", "cy", "w", "h", "feeder_excluded", "huddle_cluster_no",
            "body_len_px", "n_hybrid_birds", "frame_width", "frame_height",
        ])

    temporal_audit_handle = None
    temporal_audit_writer = None
    if args.temporal_audit_csv:
        audit_path = Path(args.temporal_audit_csv)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporal_audit_handle = open(audit_path, "w", newline="", encoding="utf-8")
        temporal_audit_writer = csv.writer(temporal_audit_handle)
        temporal_audit_writer.writerow([
            "t_s", "stage", "track_ids", "zone_overlap", "max_box_gap_bodylength",
            "mean_box_gap_bodylength", "max_centroid_distance_bodylength",
            "temporal_support"])

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
        temporal_buffer = deque()
        temporal_added = 0

        def write_temporal_result(record, context):
            """Write a fully resolved sample, delayed by two sampled frames."""
            nonlocal temporal_added
            clusters_with_source, added = temporal_recover_huddles(
                record, context, feeding_zones, ref_size)
            clusters = [cluster for cluster, origin in clusters_with_source]
            temporal_added += len(added)
            if temporal_audit_writer is not None:
                temporal_audit_writer.writerows(added)
            ids0, boxes0 = record['ids'], record['boxes']
            t0, frame_no = record['t'], record['frame_no']
            inactive0 = record['inactive']
            huddle_names = ["|".join(map(str, sorted(ids0[k] for k in cluster)))
                            for cluster in clusters]
            writer.writerow([
                f"{t0:.3f}", len(ids0), len(clusters), len(inactive0),
                f"{record['body_len']:.2f}" if record['body_len'] else "",
                record['recovered'], "|".join(map(str, sorted(inactive0))),
                ";".join(huddle_names), record['n_feeding']])

            if audit_writer is not None:
                excluded = set(range(len(boxes0))) - set(record['keep_indices'])
                membership = {bird_idx: cluster_no
                              for cluster_no, cluster in enumerate(clusters, 1)
                              for bird_idx in cluster}
                fw, fh = record['size']
                def audit_bird(source, tid, score, box, exclude="", cluster_no=""):
                    audit_writer.writerow([
                        f"{t0:.3f}", frame_no, source, tid,
                        f"{float(score):.5f}" if score is not None else "",
                        *[f"{float(v):.2f}" for v in box],
                        exclude, cluster_no,
                        f"{record['body_len']:.2f}" if record['body_len'] else "",
                        len(boxes0), fw, fh])
                for i, (tid, box) in enumerate(zip(ids0, boxes0)):
                    audit_bird('hybrid', tid,
                               record['scores'][i] if i < len(record['scores']) else None,
                               box, int(i in excluded), membership.get(i, ""))
                for tid, box, score in zip(record['byte_ids'], record['byte_boxes'], record['byte_scores']):
                    audit_bird('bytetrack', tid, score, box)
                for box, score in zip(record['raw_boxes'], record['raw_scores']):
                    audit_bird('raw_yolo', "", score, box)

            if cluster_writers and (
                selected_times is None or any(abs(t0 - x) < .01 for x in selected_times)
            ):
                out = cluster_writers[args.cluster_zone_overlap]
                if not clusters_with_source:
                    out.writerow([f"{t0:.3f}", frame_no, "", "", "[]", "", "", 0, 0, ""])
                for cluster_no, (cluster, origin) in enumerate(clusters_with_source, 1):
                    members = [
                        {"track_id": int(ids0[k]),
                         "cx": round(float(boxes0[k][0]), 2),
                         "cy": round(float(boxes0[k][1]), 2),
                         "w": round(float(boxes0[k][2]), 2),
                         "h": round(float(boxes0[k][3]), 2)}
                        for k in cluster]
                    out.writerow([
                        f"{t0:.3f}", frame_no, cluster_no,
                        "|".join(str(m["track_id"]) for m in members),
                        json.dumps(members, separators=(",", ":")),
                        round(sum(m["cx"] for m in members) / len(members), 2),
                        round(sum(m["cy"] for m in members) / len(members), 2),
                        len(members), len(clusters_with_source), origin])

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
                             # Keep average-distance ranking, but require ALL
                             # pairs to be within a single body length.
                             "average": detect_huddles_average_no_chaining}[args.huddle_method]
                filtered_huddles = huddle_fn(
                    filtered_centroids, body_len, args.min_huddle
                )

                # Convert filtered cluster indices back to original boxes/track IDs.
                huddles = [
                    [keep_indices[i] for i in cluster]
                    for cluster in filtered_huddles
                ]

                # Apply filtering for the main log threshold. For sweep output,
                # each threshold is applied independently to these same candidate
                # clusters, without re-running detection/tracking/clustering.
                candidate_huddles = huddles
                if feeding_zones and (args.cluster_zone_overlap > 0 or thresholds):
                    frame_h, frame_w = frame.shape[:2]
                    overlap_by_cluster = {
                        tuple(cluster): cluster_zone_overlap_fraction(
                            cluster, boxes, feeding_zones, ref_size, (frame_w, frame_h))
                        for cluster in candidate_huddles
                    }
                else:
                    overlap_by_cluster = {tuple(cluster): 0.0 for cluster in candidate_huddles}
                # Primary results are unchanged unless --local-recovery is enabled.
                # The same selective recovery is applied independently per sweep
                # threshold, so different thresholds never share cluster state.
                huddles = filter_clusters(
                    candidate_huddles, overlap_by_cluster, args.cluster_zone_overlap,
                    args.local_recovery, keep_indices, boxes, body_len,
                    feeding_zones, ref_size, (frame.shape[1], frame.shape[0]))
                # Final hard invariant for primary AND local-recovery clusters.
                # Connected groups violating it are rejected, never exported.
                huddles = [cluster for cluster in huddles
                           if all_pairs_within_body_length(cluster, boxes, body_len)]

                # Inactivity continues to use the complete, unfiltered history.
                inactive = update_inactivity(
                    history, t, args.window,
                    args.move_frac * body_len, min_samples
                )

            if not body_len:
                candidate_huddles = []
                overlap_by_cluster = {}

            if args.temporal_recovery:
                frame_h, frame_w = frame.shape[:2]
                temporal_buffer.append({
                    'seq': count, 't': t, 'frame_no': n,
                    'ids': list(ids), 'boxes': list(boxes),
                    'body_len': body_len, 'keep_indices': list(keep_indices),
                    'huddles': [list(cluster) for cluster in huddles],
                    'inactive': list(inactive), 'recovered': recovered,
                    'n_feeding': n_feeding, 'size': (frame_w, frame_h),
                    'scores': list(scores) if audit_writer is not None else [],
                    'byte_ids': list(byte_ids) if audit_writer is not None else [],
                    'byte_boxes': list(byte_boxes) if audit_writer is not None else [],
                    'byte_scores': list(byte_scores) if audit_writer is not None else [],
                    'raw_boxes': list(raw_boxes) if audit_writer is not None else [],
                    'raw_scores': list(raw_scores) if audit_writer is not None else [],
                })
                # Exactly 2 future samples are required. Oldest context is kept
                # until no longer needed. Streaming memory remains bounded (5 frames).
                if len(temporal_buffer) >= 3:
                    target = temporal_buffer[-3]
                    write_temporal_result(target, list(temporal_buffer))
                    if len(temporal_buffer) > 4:
                        temporal_buffer.popleft()
                count += 1
                if n % 30 == 0 or inactive:
                    print(f"[{t:8.1f}s] frame={n}, birds={len(ids)}, "
                          f"primary_huddles={len(huddles)}, inactive={len(inactive)}, "
                          f"feeder_excluded={n_feeding}, recovered={recovered}")
                continue

            huddle_ids = ["|".join(map(str, sorted(ids[k] for k in cluster)))
                          for cluster in huddles]
            writer.writerow([
                f"{t:.3f}", len(ids), len(huddles), len(inactive),
                f"{body_len:.2f}" if body_len else "", recovered,
                "|".join(map(str, sorted(inactive))), ";".join(huddle_ids),
                n_feeding
            ])
            if audit_writer is not None:
                # Export at EVERY sampled timestamp, including zero-huddle frames.
                # `source=hybrid` rows are the actual chicken positions used by
                # huddling; raw_yolo/bytetrack rows help identify upstream misses.
                excluded_indices = set(range(len(boxes))) - set(keep_indices)
                membership = {
                    bird_idx: cluster_idx
                    for cluster_idx, cluster in enumerate(huddles, 1)
                    for bird_idx in cluster
                }
                frame_h, frame_w = frame.shape[:2]

                def export_detection(source, tid, score, xywh,
                                     excluded="", cluster_no=""):
                    audit_writer.writerow([
                        f"{t:.3f}", n, source, tid,
                        f"{float(score):.5f}" if score is not None else "",
                        *[f"{float(v):.2f}" for v in xywh],
                        excluded, cluster_no,
                        f"{body_len:.2f}" if body_len else "",
                        len(boxes), frame_w, frame_h,
                    ])

                for i, (tid, box) in enumerate(zip(ids, boxes)):
                    export_detection(
                        "hybrid", tid, scores[i] if i < len(scores) else None,
                        box, int(i in excluded_indices), membership.get(i, ""))
                for tid, box, score in zip(byte_ids, byte_boxes, byte_scores):
                    export_detection("bytetrack", tid, score, box)
                for box, score in zip(raw_boxes, raw_scores):
                    export_detection("raw_yolo", "", score, box)
            if cluster_writers and (
                selected_times is None or any(abs(t - target) < 0.01 for target in selected_times)
            ):
                for threshold, cluster_writer in cluster_writers.items():
                    export_huddles = filter_clusters(
                        candidate_huddles, overlap_by_cluster, threshold,
                        args.local_recovery, keep_indices, boxes, body_len,
                        feeding_zones, ref_size, (frame.shape[1], frame.shape[0]))
                    if not export_huddles:
                        cluster_writer.writerow([f"{t:.3f}", n, "", "", "[]", "", "", 0, 0])
                    for cluster_no, cluster in enumerate(export_huddles, start=1):
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
                            len(members), len(export_huddles)
                        ])
            count += 1
            if n % 30 == 0 or inactive:
                print(f"[{t:8.1f}s] frame={n}, birds={len(ids)}, "
                      f"huddles={len(huddles)}, inactive={len(inactive)}, "
                      f"feeder_excluded={n_feeding}, "
                      f"recovered={recovered}")
        if args.temporal_recovery:
            # The final two frames lack complete future context. Keep their
            # primary clusters without inventing temporal support.
            for pending in list(temporal_buffer)[-min(count, 2):]:
                write_temporal_result(pending, list(temporal_buffer))
            print(f"Temporal recovery added {temporal_added} subgroup predictions "
                  f"(up to {2 * args.interval:.1f}s decision delay)")
    if temporal_audit_handle is not None:
        temporal_audit_handle.close()
        print(f"Saved temporal additions to {args.temporal_audit_csv}")
    if audit_file is not None:
        audit_file.close()
        print(f"Saved all-frame detection audit to {args.all_detections_csv}")
    for handle, dest in cluster_files:
        handle.close()
        print(f"Saved cluster details to {dest}")
    print(f"Saved {args.log_csv} ({count} frames, {hybrid.recovered_total} recovered detections)")
    print("NOTE: 'inactive' means centroid immobility under the rule, NOT a health diagnosis.")
    print("Validate each reported episode against independently annotated footage.")


if __name__ == "__main__":
    main()
