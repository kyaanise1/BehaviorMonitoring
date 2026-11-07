"""Feeder-aware Hybrid V2 huddling rules, independent of YOLO/video I/O.

This module retains the exact geometry, average-linkage, local-triplet and
four-neighbor temporal recovery rules from the uploaded run_behavior_v2.py.
It consumes already-tracked detections, and NEVER invokes its own tracker.
"""
import json
import math
from itertools import combinations
from behavior_linkage_test import (detect_huddles, detect_huddles_complete_linkage)

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


def expand_feeding_zones(zones, ref_size, margin_px):
    """Expand equipment rectangles to cover birds feeding beside them (reference pixels)."""
    if margin_px < 0:
        raise ValueError('Feeder margin must be nonnegative')
    w, h = ref_size
    return [[max(0, x1 - margin_px), max(0, y1 - margin_px),
             min(w, x2 + margin_px), min(h, y2 + margin_px)]
            for x1, y1, x2, y2 in zones]


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


def silhouette_gap_between(a, b):
    """Euclidean gap between two xywh rectangles (zero if touching/overlapping)."""
    dx = max(0.0, abs(a[0] - b[0]) - (a[2] + b[2]) / 2)
    dy = max(0.0, abs(a[1] - b[1]) - (a[3] + b[3]) / 2)
    return math.hypot(dx, dy)


def detect_huddles_contact_aware(boxes, body_len_px, min_birds=3):
    """Complete-pair-valid grouping, ranked by BOTH centroid and body-box gap.

    A merger is NEVER allowed when any centroid pair is >= one body length.
    Within that hard limit, equally normalized centroid separation and visible
    body-box gap are used to resolve which adjacent birds group together.
    This reduces wrong triplet membership caused by centroid-only greedy merges.
    """
    if not body_len_px or body_len_px <= 0:
        return []
    groups = [[i] for i in range(len(boxes))]
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                centers = []
                gaps = []
                for i in groups[a]:
                    for j in groups[b]:
                        center_dist = math.dist(boxes[i][:2], boxes[j][:2]) / body_len_px
                        centers.append(center_dist)
                        gaps.append(silhouette_gap_between(boxes[i], boxes[j]) / body_len_px)
                # Still enforces strictly one-body-length for EVERY pair.
                if max(centers) >= 1.0:
                    continue
                score = (sum(centers) + sum(gaps)) / len(centers)
                candidate = (score, a, b)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, a, b = best
        groups[a].extend(groups[b])
        del groups[b]
    return [sorted(group) for group in groups if len(group) >= min_birds]


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


def recover_contact_triplets(primary, keep_indices, boxes, body_len_px,
                             zones, ref_size, frame_size, overlap_limit):
    """Second-chance tightly touching 3-bird groups, without relaxing proximity.

    All centroid pairs must be within one body length. Compared to the
    legacy recovery rule (mean centroid separation <= 0.75 body lengths),
    physically contacting groups can have mean separation up to 0.90 lengths
    IF their body-box gaps are <= 0.10 lengths. No primary group is removed.
    """
    if not body_len_px or body_len_px <= 0:
        return []
    assigned = {i for cluster in primary for i in cluster}
    available = sorted(i for i in keep_indices if i not in assigned)
    if len(available) < 3:
        return []
    existing_rects = [cluster_rectangle(g, boxes) for g in primary]
    candidates = []
    for triple in combinations(available, 3):
        center_distances = [math.dist(boxes[i][:2], boxes[j][:2]) / body_len_px
                            for i, j in combinations(triple, 2)]
        if max(center_distances) >= 1.0 or sum(center_distances) / 3 > 0.90:
            continue
        gaps = [silhouette_gap_between(boxes[i], boxes[j]) / body_len_px
                for i, j in combinations(triple, 2)]
        if max(gaps) > 0.10 or sum(gaps) / 3 > 0.10:
            continue
        if zones and overlap_limit > 0:
            zone_fraction = cluster_zone_overlap_fraction(
                triple, boxes, zones, ref_size, frame_size)
            if zone_fraction >= overlap_limit:
                continue
        rect = cluster_rectangle(triple, boxes)
        if any(rectangle_iou(rect, old) > 0.30 for old in existing_rects):
            continue
        candidates.append((max(gaps), sum(gaps), sum(center_distances), triple, rect))
    candidates.sort()
    recovered = []
    used = set()
    accepted_rects = list(existing_rects)
    for _max_gap, _sum_gap, _sum_center, triple, rect in candidates:
        if any(i in used for i in triple):
            continue
        if any(rectangle_iou(rect, old) > 0.30 for old in accepted_rects):
            continue
        recovered.append(list(triple))
        used.update(triple)
        accepted_rects.append(rect)
    return recovered


def filter_clusters(candidate_huddles, overlap_by_cluster, threshold,
                    enable_recovery, keep_indices, boxes, body_len_px,
                    zones, ref_size, frame_size, contact_aware=False):
    """Apply a specific zone-overlap threshold, then optional local recovery."""
    if zones and threshold > 0:
        kept = [cluster for cluster in candidate_huddles
                if overlap_by_cluster[tuple(cluster)] < threshold]
    else:
        kept = list(candidate_huddles)
    if enable_recovery:
        recovery_fn = recover_contact_triplets if contact_aware else recover_compact_triplets
        kept += recovery_fn(
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
    primary_members = {i for group, _origin in primary for i in group}
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
        # A chicken cannot belong to two simultaneous huddling groups.
        # Rectangle overlap alone did not guarantee exclusive membership.
        if any(index in primary_members for index in indices):
            continue
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


class HuddlingModule:
    """Per-frame spatial huddling plus optional delayed temporal recovery."""
    def __init__(self, zones, ref_size, min_huddle=3, method='average',
                  cluster_zone_overlap=0.30, local_recovery=True,
                  overlap_sweep=False, feeder_margin_px=15,
                  hard_feeder_exclusion=False):
        # The supplied JSON rectangles are large feeding/activity regions, NOT
        # precise outlines of four feeding bowls. Do not drop every bird whose
        # centroid is inside those broad zones: that removes legitimate huddles.
        # Use cluster-overlap filtering as the default, with a modest perimeter.
        self.core_zones = [list(z) for z in zones]
        self.zones = expand_feeding_zones(zones, ref_size, feeder_margin_px) if zones else []
        self.ref_size = ref_size
        self.feeder_margin_px = feeder_margin_px
        self.hard_feeder_exclusion = hard_feeder_exclusion
        self.min_huddle = min_huddle
        self.method = method
        self.cluster_zone_overlap = cluster_zone_overlap
        self.local_recovery = local_recovery
        self.overlap_sweep = overlap_sweep

    def augment_with_untracked_yolo(self, ids, boxes, raw_boxes, raw_scores,
                                   body_len, frame_no, min_conf=0.30,
                                   max_duplicate_iou=0.40, max_neighbor_lengths=1.5):
        """Optional *huddling-only* raw-YOLO proposals omitted by Hybrid V2.

        The same detector already runs for RecoveryHybrid; this is NOT another
        inference pass. Accept a sufficiently confident raw detection only when
        it is not already represented by a tracked box and lies near tracked
        chickens. Each accepted detection gets a negative, frame-local ID.
        It MUST NOT enter the inactivity history or be treated as a persistent ID.
        All actual huddling clusters still pass the original centroid-pair rule.
        """
        result_ids, result_boxes = list(ids), list(boxes)
        raw_added = {}
        if not body_len or not boxes:
            return result_ids, result_boxes, raw_added
        tracked_boxes = list(boxes)
        for k, (raw_box, score) in enumerate(zip(raw_boxes, raw_scores)):
            if score < min_conf:
                continue
            candidate = tuple(float(v) for v in raw_box)
            rect = cluster_rectangle([0], [candidate])
            # Both tracked and previously accepted raw boxes prevent duplicates.
            if any(rectangle_iou(rect, cluster_rectangle([0], [old])) >= max_duplicate_iou
                   for old in result_boxes):
                continue
            if min(math.dist(candidate[:2], old[:2]) for old in tracked_boxes) > max_neighbor_lengths * body_len:
                continue
            local_id = -(int(frame_no) * 1000 + k + 1)
            result_ids.append(local_id)
            result_boxes.append(candidate)
            raw_added[local_id] = float(score)
        return result_ids, result_boxes, raw_added

    def analyze(self, ids, boxes, body_len, frame_size):
        centroids = [(box[0], box[1]) for box in boxes]
        # Count birds in the original configured zones for diagnostic purposes.
        # This is NOT a feeding classifier and does not remove those birds from
        # huddling candidacy. Only full-cluster zone-overlap is used by default.
        core_outside = get_huddling_indices(
            centroids, self.core_zones, self.ref_size, frame_size)
        feeding_indices = set(range(len(centroids))) - set(core_outside)
        keep_indices = list(range(len(centroids)))
        if self.hard_feeder_exclusion and self.zones:
            # Optional backwards-compatible strict exclusion for comparisons.
            keep_indices = get_huddling_indices(
                centroids, self.zones, self.ref_size, frame_size)
        n_feeding = len(feeding_indices)
        candidates = []
        overlaps = {}
        huddles = []
        if body_len:
            filtered = [centroids[idx] for idx in keep_indices]
            if self.method == 'contact':
                # Contact-aware ranking uses both centroids and full bird boxes.
                local_groups = detect_huddles_contact_aware(
                    [boxes[idx] for idx in keep_indices], body_len, self.min_huddle)
            else:
                huddle_fn = {
                    'connected': detect_huddles,
                    'complete': detect_huddles_complete_linkage,
                    'average': detect_huddles_average_no_chaining,
                }[self.method]
                local_groups = huddle_fn(filtered, body_len, self.min_huddle)
            candidates = [[keep_indices[i] for i in group]
                          for group in local_groups]
            if self.zones and (self.cluster_zone_overlap > 0 or self.overlap_sweep):
                overlaps = {tuple(group): cluster_zone_overlap_fraction(
                    group, boxes, self.zones, self.ref_size, frame_size)
                    for group in candidates}
            else:
                overlaps = {tuple(group): 0.0 for group in candidates}
            huddles = self.filter_candidates(candidates, overlaps,
                                             self.cluster_zone_overlap,
                                             keep_indices, boxes, body_len, frame_size)
        return {
            'keep_indices': keep_indices,
            'feeding_indices': feeding_indices,
            'n_feeding': n_feeding,
            'candidates': candidates,
            'overlaps': overlaps,
            'huddles': huddles,
        }

    def filter_candidates(self, candidates, overlaps, threshold,
                          keep_indices, boxes, body_len, frame_size):
        huddles = filter_clusters(
            candidates, overlaps, threshold, self.local_recovery,
            keep_indices, boxes, body_len, self.zones, self.ref_size, frame_size,
            contact_aware=(self.method == 'contact'))
        return [group for group in huddles
                if all_pairs_within_body_length(group, boxes, body_len)]

    def resolve_temporal(self, record, context):
        return temporal_recover_huddles(record, context, self.zones, self.ref_size)
