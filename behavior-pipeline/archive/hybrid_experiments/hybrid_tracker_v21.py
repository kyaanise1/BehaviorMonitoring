"""Experimental V2.1: V2 raw-detection recovery plus conflict-aware ID association.

- Reuses V2's RecoveryHybrid.update() AS IS, including its recovery policy.
- Replaces only the stable-ID manager's first association pass.
- Optimizes close associations globally (Hungarian), with a bounded preference
  for an existing, spatially plausible ByteTrack ID.
- Keeps V2's second ByteTrack-continuity and third wide-centroid passes.
- Every ByteTrack output remains present, with its original box/confidence.

This does not use ground truth at inference time and does not implement
appearance-based reidentification. Accuracy is NOT guaranteed to improve.
"""
import math
import statistics
from scipy.optimize import linear_sum_assignment

from hybrid_tracker import HybridIDManager, _HybridTrack
from hybrid_tracker_v2 import RecoveryHybrid


class ConflictAwareIDManager(HybridIDManager):
    """Global close-distance matching with a bounded ByteTrack continuity cue."""

    def __init__(self, *args, byte_bonus=0.35, same_id_gate=1.25,
                 size_weight=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        if byte_bonus < 0:
            raise ValueError("byte_bonus must be >= 0")
        if not 0 < same_id_gate <= self.wide_frac:
            raise ValueError("same_id_gate must be in (0, wide_frac]")
        if size_weight < 0:
            raise ValueError("size_weight must be >= 0")
        self.byte_bonus = float(byte_bonus)
        self.same_id_gate = float(same_id_gate)
        self.size_weight = float(size_weight)
        self.frames = 0
        self.close_matches = 0
        self.byte_continuity_matches = 0

    def _global_close_match(self, tids, dids, boxes, byte_ids, body):
        """One-to-one optimal matches for tight candidates and plausible same-ID edges.

        Same ByteTrack IDs are preferred, not required. They receive a bonus only
        when the candidate is within same_id_gate body lengths and the width and
        height are compatible with the previous detection.
        """
        if not tids or not dids:
            return {}
        n, m = len(tids), len(dids)
        BIG = 1e8
        # A separate dummy column per track permits unmatched tracks; all
        # accepted edges are cheaper than a dummy assignment.
        dummy = max(self.tight_frac, self.same_id_gate) + 2.0 + self.size_weight * 2.0
        costs = [[BIG] * (m + n) for _ in range(n)]
        for i, tid in enumerate(tids):
            tr = self.tracks[tid]
            costs[i][m + i] = dummy
            for j, di in enumerate(dids):
                box = boxes[di]
                if not self._valid_size(tr, box):
                    continue
                distance = self._dist(tr, box) / body
                same = int(byte_ids[di]) == tr.byte_id
                eligible = (distance <= self.tight_frac or
                            (same and distance <= self.same_id_gate))
                if not eligible:
                    continue
                size_change = (abs(math.log(tr.w / box[2])) +
                               abs(math.log(tr.h / box[3])))
                # A small size-consistency cost and a bounded same-ID bonus.
                costs[i][j] = (distance + self.size_weight * size_change -
                               (self.byte_bonus if same else 0.0))

        rows, cols = linear_sum_assignment(costs)
        matched = {}
        for i, j in zip(rows, cols):
            if j < m and costs[i][j] < dummy:
                tid, di = tids[i], dids[j]
                matched[tid] = di
                self.close_matches += 1
                if self.tracks[tid].byte_id == int(byte_ids[di]):
                    self.byte_continuity_matches += 1
        return matched

    def update(self, byte_ids, xywh, scores):
        """Return (stable_ids, boxes, scores) in original ByteTrack row order."""
        if not (len(byte_ids) == len(xywh) == len(scores)):
            raise ValueError("byte_ids, xywh, scores must have equal lengths")
        if len(set(int(v) for v in byte_ids)) != len(byte_ids):
            raise ValueError("ByteTrack returned duplicate IDs in one frame")

        boxes = [tuple(map(float, b)) for b in xywh]
        for box in boxes:
            if len(box) != 4 or not all(math.isfinite(x) for x in box):
                raise ValueError("Each box must contain four finite values")
            if box[2] <= 0 or box[3] <= 0:
                raise ValueError("Box width and height must be positive")

        if boxes:
            self._body.append(statistics.median(max(b[2], b[3]) for b in boxes))
        body = max(1e-6, statistics.median(self._body) if self._body else 1.0)

        tids = list(self.tracks)
        dids = list(range(len(boxes)))
        # V2 used greedy centroid first. V2.1 uses global, continuity-aware
        # assignment for plausible close matches, and keeps later passes.
        matched = self._global_close_match(tids, dids, boxes, byte_ids, body)

        # Pass 2: original V2's ByteTrack ID continuity for remaining tracks.
        used_d = set(matched.values())
        bt_candidates = []
        for tid in tids:
            if tid in matched:
                continue
            tr = self.tracks[tid]
            for di in dids:
                if di in used_d or int(byte_ids[di]) != tr.byte_id:
                    continue
                if self._valid_size(tr, boxes[di]):
                    distance = self._dist(tr, boxes[di])
                    if distance <= self.wide_frac * body:
                        bt_candidates.append((distance, tid, di))
        bt_candidates.sort()
        for _, tid, di in bt_candidates:
            if tid not in matched and di not in used_d:
                matched[tid] = di
                used_d.add(di)

        # Pass 3: original V2 wide-distance centroid fallback.
        rem_t = [tid for tid in tids if tid not in matched]
        rem_d = [di for di in dids if di not in used_d and scores[di] >= self.wide_conf]
        matched.update(self._greedy(rem_t, rem_d, boxes, body, self.wide_frac))

        stable = [None] * len(boxes)
        for tid, di in matched.items():
            tr = self.tracks[tid]
            tr.cx, tr.cy, tr.w, tr.h = boxes[di]
            tr.byte_id = int(byte_ids[di])
            tr.missed = 0
            tr.hits += 1
            stable[di] = tid

        for tid in tids:
            if tid not in matched:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        for di in dids:
            if stable[di] is None:
                tid = self.next_id
                self.next_id += 1
                cx, cy, w, h = boxes[di]
                self.tracks[tid] = _HybridTrack(tid, byte_ids[di], cx, cy, w, h)
                stable[di] = tid

        self.frames += 1
        assert all(sid is not None for sid in stable)
        assert len(set(stable)) == len(stable), "Repeated stable ID in frame"
        return stable, [list(box) for box in boxes], list(scores)


class RecoveryHybridV21(RecoveryHybrid):
    """V2 recovery unchanged; optionally replaces V2's stable-ID manager.

    identity_mode='baseline' is an exact V2 recovery + V2 manager control.
    identity_mode='conflict' uses the new global close-association manager.
    """

    def __init__(self, tight=0.5, wide=4.0, max_missed=4,
                 wide_conf=0.3, size_ratio=2.0,
                 recovery_conf=0.7, recovery_distance=1.5,
                 duplicate_iou=0.5, duplicate_center=0.25,
                 identity_mode="conflict", byte_bonus=0.35,
                 same_id_gate=1.25, size_weight=0.10):
        super().__init__(tight=tight, wide=wide, max_missed=max_missed,
                         wide_conf=wide_conf, size_ratio=size_ratio,
                         recovery_conf=recovery_conf,
                         recovery_distance=recovery_distance,
                         duplicate_iou=duplicate_iou,
                         duplicate_center=duplicate_center)
        if identity_mode not in ("baseline", "conflict"):
            raise ValueError("identity_mode must be baseline or conflict")
        self.identity_mode = identity_mode
        if identity_mode == "conflict":
            self.manager = ConflictAwareIDManager(
                tight_frac=tight, wide_frac=wide, max_missed=max_missed,
                wide_conf=wide_conf, max_size_ratio=size_ratio,
                byte_bonus=byte_bonus, same_id_gate=same_id_gate,
                size_weight=size_weight)
        # The inherited RecoveryHybrid.update is deliberately not overridden.
