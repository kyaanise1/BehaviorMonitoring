"""Experimental recovery layer for broiler ByteTrack + centroid hybrid.

Does not alter the original tracker.py or hybrid_tracker.py.
Step 1: Existing HybridIDManager receives ByteTrack outputs.
Step 2: Compare raw YOLO detections against emitted boxes.
Step 3: Add a high-confidence detection ONLY if it does not duplicate an
        emitted box AND it can plausibly reconnect a dormant stable ID.

This cautious variant never registers completely new stable IDs from raw-only
boxes. It cannot recover birds that have never received a stable track ID.

This is an experimental prototype, not proven improvement or deployable
single-inference implementation. No ground truth enters runtime inference.
"""
import math
import statistics
from collections import deque

from hybrid_tracker import HybridIDManager


def _iou(a, b):
    """IoU for two (cx,cy,w,h) boxes."""
    ax, ay, aw, ah = map(float, a[:4])
    bx, by, bw, bh = map(float, b[:4])
    al, at, ar, ab = ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2
    bl, bt, br, bb = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2
    w = max(0.0, min(ar,br)-max(al,bl))
    h = max(0.0, min(ab,bb)-max(at,bt))
    inter = w*h
    union = max(0.0,aw*ah) + max(0.0,bw*bh) - inter
    return inter/union if union>0 else 0.0


class RecoveryHybrid:
    """Wrap current stable-ID manager and recover credible omitted detections.

    Incoming *tracked* detections are authoritative. Recovered detections are
    assigned only to dormant stable IDs, never to an ID already used this frame.
    All distances are based on body length estimated by the baseline manager.
    """

    def __init__(self, tight=0.5, wide=4.0, max_missed=4,
                 wide_conf=0.3, size_ratio=2.0,
                 recovery_conf=0.7, recovery_distance=1.5,
                 duplicate_iou=0.5, duplicate_center=0.25,
                 refind_missed=0, refind_frac=None):
        if not 0 <= recovery_conf <= 1:
            raise ValueError("recovery_conf must be between 0 and 1")
        if not 0 < recovery_distance:
            raise ValueError("recovery_distance must be positive")
        if not 0 < duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (0,1]")
        self.manager = HybridIDManager(tight_frac=tight, wide_frac=wide,
                                       max_missed=max_missed, wide_conf=wide_conf,
                                       max_size_ratio=size_ratio,
                                       refind_missed=refind_missed, refind_frac=refind_frac)
        self.recovery_conf = recovery_conf
        self.recovery_distance = recovery_distance
        self.duplicate_iou = duplicate_iou
        self.duplicate_center = duplicate_center
        self.frames = 0
        self.recovered_total = 0

    def update(self, byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores):
        if len(raw_boxes) != len(raw_scores):
            raise ValueError("raw_boxes and raw_scores must have equal length")
        stable_ids, boxes, scores = self.manager.update(byte_ids, byte_boxes, byte_scores)
        stable_ids, boxes, scores = list(stable_ids), list(boxes), list(scores)
        present = set(stable_ids)

        # Last-known body length was already updated by normal hybrid tracking.
        body = statistics.median(self.manager._body) if self.manager._body else 1.0
        # Filter raw boxes already represented by a ByteTrack output.
        # Center distance is also used to suppress near-identical duplicates.
        raw_candidates = []
        for i, (box, score) in enumerate(zip(raw_boxes, raw_scores)):
            if float(score) < self.recovery_conf:
                continue
            b = list(map(float, box))
            if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                continue
            overlaps = any(
                _iou(b, existing) >= self.duplicate_iou or
                math.hypot(b[0]-existing[0], b[1]-existing[1]) <= self.duplicate_center*body
                for existing in boxes
            )
            if not overlaps:
                raw_candidates.append((i,b,float(score)))

        # Only stable tracks not already used this frame can be recovered.
        dormant = [tid for tid,tr in self.manager.tracks.items()
                   if tid not in present and tr.missed > 0 and tr.missed <= self.manager.max_missed]
        pairs = []
        for tid in dormant:
            tr = self.manager.tracks[tid]
            for i,b,score in raw_candidates:
                if not self.manager._valid_size(tr,b):
                    continue
                distance = math.hypot(tr.cx-b[0],tr.cy-b[1])
                if distance <= self.recovery_distance*body:
                    # Nearest first; small confidence preference only for exact ties.
                    pairs.append((distance,-score,tid,i,b,score))
        pairs.sort()
        used_d=set()
        recovered=0
        for _,_,tid,i,b,score in pairs:
            if tid in present or i in used_d:
                continue
            # Keep recovered boxes distinct from already accepted boxes.
            if any(_iou(b,prev)>=self.duplicate_iou for prev in boxes):
                continue
            tr = self.manager.tracks[tid]
            tr.cx,tr.cy,tr.w,tr.h=b
            tr.missed=0
            tr.hits+=1
            # Do not alter tr.byte_id: this YOLO box had no ByteTrack ID.
            stable_ids.append(tid)
            boxes.append(b)
            scores.append(score)
            present.add(tid)
            used_d.add(i)
            recovered+=1
        assert len(stable_ids) == len(set(stable_ids)), "Duplicate stable IDs in one frame"
        self.frames+=1
        self.recovered_total+=recovered
        return stable_ids,boxes,scores,recovered
