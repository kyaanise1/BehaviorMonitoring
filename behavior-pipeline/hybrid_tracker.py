"""Broiler multi-object tracking: ByteTrack, centroid, and an experimental hybrid.

HYBRID DESIGN (proposed integration of thesis Section 3.3.2.1 / Figure 3.5):
    YOLOv11 -> ByteTrack associations -> centroid-based ID continuity -> behavior.

Hybrid changes track IDs only. It PRESERVES ByteTrack's box coordinates, number of
boxes, and detection confidences. Thus it cannot restore detections that ByteTrack
has already discarded. Compare it against BOTH standalone trackers using MOT metrics.

This is a simple heuristic centroid/ByteTrack integration, NOT an implementation of
the full ODBO identity connector or an appearance-based re-identification network.
"""
import math
import statistics
from collections import deque


class _Track:
    __slots__ = ("id", "cx", "cy", "w", "h", "missed", "hits")

    def __init__(self, tid, cx, cy, w, h):
        self.id, self.cx, self.cy, self.w, self.h = tid, cx, cy, w, h
        self.missed, self.hits = 0, 1


class CentroidTracker:
    """Original two-pass centroid tracker, unchanged for baseline comparisons."""

    def __init__(self, tight_frac=0.5, wide_frac=4.0, max_missed=4,
                 start_conf=0.6, wide_conf=0.3, body_window=200):
        self.tight_frac = tight_frac
        self.wide_frac = wide_frac
        self.max_missed = max_missed
        self.start_conf = start_conf
        self.wide_conf = wide_conf
        self.tracks = {}
        self.next_id = 1
        self._body = deque(maxlen=body_window)

    def body_length(self):
        return statistics.median(self._body) if self._body else None

    @staticmethod
    def _greedy(track_ids, det_ids, tracks, dets, limit):
        """Distance-first, one-to-one greedy matching."""
        pairs = []
        for ti in track_ids:
            t = tracks[ti]
            for di in det_ids:
                d = math.hypot(t.cx - dets[di][0], t.cy - dets[di][1])
                if d <= limit:
                    pairs.append((d, ti, di))
        pairs.sort()
        used_t, used_d, matches = set(), set(), {}
        for _, ti, di in pairs:
            if ti not in used_t and di not in used_d:
                used_t.add(ti)
                used_d.add(di)
                matches[ti] = di
        return matches

    def update(self, dets):
        """dets: [(cx, cy, w, h, confidence), ...]."""
        if dets:
            ref = [d for d in dets if d[4] >= self.start_conf] or dets
            self._body.append(statistics.median(max(d[2], d[3]) for d in ref))
        body = self.body_length()
        if body is None:
            return []

        tids = list(self.tracks)
        dids = list(range(len(dets)))
        matched = self._greedy(tids, dids, self.tracks, dets, self.tight_frac * body)
        used_d = set(matched.values())
        rem_t = [t for t in tids if t not in matched]
        rem_d = [d for d in dids if d not in used_d and dets[d][4] >= self.wide_conf]
        matched.update(self._greedy(rem_t, rem_d, self.tracks, dets, self.wide_frac * body))

        out = []
        used_d = set(matched.values())
        for tid, did in matched.items():
            cx, cy, w, h, c = dets[did]
            tr = self.tracks[tid]
            tr.cx, tr.cy, tr.w, tr.h = cx, cy, w, h
            tr.missed = 0
            tr.hits += 1
            out.append((tid, cx, cy, w, h, c))
        for tid in tids:
            if tid not in matched:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        for did in dids:
            if did not in used_d and dets[did][4] >= self.start_conf:
                cx, cy, w, h, c = dets[did]
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = _Track(tid, cx, cy, w, h)
                out.append((tid, cx, cy, w, h, c))
        return out


class _HybridTrack(_Track):
    __slots__ = ("byte_id",)

    def __init__(self, tid, byte_id, cx, cy, w, h):
        super().__init__(tid, cx, cy, w, h)
        self.byte_id = int(byte_id)


class HybridIDManager:
    """Assign stable IDs to ByteTrack outputs, using centroid recovery when needed.

    Three ONE-TO-ONE matching passes, on the same frame:
      1) Tight centroid matching: preserve stationary birds even when ByteTrack ID
         changes (or two ByteTrack IDs are swapped).
      2) ByteTrack ID continuity: trust the same ByteTrack ID only when spatially
         plausible; this retains ByteTrack's motion association for moving birds.
      3) Wide centroid fallback: rejoin unmatched/fragmented tracks by distance.

    Unmatched stable tracks persist for max_missed *processed* frames. Any new
    ByteTrack output receives a new stable ID, even if its score is low, so the
    hybrid preserves all ByteTrack detections (fair comparison of ID quality).

    Limitations: proximity does not prove identity, particularly in crowded birds;
    this cannot recover missing ByteTrack boxes or guarantee 30-min identity.

    Optional pass 4, "refind" (refind_missed > 0; 0 = off, the behaviour above
    exactly): in a closed pen a bird whose stable ID ran out after max_missed
    frames was almost always hidden or missed by the detector, not gone, and a
    new ID restarts its 30-minute inactivity clock (the Day 12 non-gap audit:
    nearly every track ends never_spanned_full_30min_window). With refind on, a
    stable ID missed for more than max_missed frames is kept up to
    refind_missed frames, but it only competes in a LAST pass, for the boxes no
    seen bird claimed (those that would otherwise get a brand-new ID), within
    refind_frac body lengths (default: wide_frac). It never takes a box from a
    bird that is being seen, and RecoveryHybrid still only revives IDs missed
    for at most max_missed frames.
    """

    def __init__(self, tight_frac=0.5, wide_frac=4.0, max_missed=4,
                 wide_conf=0.3, max_size_ratio=2.0, body_window=200,
                 refind_missed=0, refind_frac=None):
        if not (0 < tight_frac <= wide_frac):
            raise ValueError("Require 0 < tight_frac <= wide_frac")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        if refind_missed and refind_missed <= max_missed:
            raise ValueError("refind_missed must be 0 (off) or more than max_missed")
        self.refind_missed = int(refind_missed)
        self.refind_frac = float(refind_frac if refind_frac is not None else wide_frac)
        if max_size_ratio < 1:
            raise ValueError("max_size_ratio must be >= 1")
        self.tight_frac = float(tight_frac)
        self.wide_frac = float(wide_frac)
        self.max_missed = int(max_missed)
        self.wide_conf = float(wide_conf)
        self.max_size_ratio = float(max_size_ratio)
        self._body = deque(maxlen=body_window)
        self.tracks = {}
        self.next_id = 1

    def _valid_size(self, tr, box):
        _, _, w, h = box
        if min(tr.w, tr.h, w, h) <= 0:
            return False
        # Check both dimensions; don't re-ID small birds as huge overlapping blobs.
        return (max(tr.w / w, w / tr.w, tr.h / h, h / tr.h)
                <= self.max_size_ratio)

    def _dist(self, tr, box):
        cx, cy = box[:2]
        return math.hypot(tr.cx - cx, tr.cy - cy)

    def _greedy(self, tids, dids, boxes, body, frac):
        candidates = []
        for tid in tids:
            tr = self.tracks[tid]
            for di in dids:
                if self._valid_size(tr, boxes[di]):
                    distance = self._dist(tr, boxes[di])
                    if distance <= frac * body:
                        candidates.append((distance, tid, di))
        candidates.sort()
        out = {}
        used_d = set()
        for _, tid, di in candidates:
            if tid not in out and di not in used_d:
                out[tid] = di
                used_d.add(di)
        return out

    def update(self, byte_ids, xywh, scores):
        """Input ByteTrack IDs/boxes/scores; return (stable_ids, xywh, scores).

        All returned rows retain the same order and boxes as the ByteTrack input.
        """
        if not (len(byte_ids) == len(xywh) == len(scores)):
            raise ValueError("byte_ids, xywh, scores must be equal-length")
        if len(set(int(v) for v in byte_ids)) != len(byte_ids):
            raise ValueError("ByteTrack returned duplicate IDs in a single frame")

        boxes = [tuple(map(float, box)) for box in xywh]
        if boxes:
            body_values = [max(box[2], box[3]) for box in boxes if box[2] > 0 and box[3] > 0]
            if body_values:
                self._body.append(statistics.median(body_values))
        body = statistics.median(self._body) if self._body else 1.0

        all_tids = list(self.tracks)
        # Passes 1-3 see only IDs missed for at most max_missed frames (all IDs, when refind is off).
        tids = [tid for tid in all_tids if self.tracks[tid].missed <= self.max_missed]
        dormant = [tid for tid in all_tids if self.tracks[tid].missed > self.max_missed]
        dids = list(range(len(boxes)))
        matched = self._greedy(tids, dids, boxes, body, self.tight_frac)

        # Pass 2: use ByteTrack's own association as a cue, but reject huge jumps.
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

        # Pass 3: centroid rescue for ByteTrack IDs that were lost or changed.
        rem_t = [tid for tid in tids if tid not in matched]
        rem_d = [di for di in dids if di not in used_d and scores[di] >= self.wide_conf]
        matched.update(self._greedy(rem_t, rem_d, boxes, body, self.wide_frac))

        # Pass 4 (refind, off unless refind_missed > 0): long-missing IDs, only on unclaimed boxes.
        if dormant:
            used_d = set(matched.values())
            left = [di for di in dids if di not in used_d and scores[di] >= self.wide_conf]
            matched.update(self._greedy(dormant, left, boxes, body, self.refind_frac))

        stable = [None] * len(boxes)
        for tid, di in matched.items():
            tr = self.tracks[tid]
            cx, cy, w, h = boxes[di]
            tr.cx, tr.cy, tr.w, tr.h = cx, cy, w, h
            tr.byte_id = int(byte_ids[di])
            tr.missed = 0
            tr.hits += 1
            stable[di] = tid

        # Count missing processed frames only, not elapsed wall-clock seconds.
        limit = self.refind_missed or self.max_missed
        for tid in all_tids:
            if tid not in matched:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > limit:
                    del self.tracks[tid]

        # Keep all ByteTrack boxes, rather than hiding unmatched predictions.
        for di in dids:
            if stable[di] is None:
                tid = self.next_id
                self.next_id += 1
                cx, cy, w, h = boxes[di]
                self.tracks[tid] = _HybridTrack(tid, byte_ids[di], cx, cy, w, h)
                stable[di] = tid

        assert all(sid is not None for sid in stable)
        assert len(set(stable)) == len(stable), "One stable ID was assigned twice in a frame"
        return stable, [list(box) for box in boxes], list(scores)


def add_tracker_args(ap):
    """Add tracker choice and centroid/hybrid tuning options to argparse."""
    ap.add_argument("--tracker-type", choices=["bytetrack", "centroid", "hybrid"],
                    default="bytetrack")
    ap.add_argument("--tight", type=float, default=0.5,
                    help="centroid/hybrid: tight matching distance in body lengths")
    ap.add_argument("--wide", type=float, default=4.0,
                    help="centroid/hybrid: max recovery distance in body lengths")
    ap.add_argument("--max-missed", type=int, default=4,
                    help="centroid/hybrid: frames a lost ID survives")
    ap.add_argument("--start-conf", type=float, default=0.6,
                    help="standalone centroid: confidence for creating new tracks")
    ap.add_argument("--wide-conf", type=float, default=0.3,
                    help="centroid/hybrid: min confidence for wide recovery")
    ap.add_argument("--hybrid-size-ratio", type=float, default=2.0,
                    help="hybrid: maximum width/height ratio change during association")
    ap.add_argument("--refind-missed", type=int, default=0,
                    help="hybrid: frames a long-missing ID may still be found again on an unclaimed box (0 = off)")
    ap.add_argument("--refind-frac", type=float, default=None,
                    help="hybrid: refind distance in body lengths (default: --wide)")


def make_runner(args, model):
    """Return track(frame) -> (ids, xywh, confidences) for selected tracker."""
    kind = getattr(args, "tracker_type", "bytetrack")
    conf, imgsz = args.conf, args.imgsz

    if kind in ("bytetrack", "hybrid"):
        hybrid = None
        if kind == "hybrid":
            hybrid = HybridIDManager(
                tight_frac=getattr(args, "tight", 0.5),
                wide_frac=getattr(args, "wide", 4.0),
                max_missed=getattr(args, "max_missed", 4),
                wide_conf=getattr(args, "wide_conf", 0.3),
                max_size_ratio=getattr(args, "hybrid_size_ratio", 2.0),
                refind_missed=getattr(args, "refind_missed", 0),
                refind_frac=getattr(args, "refind_frac", None),
            )

        def run(frame):
            r = model.track(frame, tracker=args.tracker, conf=conf, imgsz=imgsz,
                            persist=True, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                byte_ids, xywh, scores = [], [], []
            else:
                byte_ids = r.boxes.id.int().tolist()
                xywh = r.boxes.xywh.tolist()
                scores = r.boxes.conf.tolist()
            if hybrid is not None:
                return hybrid.update(byte_ids, xywh, scores)
            return byte_ids, xywh, scores
        return run

    if kind != "centroid":
        raise ValueError(f"Unsupported tracker type: {kind}")

    trk = CentroidTracker(
        tight_frac=getattr(args, "tight", 0.5),
        wide_frac=getattr(args, "wide", 4.0),
        max_missed=getattr(args, "max_missed", 4),
        start_conf=getattr(args, "start_conf", 0.6),
        wide_conf=getattr(args, "wide_conf", 0.3),
    )

    def run(frame):
        r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for (cx, cy, w, h), c in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
                dets.append((cx, cy, w, h, c))
        out = trk.update(dets)
        return [o[0] for o in out], [list(o[1:5]) for o in out], [o[5] for o in out]
    return run
