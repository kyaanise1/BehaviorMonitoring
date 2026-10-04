"""Two-pass centroid tracker (alternative to ByteTrack) + a runner used by all scripts.

Why: at 5-second frame gaps a running bird's new box may not overlap its old box
at all, so IoU matching (ByteTrack) cannot link them. This tracker matches by
distance between box centers, in two passes:

  Pass 1 (tight): link each track to the nearest detection within ~0.5 body length.
                  This claims all the static birds first.
  Pass 2 (wide) : link the leftovers within ~4 body lengths. With the static birds
                  already taken, a runner's old track can only match its new position.

Unmatched tracks survive `max_missed` frames (the "counter"), then are deleted.
Unmatched confident detections (conf >= start_conf) start new tracks.
YOLO still only detects; this file assigns the IDs.
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
    def __init__(self, tight_frac=0.5, wide_frac=4.0, max_missed=4,
                 start_conf=0.6, wide_conf=0.3, body_window=200):
        self.tight_frac = tight_frac      # pass-1 limit, in body lengths
        self.wide_frac = wide_frac        # pass-2 limit, in body lengths
        self.max_missed = max_missed      # frames a lost track survives
        self.start_conf = start_conf      # min confidence to start a new track
        self.wide_conf = wide_conf        # min confidence for a pass-2 match
        self.tracks = {}
        self.next_id = 1
        self._body = deque(maxlen=body_window)

    def body_length(self):
        return statistics.median(self._body) if self._body else None

    @staticmethod
    def _greedy(track_ids, det_ids, tracks, dets, limit):
        """Globally nearest pairs first; each track/detection used at most once."""
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
        """dets: [(cx, cy, w, h, conf), ...] in pixels.
        Returns [(id, cx, cy, w, h, conf), ...] for tracks seen in this frame."""
        if dets:
            ref = [d for d in dets if d[4] >= self.start_conf] or dets
            self._body.append(statistics.median(max(d[2], d[3]) for d in ref))
        body = self.body_length()
        if body is None:
            return []

        tids = list(self.tracks)
        dids = list(range(len(dets)))

        # Pass 1: tight limit, every detection allowed
        matched = self._greedy(tids, dids, self.tracks, dets, self.tight_frac * body)

        # Pass 2: wide limit on the leftovers (skip very weak detections)
        used_d = set(matched.values())
        rem_t = [t for t in tids if t not in matched]
        rem_d = [d for d in dids if d not in used_d and dets[d][4] >= self.wide_conf]
        matched.update(self._greedy(rem_t, rem_d, self.tracks, dets, self.wide_frac * body))

        out = []
        used_d = set(matched.values())

        for tid, did in matched.items():              # update matched tracks
            cx, cy, w, h, c = dets[did]
            tr = self.tracks[tid]
            tr.cx, tr.cy, tr.w, tr.h = cx, cy, w, h
            tr.missed = 0
            tr.hits += 1
            out.append((tid, cx, cy, w, h, c))

        for tid in tids:                              # counter-based deletion
            if tid not in matched:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        for did in dids:                              # new IDs for confident leftovers
            if did not in used_d and dets[did][4] >= self.start_conf:
                cx, cy, w, h, c = dets[did]
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = _Track(tid, cx, cy, w, h)
                out.append((tid, cx, cy, w, h, c))
        return out


def add_tracker_args(ap):
    """Adds the tracker-choice options to an argparse parser."""
    ap.add_argument("--tracker-type", choices=["bytetrack", "centroid"], default="bytetrack")
    ap.add_argument("--tight", type=float, default=0.5, help="centroid pass 1 limit (body lengths)")
    ap.add_argument("--wide", type=float, default=4.0, help="centroid pass 2 limit (body lengths)")
    ap.add_argument("--max-missed", type=int, default=4, help="centroid: frames a lost track survives")
    ap.add_argument("--start-conf", type=float, default=0.6, help="centroid: min conf to start a track")
    ap.add_argument("--wide-conf", type=float, default=0.3, help="centroid: min conf for a pass-2 match")


def make_runner(args, model):
    """Return track(frame) -> (ids, xywh, confs) for the chosen tracker.
    xywh rows are [cx, cy, w, h] in pixels."""
    kind = getattr(args, "tracker_type", "bytetrack")
    conf, imgsz = args.conf, args.imgsz

    if kind == "bytetrack":
        def run(frame):
            r = model.track(frame, tracker=args.tracker, conf=conf, imgsz=imgsz,
                            persist=True, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                return [], [], []
            return r.boxes.id.int().tolist(), r.boxes.xywh.tolist(), r.boxes.conf.tolist()
        return run

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