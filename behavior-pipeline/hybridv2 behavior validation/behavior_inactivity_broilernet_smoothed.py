"""Option A: Broiler-Net-inspired rolling cumulative centroid displacement.

Designed as a DROP-IN ALTERNATIVE to behavior_inactivity_broilernet.py for
Hybrid V2's existing stable track IDs (no YOLO / tracker / huddling edits).

Activity = sum of successive Euclidean distances over a rolling 30-min window.
Optional median filter acts on centroid x and y separately BEFORE distances
are summed. Uses a centered 3- or 5-observation median within the available
past/current window. This uses no future VIDEO frames; the newest observation
uses a truncated neighborhood. It intentionally suppresses short movements,
so the choice and threshold require independent validation.

Not the published Broiler-Net threshold. Threshold 450 px is PROVISIONAL.

smoothing='block30' (added on bytetrack-tuning, 17 Nov 2026) measures activity
so that one threshold means the same thing at any sampling rate and bird size:
the centre is averaged over 30-s blocks (this removes detector jitter, which a
step-by-step sum adds up once per picture), the path between block means is
summed, and it is divided by the flock's median body length. The decision is
activity_bl < activity_threshold_bl (default 4.0 body lengths per 30 min).
Why: on the original Day 12 mkv the same still bird (15) scores 551 / 373 /
302 px at 2 / 4 / 6 s sampling with median3, but 1.89 / 2.00 / 2.08 BL with
block30; the next stillest bird scores 9.4 BL. 4.0 BL is PROVISIONAL: choose
it on one day and test it on another, on the original video.
Never treats missing observations as stationary, and never interpolates gaps.
Suspected ID switches are AUDITED, not silently removed or corrected.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil, hypot, isfinite
from statistics import median
from typing import Optional

from behavior import estimate_body_length, prune_history

# Original audit field names/order preserved for compatibility.
_BASE_FIELDS = [
    'track_id', 'cx', 'cy', 'history_span_s',
    'samples_in_window', 'required_samples', 'span_ok', 'samples_ok',
    'window_edge_ok', 'max_observation_gap_s', 'max_gap_ok',
    'movement_pairs', 'cumulative_activity_px', 'cumulative_activity_bl',
    'estimated_body_length_px', 'largest_step_px', 'activity_threshold_px',
    'activity_ok', 'eligible', 'inactive',
]
AUDIT_FIELDS = _BASE_FIELDS + [
    'smoothing_method', 'raw_activity_px', 'median3_activity_px',
    'median5_activity_px', 'max_raw_step_px', 'possible_id_jump',
    'activity_difference_from_raw_px',
    'cumulative_activity_level_px',  # user-facing alias for cumulative_activity_px
    'activity_threshold_bl', 'block_activity_bl', 'range_bl',
]


def _median_filtered(coords, size):
    """Apply median to x,y independently, including endpoint neighborhoods.

    Only samples inside the current rolling window are used. With size 3,
    center sample i is compared against i-1, i, i+1 if each is available.
    Boundary uses 2 samples, following Python statistics.median semantics.
    """
    if size == 1:
        return list(coords)
    half = size // 2
    out = []
    for i in range(len(coords)):
        lo, hi = max(0, i-half), min(len(coords), i+half+1)
        out.append((median(coords[j][0] for j in range(lo, hi)),
                    median(coords[j][1] for j in range(lo, hi))))
    return out


def _block_means(points, block_s):
    """Mean (x, y) of the points in each block_s-second block, in time order; points are (t, x, y)."""
    blocks = defaultdict(list)
    for t, x, y in points:
        blocks[int(t // block_s)].append((x, y))
    return [(sum(p[0] for p in b) / len(b), sum(p[1] for p in b) / len(b))
            for _, b in sorted(blocks.items())]


def _path_length(coords):
    return sum(hypot(b[0]-a[0], b[1]-a[1])
               for a, b in zip(coords, coords[1:]))


@dataclass
class InactivitySample:
    body_len: Optional[float]
    inactive: list
    tracked_histories: int
    audit_rows: list
    eligible_activity_by_id_px: dict  # full 30-min eligible, currently visible birds only


class InactivityModule:
    def __init__(self, window=1800.0, interval=2.0,
                 activity_threshold_px=450.0, min_fill=0.95,
                 max_gap_s=None, smoothing='median3',
                 jump_audit_bodylengths=1.5, activity_threshold_bl=None,
                 block_s=30.0):
        if not all(isfinite(float(v)) and float(v) > 0 for v in (window, interval)):
            raise ValueError('window and interval must be finite and positive')
        if not isfinite(min_fill) or not 0 < min_fill <= 1:
            raise ValueError('min_fill must be in (0,1]')
        if smoothing not in ('raw', 'median3', 'median5', 'block30'):
            raise ValueError('smoothing must be raw, median3, median5 or block30')
        if activity_threshold_bl is not None and (
                not isfinite(activity_threshold_bl) or activity_threshold_bl <= 0):
            raise ValueError('activity_threshold_bl must be positive or None')
        if not isfinite(block_s) or block_s < interval:
            raise ValueError('block_s must be >= interval')
        if activity_threshold_px is not None and (
                not isfinite(activity_threshold_px) or activity_threshold_px <= 0):
            raise ValueError('activity_threshold_px must be positive or None')
        if max_gap_s is not None and (
                not isfinite(max_gap_s) or max_gap_s < interval):
            raise ValueError('max_gap_s must be >= interval')
        if not isfinite(jump_audit_bodylengths) or jump_audit_bodylengths <= 0:
            raise ValueError('jump_audit_bodylengths must be positive')
        self.window = float(window)
        self.interval = float(interval)
        self.threshold_px = activity_threshold_px
        self.min_fill = float(min_fill)
        self.min_samples = max(2, ceil(min_fill * window / interval))
        self.max_gap_s = float(max_gap_s) if max_gap_s is not None else 3*self.interval
        self.smoothing = smoothing
        self.threshold_bl = activity_threshold_bl
        self.block_s = float(block_s)
        self.jump_audit_bodylengths = float(jump_audit_bodylengths)
        self.history = defaultdict(deque)
        self.body_lens = deque(maxlen=500)
        self.last_t = None

    def observe(self, t, ids, boxes, audit=False, body_len_override=None):
        """Boxes are (cx,cy,w,h), IDs match boxes; timestamps in seconds."""
        t = float(t)
        if not isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('timestamps must be strictly increasing and finite')
        if len(ids) != len(boxes) or len(set(ids)) != len(ids):
            raise ValueError('ids and boxes mismatch or contain duplicate IDs')
        self.last_t = t
        for tid, box in zip(ids, boxes):
            if len(box) != 4 or not all(isfinite(float(v)) for v in box):
                raise ValueError('each box must contain four finite numbers')
            self.history[tid].append((t, float(box[0]), float(box[1])))
        prune_history(self.history, t, self.window, self.interval)

        if body_len_override is not None:
            body_len = float(body_len_override)
        else:
            if boxes:
                this_len = estimate_body_length([(b[2], b[3]) for b in boxes])
                if this_len is not None:
                    self.body_lens.append(this_len)
            body_len = median(self.body_lens) if self.body_lens else None
        if body_len is not None and (not isfinite(body_len) or body_len <= 0):
            raise ValueError('body length must be finite and positive')

        inactive = []
        audit_rows = []
        eligible_activity_by_id_px = {}
        for tid, box in zip(ids, boxes):
            hist = self.history[tid]
            recent = [p for p in hist if p[0] >= t-self.window-1e-8]
            span = t-hist[0][0]
            span_ok = span >= self.window-1e-7
            samples_ok = len(recent) >= self.min_samples
            edge_ok = bool(recent and recent[0][0] <= t-self.window+1.5*self.interval)
            gaps = [b[0]-a[0] for a,b in zip(recent,recent[1:])]
            max_gap = max(gaps,default=0.0)
            gap_ok = bool(gaps) and max_gap <= self.max_gap_s+1e-7
            eligible = bool(span_ok and samples_ok and edge_ok and gap_ok)
            coords = [(p[1],p[2]) for p in recent]
            raw_steps = [hypot(b[0]-a[0],b[1]-a[1])
                         for a,b in zip(coords,coords[1:])]
            raw_activity = sum(raw_steps)
            max_raw_step = max(raw_steps,default=0.0)
            if self.smoothing == 'block30':
                selected_coords = _block_means(recent, self.block_s)
            else:
                size = {'raw':1,'median3':3,'median5':5}[self.smoothing]
                selected_coords = _median_filtered(coords,size)
            filtered_steps = [hypot(b[0]-a[0],b[1]-a[1])
                              for a,b in zip(selected_coords,selected_coords[1:])]
            activity_px = sum(filtered_steps)
            if eligible:
                eligible_activity_by_id_px[tid] = activity_px
            activity_bl = activity_px/body_len if body_len else None
            if self.threshold_bl is not None:
                activity_ok = (activity_bl < self.threshold_bl) if activity_bl is not None else None
            else:
                activity_ok = activity_px < self.threshold_px if self.threshold_px is not None else None
            prediction = int(eligible and activity_ok) if activity_ok is not None else None
            if prediction:
                inactive.append(tid)
            if audit:
                # Calculate all variants only for windows eligible for decision:
                # preserves speed during first 30 minutes of processing.
                m3 = (_path_length(_median_filtered(coords,3)) if eligible
                      else None)
                m5 = (_path_length(_median_filtered(coords,5)) if eligible
                      else None)
                jump_suspect = (int(max_raw_step > self.jump_audit_bodylengths*body_len)
                                if body_len else None)
                def f(value):
                    return '' if value is None else f'{value:.3f}'
                audit_rows.append([
                    tid, f(float(box[0])), f(float(box[1])), f(span),
                    len(recent),self.min_samples,int(span_ok),int(samples_ok),
                    int(edge_ok),f(max_gap),int(gap_ok),len(raw_steps),
                    f(activity_px),f(activity_bl),f(body_len),
                    f(max(filtered_steps,default=0.0)),f(self.threshold_px),
                    '' if activity_ok is None else int(activity_ok),
                    int(eligible),'' if prediction is None else prediction,
                    self.smoothing, f(raw_activity),f(m3),f(m5),f(max_raw_step),
                    '' if jump_suspect is None else jump_suspect,
                    f(activity_px-raw_activity),
                    f(activity_px),  # cumulative_activity_level_px (same selected score)
                    f(self.threshold_bl),
                    f(_path_length(_block_means(recent, self.block_s))/body_len) if body_len else '',
                    f(max(max(c[0] for c in coords)-min(c[0] for c in coords),
                          max(c[1] for c in coords)-min(c[1] for c in coords))/body_len)
                    if body_len and coords else '',
                ])
        return InactivitySample(body_len,inactive,len(self.history),
                                audit_rows,eligible_activity_by_id_px)
