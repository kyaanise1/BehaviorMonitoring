"""Option B: consecutive centroid stillness with 5/15/30-minute warning markers.

This is an EXPERIMENTAL, separately validated alternative to Option A.
- No accumulated-path threshold (450 px has no role here).
- A bird remains in one stillness streak while its *causally smoothed* centroid
  stays within tolerance_bl * its own median apparent body length of an ANCHOR.
- A sustained shift beyond the anchor tolerance resets the timer.
- A large sudden jump (possible ID switch) also invalidates the current result.
- Observation gaps longer than max_gap_s reset the streak (never infer stillness).
- Yellow if >5 minutes, Orange if >15 minutes, Red if >30 minutes.
- At any marker threshold, minimum expected observation coverage is required.
- The tolerance 0.5 body lengths is a starting heuristic, NOT validated.

This is centroid positional stillness, not proof of absence of body motion.
"""
from collections import deque
from dataclasses import dataclass, field
from math import floor, hypot, isfinite
from statistics import median
from typing import Dict, Optional

from behavior import estimate_body_length

AUDIT_FIELDS = [
    'track_id', 'cx', 'cy', 'smoothed_cx', 'smoothed_cy',
    'individual_body_length_px', 'movement_tolerance_px',
    'stillness_anchor_x', 'stillness_anchor_y', 'distance_from_anchor_px',
    'still_since_s', 'stillness_duration_s', 'stillness_minutes',
    'samples_in_streak', 'expected_samples_in_streak', 'streak_fill_ratio',
    'observation_gap_s', 'raw_step_px', 'possible_id_jump',
    'significant_movement', 'reset_reason', 'reset_count',
    'marker', 'inactive', 'smoothing_method',
]


@dataclass
class _Track:
    last_seen: float
    last_raw: tuple
    samples: deque = field(default_factory=lambda: deque(maxlen=5))
    body_lengths: deque = field(default_factory=lambda: deque(maxlen=30))
    anchor: Optional[tuple] = None
    since: float = 0.0
    count: int = 0
    resets: int = 0


@dataclass
class InactivitySample:
    body_len: Optional[float]
    inactive: list
    tracked_histories: int
    audit_rows: list
    markers_by_id: Dict[int, str]
    durations_by_id: Dict[int, float]
    counts_by_marker: Dict[str, int]


class InactivityModule:
    def __init__(self, window=1800.0, interval=2.0, stillness_tolerance_bl=0.5,
                 min_fill=0.95, max_gap_s=None, smoothing='median3',
                 jump_audit_bodylengths=1.5):
        for label, val in (('window', window), ('interval', interval),
                           ('stillness_tolerance_bl', stillness_tolerance_bl),
                           ('jump_audit_bodylengths', jump_audit_bodylengths)):
            if not isfinite(float(val)) or float(val) <= 0:
                raise ValueError(f'{label} must be finite and positive')
        if window < 900:
            raise ValueError('window (red cutoff) must be at least 900 seconds')
        if not isfinite(min_fill) or not (0 < min_fill <= 1):
            raise ValueError('min_fill must be within (0,1]')
        if smoothing not in ('raw', 'median3', 'median5'):
            raise ValueError('smoothing must be raw, median3, or median5')
        gap = 3*interval if max_gap_s is None else max_gap_s
        if not isfinite(float(gap)) or gap < interval:
            raise ValueError('max_gap_s must be >= interval')
        self.window = float(window)
        self.interval = float(interval)
        self.tolerance_bl = float(stillness_tolerance_bl)
        self.min_fill = float(min_fill)
        self.max_gap_s = float(gap)
        self.smoothing = smoothing
        self.smoothing_n = {'raw': 1, 'median3': 3, 'median5': 5}[smoothing]
        self.jump_bl = float(jump_audit_bodylengths)
        self.tracks = {}
        self.body_lens = deque(maxlen=500)  # flock size solely for unchanged huddling
        self.last_t = None

    @staticmethod
    def _fmt(x):
        return '' if x is None else f'{x:.3f}'

    def observe(self, t, ids, boxes, audit=False):
        """Sample current tracked detections; boxes are (cx, cy, width, height)."""
        t = float(t)
        if not isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('timestamps must increase strictly and be finite')
        if len(ids) != len(boxes) or len(set(ids)) != len(ids):
            raise ValueError('IDs and boxes mismatch or contain duplicate IDs')
        self.last_t = t
        for b in boxes:
            if len(b) != 4 or not all(isfinite(float(v)) for v in b):
                raise ValueError('each box must contain four finite values')
            if b[2] <= 0 or b[3] <= 0:
                raise ValueError('box width and height must be positive')

        if boxes:
            self.body_lens.append(estimate_body_length([(b[2],b[3]) for b in boxes]))
        flock_body_len = median(self.body_lens) if self.body_lens else None
        for tid, tr in list(self.tracks.items()):
            if t - tr.last_seen > max(self.window * 2, 3600):
                del self.tracks[tid]

        inactive, audit_rows, markers, durations = [], [], {}, {}
        counts = {'NONE': 0, 'YELLOW': 0, 'ORANGE': 0, 'RED': 0, 'UNDETERMINED': 0}
        for tid, box in zip(ids, boxes):
            cx, cy, w, h = map(float, box)
            point = (cx, cy)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = _Track(last_seen=t, last_raw=point)
                self.tracks[tid] = tr
                tr.since = t
                tr.count = 0
            gap = t - tr.last_seen
            raw_step = hypot(cx-tr.last_raw[0], cy-tr.last_raw[1]) if tr.count else 0.0
            tr.body_lengths.append(max(w,h))
            local_body_len = median(tr.body_lengths)
            tolerance = self.tolerance_bl * local_body_len
            jump = bool(tr.count and raw_step > self.jump_bl * local_body_len)
            reset_reason = ''
            significant_move = False
            if tr.count and gap > self.max_gap_s + 1e-7:
                reset_reason = 'observation_gap'
            elif jump:
                reset_reason = 'possible_id_jump'
            if reset_reason:
                tr.samples.clear()
                tr.resets += 1
                tr.since = t
                tr.count = 0
            tr.samples.append(point)
            n = min(len(tr.samples), self.smoothing_n)
            last_n = list(tr.samples)[-n:]
            smoothed = (median(p[0] for p in last_n), median(p[1] for p in last_n))
            if tr.anchor is None or reset_reason:
                tr.anchor = smoothed
                tr.since = t
                tr.count = 1
                distance = 0.0
            else:
                distance = hypot(smoothed[0]-tr.anchor[0], smoothed[1]-tr.anchor[1])
                if distance > tolerance:
                    reset_reason = 'significant_movement'
                    significant_move = True
                    tr.resets += 1
                    tr.samples.clear()
                    tr.samples.append(point)
                    smoothed = point
                    tr.anchor = smoothed
                    tr.since = t
                    tr.count = 1
                    distance = 0.0
                else:
                    tr.count += 1
            tr.last_seen = t
            tr.last_raw = point
            duration = max(0, t - tr.since)
            expected = floor(duration / self.interval + 1e-6) + 1
            coverage = min(1.0, tr.count / max(1, expected))
            if reset_reason == 'possible_id_jump':
                marker = 'UNDETERMINED'
            elif duration > 300 and coverage + 1e-9 < self.min_fill:
                marker = 'UNDETERMINED'
            elif duration > self.window:
                marker = 'RED'
            elif duration > 900:
                marker = 'ORANGE'
            elif duration > 300:
                marker = 'YELLOW'
            else:
                marker = 'NONE'
            if marker == 'RED':
                inactive.append(tid)
            counts[marker] += 1
            markers[tid] = marker
            durations[tid] = duration
            if audit:
                audit_rows.append([
                    tid, self._fmt(cx), self._fmt(cy),
                    self._fmt(smoothed[0]), self._fmt(smoothed[1]),
                    self._fmt(local_body_len), self._fmt(tolerance),
                    self._fmt(tr.anchor[0]), self._fmt(tr.anchor[1]),
                    self._fmt(distance), self._fmt(tr.since),
                    self._fmt(duration), self._fmt(duration/60),
                    tr.count, expected, self._fmt(coverage), self._fmt(gap),
                    self._fmt(raw_step), int(jump), int(significant_move),
                    reset_reason, tr.resets, marker, int(marker == 'RED'), self.smoothing,
                ])
        return InactivitySample(flock_body_len, inactive, len(self.tracks),
                                audit_rows, markers, durations, counts)
