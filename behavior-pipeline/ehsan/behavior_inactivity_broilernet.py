"""Broiler-Net-INSPIRED cumulative centroid displacement, 30-minute window.

This is an experimental adaptation, not a reproduction of the Broiler-Net
implementation or a threshold sourced from that paper.

In each sample, the already-tracked Hybrid V2 boxes provide (cx, cy). For a
tracked ID, in the *most recent* window_s seconds calculate:

    A_30 = sum(hypot(cx[i]-cx[i-1], cy[i]-cy[i-1]))

Report inactive when A_30 < threshold_px (default: PROVISIONAL 360 px), and
all time/coverage/gap checks pass. No reset-on-movement timer is used (Option A).

Safeguards: require a 30-minute-old history, >=95% of expected samples by
default, a sample near the rolling window's beginning, and NO per-ID observation
gap >3 sampling intervals (6 s for interval=2). These are deliberate
engineering additions, not Broiler-Net's published threshold specifications.
A track-ID switch or frozen VIDEO can still invalidate findings. Human review
and independent validation are required before adopting the threshold.

This module does not change huddling or Hybrid V2 tracking.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil, hypot, isfinite
from statistics import median
from typing import Optional

from behavior import estimate_body_length, prune_history


AUDIT_FIELDS = [
    'track_id', 'cx', 'cy', 'history_span_s',
    'samples_in_window', 'required_samples', 'span_ok', 'samples_ok',
    'window_edge_ok', 'max_observation_gap_s', 'max_gap_ok',
    'movement_pairs', 'cumulative_activity_px', 'cumulative_activity_bl',
    'estimated_body_length_px', 'largest_step_px', 'activity_threshold_px',
    'activity_ok', 'eligible', 'inactive',
]


@dataclass
class InactivitySample:
    body_len: Optional[float]
    inactive: list
    tracked_histories: int
    audit_rows: list


class InactivityModule:
    def __init__(self, window=1800.0, interval=2.0,
                 activity_threshold_px=360.0, min_fill=0.95,
                 max_gap_s=None):
        if not all(isfinite(v) and v > 0 for v in (window, interval)):
            raise ValueError('window and interval must be finite positive numbers')
        if not isfinite(min_fill) or not 0 < min_fill <= 1:
            raise ValueError('min_fill must be between 0 and 1')
        if activity_threshold_px is not None and (not isfinite(activity_threshold_px)
                                                  or activity_threshold_px <= 0):
            raise ValueError('activity_threshold_px must be positive (or None for diagnostics)')
        if max_gap_s is not None and (not isfinite(max_gap_s) or max_gap_s < interval):
            raise ValueError('max_gap_s must be finite and at least interval')
        self.window = float(window)
        self.interval = float(interval)
        self.threshold_px = activity_threshold_px
        self.min_fill = float(min_fill)
        self.min_samples = max(2, ceil(min_fill * window / interval))
        self.max_gap_s = float(max_gap_s) if max_gap_s is not None else 3.0 * interval
        self.history = defaultdict(deque)
        self.body_lens = deque(maxlen=500)
        self.last_t = None

    def observe(self, t, ids, boxes, audit=False, body_len_override=None):
        """Process a sampled frame; ids/boxes must represent current tracked birds.

        Override body_len only when replaying precomputed observation CSVs.
        """
        t = float(t)
        if not isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('Sample timestamps must be finite and strictly increasing')
        self.last_t = t
        if len(ids) != len(boxes):
            raise ValueError('ids and boxes must be aligned')
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate tracking IDs in one sampled frame')
        for tid, (cx, cy, _w, _h) in zip(ids, boxes):
            if not (isfinite(float(cx)) and isfinite(float(cy))):
                raise ValueError('Centroid coordinates must be finite')
            self.history[tid].append((t, float(cx), float(cy)))
        prune_history(self.history, t, self.window, self.interval)

        if body_len_override is not None:
            body_len = float(body_len_override)
        else:
            if boxes:
                self.body_lens.append(estimate_body_length([(b[2], b[3]) for b in boxes]))
            body_len = sorted(self.body_lens)[len(self.body_lens)//2] if self.body_lens else None
        if body_len is not None and (not isfinite(body_len) or body_len <= 0):
            raise ValueError('Body length must be finite and positive when supplied')

        inactive = []
        audit_rows = []
        for tid, (cx, cy, _w, _h) in zip(ids, boxes):
            hist = self.history[tid]
            recent = [p for p in hist if p[0] >= t - self.window - 1e-8]
            span = t - hist[0][0]
            span_ok = span >= self.window - 1e-7
            samples_ok = len(recent) >= self.min_samples
            # Require data near the earliest edge, not merely a stale old point.
            edge_ok = bool(recent and recent[0][0] <= t-self.window+1.5*self.interval)
            gaps = [b[0] - a[0] for a,b in zip(recent, recent[1:])]
            max_gap = max(gaps, default=0.0)
            gap_ok = bool(gaps) and max_gap <= self.max_gap_s + 1e-7
            steps = [hypot(b[1]-a[1],b[2]-a[2]) for a,b in zip(recent, recent[1:])]
            activity_px = sum(steps)
            activity_bl = activity_px/body_len if body_len else None  # audit only
            activity_ok = (activity_px < self.threshold_px) if self.threshold_px is not None else None
            eligible = bool(span_ok and samples_ok and edge_ok and gap_ok)
            prediction = int(eligible and activity_ok) if activity_ok is not None else None
            if prediction:
                inactive.append(tid)
            if audit:
                fmt=lambda value: '' if value is None else f'{value:.3f}'
                audit_rows.append([
                    tid, f'{cx:.2f}', f'{cy:.2f}', fmt(span), len(recent),
                    self.min_samples, int(span_ok), int(samples_ok), int(edge_ok),
                    fmt(max_gap), int(gap_ok), len(steps), fmt(activity_px),
                    fmt(activity_bl), fmt(body_len), fmt(max(steps,default=0.0)),
                    fmt(self.threshold_px), '' if activity_ok is None else int(activity_ok),
                    int(eligible), '' if prediction is None else prediction,
                ])
        return InactivitySample(body_len, inactive, len(self.history), audit_rows)
