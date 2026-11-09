"""Original Hybrid V2 inactivity rule from run_pipeline_hybrid_v2(3).py.

Accepts one shared Hybrid V2 tracking stream from run_behavior_system.py.
Uses the ORIGINAL behavior.py functions and default values. No freeze/gap
classification, automatic tracker resets, or warm-up suppression.

A continuous, correctly timed source and reliable per-bird IDs are required
for trustworthy 30-minute inactivity classifications.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from behavior import estimate_body_length, prune_history, update_inactivity


@dataclass
class InactivitySample:
    body_len: float | None
    inactive: list
    tracked_histories: int
    audit_rows: list


class InactivityModule:
    def __init__(self, window=1800.0, interval=2.0, move_frac=0.5, min_fill=0.5):
        self.window = window
        self.interval = interval
        self.move_frac = move_frac
        self.min_samples = max(1, int(min_fill * window / interval))
        self.history = defaultdict(deque)
        self.body_lens = deque(maxlen=500)

    def observe(self, t, ids, boxes, audit=False):
        # Matches the source runner: update each track, prune, then estimate
        # body length from the running median (up to 500 sample estimates).
        for tid, (cx, cy, _w, _h) in zip(ids, boxes):
            self.history[tid].append((t, cx, cy))
        prune_history(self.history, t, self.window, self.interval)
        if boxes:
            self.body_lens.append(estimate_body_length([(b[2], b[3]) for b in boxes]))
        body_len = sorted(self.body_lens)[len(self.body_lens) // 2] if self.body_lens else None
        inactive = []
        if body_len:
            inactive = update_inactivity(
                self.history, t, self.window, self.move_frac * body_len, self.min_samples)

        # Optional diagnostic CSV. This does not influence classifications.
        audit_rows = []
        if audit:
            for tid, (cx, cy, _w, _h) in zip(ids, boxes):
                pts = self.history[tid]
                recent = [p for p in pts if t - p[0] <= self.window]
                span = t - pts[0][0]
                seen = len(recent)
                dx = max(p[1] for p in recent) - min(p[1] for p in recent)
                dy = max(p[2] for p in recent) - min(p[2] for p in recent)
                limit = self.move_frac * body_len if body_len else None
                span_ok = span >= self.window
                samples_ok = seen >= self.min_samples
                motion_ok = bool(limit and dx < limit and dy < limit)
                audit_rows.append([
                    tid, f'{cx:.2f}', f'{cy:.2f}', f'{span:.3f}', seen,
                    self.min_samples, int(span_ok), int(samples_ok),
                    f'{dx:.2f}', f'{dy:.2f}',
                    f'{limit:.2f}' if limit is not None else '',
                    int(motion_ok), int(tid in inactive),
                ])

        return InactivitySample(body_len, inactive, len(self.history), audit_rows)
