"""Detect suspicious stretches in chronologically sampled video.

A repeated sampled image is only labelled a freeze after >= freeze_frames
near-identical samples. The initial image is retained; subsequent duplicates
are GAP. A sudden jump in timestamps is another GAP. Explicit intervals can
be excluded as GAP (useful when damage is known in advance).

This is a conservative heuristic, NOT proof of intact or missing recording.
"""
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class CheckedFrame:
    time: float
    number: Optional[int]
    frame: Optional[np.ndarray]
    status: str
    reason: str = ""


def parse_time(value):
    parts = str(value).split(":")
    if len(parts) not in (1, 2, 3):
        raise ValueError(f"Invalid time: {value}; use seconds or HH:MM:SS")
    try:
        sec = 0.0
        for part in parts:
            sec = 60 * sec + float(part)
    except ValueError as exc:
        raise ValueError(f"Invalid time: {value}") from exc
    if sec < 0:
        raise ValueError("Times must not be negative")
    return sec


def parse_exclusions(items):
    ranges = []
    for item in items:
        if "-" not in item:
            raise ValueError(f"Invalid exclusion {item!r}; use HH:MM:SS-HH:MM:SS")
        start, end = item.split("-", 1)
        a, b = parse_time(start), parse_time(end)
        if a >= b:
            raise ValueError(f"Exclusion start must precede end: {item}")
        ranges.append((a,b))
    return ranges


def signature(frame):
    """Coarse view, excluding the top/bottom edges where clock overlays appear."""
    if frame is None or not isinstance(frame, np.ndarray) or not frame.size:
        return None
    h = frame.shape[0]
    crop = frame[h // 10 : max(h // 10 + 1, 9 * h // 10), :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)


def iter_guarded_samples(samples, interval=2.0, freeze_frames=3,
                         freeze_diff=0.25, max_step=6.0, exclusions=()):
    """Convert (t,n,image) to CheckedFrame.

    Results remain chronological. Only the first visually identical frame is
    accepted when a sufficiently long repeat is detected. We buffer at most
    freeze_frames images to avoid reporting early frozen samples as VALID.
    """
    if freeze_frames < 2 or interval <= 0 or freeze_diff < 0 or max_step <= interval:
        raise ValueError("Check freeze_frames, interval, freeze_diff and max_step")
    pending = []
    sig = None
    frozen = False
    prev_t = None

    def flush():
        nonlocal pending, sig, frozen
        if not frozen:
            for t0, n0, fr0 in pending:
                yield CheckedFrame(t0, n0, fr0, "VALID")
        pending = []
        sig = None
        frozen = False

    for t, n, frame in samples:
        t = float(t)
        if prev_t is not None and t <= prev_t:
            raise ValueError(f"Non-increasing video timestamp: {t} after {prev_t}")
        if prev_t is not None and t - prev_t > max_step:
            yield from flush()
            yield CheckedFrame(prev_t + interval, None, None, "GAP", "timestamp_jump")
        prev_t = t

        if any(a <= t < b for a,b in exclusions):
            yield from flush()
            yield CheckedFrame(t, n, None, "GAP", "excluded_interval")
            continue

        s = signature(frame)
        if s is None:
            yield from flush()
            yield CheckedFrame(t, n, None, "GAP", "unreadable_frame")
            continue

        if sig is None:
            pending = [(t,n,frame)]
            sig = s
            continue

        difference = float(np.mean(cv2.absdiff(sig, s)))
        if difference <= freeze_diff:
            if frozen:
                yield CheckedFrame(t, n, None, "GAP", "frozen_frame")
            else:
                pending.append((t,n,frame))
                if len(pending) >= freeze_frames:
                    # First image may be genuine; subsequent repeated images
                    # do not constitute new observed bird movement.
                    first = pending[0]
                    yield CheckedFrame(first[0], first[1], first[2], "VALID")
                    for t0,n0,_ in pending[1:]:
                        yield CheckedFrame(t0,n0,None,"GAP","frozen_frame")
                    pending = []
                    frozen = True
        else:
            yield from flush()
            pending = [(t,n,frame)]
            sig = s
    yield from flush()
