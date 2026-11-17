#!/usr/bin/env python3
"""Append one column to the Day 12 CVAT ground truth CSV.

The computed column is the *algorithm-derived* rolling cumulative activity from
CVAT bounding-box centroids, using the same median3 definition as the user's
Option A code (three-point centered median; two-point median at each window
edge; available points only). It is NOT a manually labeled behavioral GT.

Usage:
    python add_day12_gt_activity_level.py \
      --input day12_ground_truth_2s.csv \
      --output day12_ground_truth_2s_with_activity.csv

No dependencies beyond NumPy.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def rolling_activity(rows, window_seconds=1800.0):
    """Return per-original-row scores. Times/coordinates are numeric values."""
    by_id = defaultdict(list)
    for row_index, row in enumerate(rows):
        tid = int(row['track_id'])
        by_id[tid].append((float(row['time_s']), float(row['cx']), float(row['cy']), row_index))

    output = [None] * len(rows)
    for tid, values in by_id.items():
        values.sort(key=lambda v: (v[0], v[3]))
        times = np.asarray([v[0] for v in values], dtype=float)
        coords = np.asarray([[v[1], v[2]] for v in values], dtype=float)
        if len(set(times.tolist())) != len(times):
            raise ValueError(f'Duplicate timestamps within GT track {tid}')
        if not np.all(np.isfinite(coords)):
            raise ValueError(f'Nonfinite centroid found in GT track {tid}')
        left = 0
        for right, t in enumerate(times):
            while left < right and times[left] < t - window_seconds - 1e-8:
                left += 1
            # The existing algorithm calculates medians ONLY within the window.
            xy = coords[left:right + 1]
            n = len(xy)
            if n == 1:
                activity = 0.0
            else:
                smoothed = np.empty_like(xy)
                # statistics.median(two coordinates) = their average
                smoothed[0] = (xy[0] + xy[1]) / 2.0
                smoothed[-1] = (xy[-2] + xy[-1]) / 2.0
                if n > 2:
                    smoothed[1:-1] = np.median(
                        np.stack([xy[:-2], xy[1:-1], xy[2:]], axis=0), axis=0)
                diffs = np.diff(smoothed, axis=0)
                activity = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
            output[values[right][3]] = activity
    if any(value is None for value in output):
        raise AssertionError('Not every GT row received an Activity Level')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('day12_ground_truth_2s.csv'))
    p.add_argument('--output', type=Path, default=Path('day12_ground_truth_2s_with_activity.csv'))
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve():
        p.error('Output path must differ from original input')
    with args.input.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers or not {'time_s', 'track_id', 'cx', 'cy'}.issubset(headers):
            p.error('Missing time_s, track_id, cx, or cy in ground-truth file')
        if 'cumulative_activity_level_px' in headers:
            p.error('Cumulative activity column already exists in the input')
        rows = list(reader)
    scores = rolling_activity(rows, window_seconds=1800.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[*headers, 'cumulative_activity_level_px'],
                                lineterminator='\n')
        writer.writeheader()
        for row, score in zip(rows, scores):
            row['cumulative_activity_level_px'] = f'{score:.3f}'
            writer.writerow(row)
    print(f'Created {args.output} with {len(rows)} rows and {len(headers)+1} columns')
    print('NOTE: Early entries are partial windows; this score alone does NOT assert inactivity.')
    print('NOTE: GT interpolated positions can understate short movements.')


if __name__ == '__main__':
    main()
