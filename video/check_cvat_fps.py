#!/usr/bin/env python3
r"""Find the frame rate that turns CVAT frame numbers into the right video times.

A CVAT video export stores SOURCE frame numbers (here 0, 30, 60, ... 23100 with step=30) but not the video's frame
rate, so time = frame / fps needs the fps from somewhere else. This script measures it instead of assuming it: for
each candidate fps it places every hand-drawn CVAT box at frame / fps seconds and checks whether the system's
tracked birds (an *_observations.csv from the pipeline, whose t_s comes from the video itself) are inside the box
at that moment. The right fps is the one where the boxes sit on the birds.

The CVAT frame numbers must NOT be rescaled (e.g. frame * 15 / 11.01): they already count frames of the source
video, so rescaling them and dividing by 15 gives exactly the same times as dividing the originals by 11.01.

Standard library only. Optional --video reads the frame rate from the file's header too (needs OpenCV).

    python check_cvat_fps.py --xml finald12_annotations.xml ^
        --observations ..\behavior-pipeline\results\Day12_15fps_broilernet450\Day12_15fps_observations.csv ^
        --video ..\Videosamanok\Day_12_2026-09-08_0600_to_0700.mkv
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

CANDIDATES = (10.0, 11.0, 11.01, 12.0, 12.5, 13.0, 14.0, 14.5, 15.0, 15.5, 16.0, 18.0, 20.0, 24.0, 25.0, 30.0)


def load_boxes(xml: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """[(source frame, x1, y1, x2, y2)] of every visible box, and the XML's stop_frame."""
    root = ET.parse(xml).getroot()
    stop = root.findtext('.//stop_frame')
    boxes = []
    for track in root.iter('track'):
        for b in track.iter('box'):
            if b.get('outside') == '1':
                continue
            boxes.append((int(b.get('frame')), *(float(b.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr'))))
    return boxes, int(stop) if stop else max(f for f, *_ in boxes)


def load_centres(path: Path) -> tuple[list[float], dict[float, list[tuple[float, float]]]]:
    """Sorted sample times and {time: [(cx, cy) of every tracked bird]} from a pipeline observations CSV."""
    by_t: dict[float, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            by_t[round(float(row['t_s']), 3)].append((float(row['cx']), float(row['cy'])))
    return sorted(by_t), by_t


def score(fps: float, boxes, times, by_t, max_dt: float) -> tuple[float, int]:
    """Share of CVAT boxes (that fall inside the system's time range) with a tracked bird's centre inside them."""
    hit = n = 0
    for frame, x1, y1, x2, y2 in boxes:
        t = frame / fps
        i = bisect_left(times, t)
        near = [times[j] for j in (i - 1, i) if 0 <= j < len(times)]
        if not near:
            continue
        ts = min(near, key=lambda v: abs(v - t))
        if abs(ts - t) > max_dt:
            continue
        n += 1
        hit += any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in by_t[ts])
    return (hit / n if n else 0.0), n


def main() -> None:
    ap = argparse.ArgumentParser(description='Measure which fps lines CVAT boxes up with the tracked birds')
    ap.add_argument('--xml', type=Path, required=True, help='CVAT video XML with the ORIGINAL frame numbers')
    ap.add_argument('--observations', type=Path, required=True, help='pipeline *_observations.csv (t_s, cx, cy)')
    ap.add_argument('--video', type=Path, help='optional: the source video, to read its header frame rate')
    ap.add_argument('--max-dt', type=float, default=1.0, help='largest gap between a CVAT time and a system sample')
    args = ap.parse_args()

    boxes, stop = load_boxes(args.xml)
    times, by_t = load_centres(args.observations)
    print(f'{len(boxes)} CVAT boxes up to source frame {stop}; system samples {times[0]:.0f}-{times[-1]:.0f} s')
    if args.video:
        try:
            import cv2
            cap = cv2.VideoCapture(str(args.video))
            vfps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            print(f'video header: {vfps:g} fps, {int(frames)} frames = {frames / vfps / 60:.1f} min' if vfps > 0
                  else 'video header: no frame rate')
        except ImportError:
            print('video header: OpenCV not installed, skipped')
    print('\n   fps   boxes on a bird   boxes checked   annotation ends at')
    results = []
    for fps in CANDIDATES:
        share, n = score(fps, boxes, times, by_t, args.max_dt)
        results.append((share, fps))
        print(f'{fps:6.2f}   {100 * share:14.1f} %   {n:13d}   {stop / fps / 60:8.1f} min')
    best_share, best = max(results)
    runner_up = max(s for s, f in results if f != best)
    print(f'\nBest: {best:g} fps ({100 * best_share:.1f} % of boxes on a bird; next best {100 * runner_up:.1f} %).')
    print(f'Use time = CVAT frame / {best:g}; the annotation covers {stop / best / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
