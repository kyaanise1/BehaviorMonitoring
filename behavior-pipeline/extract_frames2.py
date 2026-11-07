"""Create a reproducible, evenly distributed random validation sample.

Uses the SAME video-frame sampling as the Hybrid V2 runner (common.sample_frames).
Selects one random pipeline frame from each of --count time-ordered strata.
It does not run the detector or use the detector's predictions.

Example (run from behavior-pipeline):
 python extract_frames2.py --video "D:\\path\\Day_05_2026-08-31_1000_to_1100.mkv" --count 60
"""
import argparse
import csv
import random
from pathlib import Path

import cv2

from common import sample_frames


def hms(seconds):
    ms = round(seconds * 1000)
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def count_available_samples(video, interval_s):
    """Use video metadata, falling back to counting if MKV lacks frame count."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError("Could not read FPS from video")
    step = max(1, round(fps * interval_s))
    if frames > 0:
        return (frames + step - 1) // step
    print("Video frame count unavailable; counting pipeline samples (extra pass).")
    return sum(1 for _ in sample_frames(str(video), interval_s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Source video (Day 5 MKV)')
    parser.add_argument('--count', type=int, default=60,
                        help='Number of randomly sampled frames (default: 60)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Match Hybrid V2 --interval (default: 2 seconds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Fixed random seed so selection is reproducible')
    parser.add_argument('--output', default='day5_validation_frames',
                        help='Directory for selected images')
    parser.add_argument('--csv', default='day5_validation_timestamps.csv',
                        help='Timestamp CSV for Hybrid V2 --timestamps-csv')
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_file():
        parser.error(f'Video not found: {video}')
    if args.count <= 0 or args.interval <= 0:
        parser.error('--count and --interval must be positive')

    sample_count = count_available_samples(video, args.interval)
    if sample_count < args.count:
        parser.error(f'Only {sample_count} pipeline samples; cannot select {args.count} unique frames')

    rng = random.Random(args.seed)
    # One randomly chosen sample in each stratum: covers the full video hour.
    selected_samples = []
    for k in range(args.count):
        start = k * sample_count // args.count
        end = (k + 1) * sample_count // args.count
        selected_samples.append(rng.randrange(start, end) + 1)  # 1-based
    chosen = set(selected_samples)
    last_sample = max(chosen)

    output = Path(args.output)
    csv_path = Path(args.csv)
    output.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(args.count)))
    rows = []
    for t, sample_no, frame in sample_frames(str(video), args.interval):
        if sample_no in chosen:
            index = len(rows) + 1
            # Timestamp format is required by evaluate_huddling_clusters.py.
            name = f'frame_{index:0{width}d}_t_{t:010.3f}s.jpg'
            path = output / name
            if frame is None or not cv2.imwrite(str(path), frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f'Failed to save frame {sample_no}: {path}')
            rows.append({
                'frame_no': index,
                't_s': f'{t:.3f}',
                'timestamp': hms(t),
                'pipeline_sample_no': sample_no,
                'image_filename': name,
            })
            print(f'[{index:02d}/{args.count}] {name} (pipeline sample {sample_no})')
        if sample_no >= last_sample:
            break

    if len(rows) != args.count:
        raise RuntimeError(
            f'Expected {args.count} samples but extracted {len(rows)}. '
            'Check video readability and reported frame count.'
        )
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved {len(rows)} exact pipeline frames to {output.resolve()}')
    print(f'Saved timestamps CSV: {csv_path.resolve()}')
    print(f'Sampling: every {args.interval:g}s, stratified random, seed {args.seed}')
    print('Do not use model predictions to revise these validation annotations.')


if __name__ == '__main__':
    main()
