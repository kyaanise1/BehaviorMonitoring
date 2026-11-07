"""Re-extract exact Hybrid V2 pipeline frames from an existing timestamp CSV.

Works with Day 5, Day 6, MP4 and MKV. Run from behavior-pipeline.
The timestamps must have been produced using the same Hybrid V2 --interval.

Example:
 python extract_day6_validation_frames.py --video "D:\\path\\Day_05_2026-08-31_1000_to_1100.mkv" --csv day5_validation_timestamps.csv --output day5_validation_frames
"""
import argparse
import csv
from pathlib import Path

import cv2

from common import sample_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Source video')
    parser.add_argument('--csv', required=True, help='CSV with t_s timestamps')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Same interval used by Hybrid V2 (default: 2s)')
    parser.add_argument('--output', default='validation_frames_exact',
                        help='Output image directory')
    args = parser.parse_args()

    video = Path(args.video)
    csv_file = Path(args.csv)
    if not video.is_file():
        parser.error(f'Video not found: {video}')
    if not csv_file.is_file():
        parser.error(f'Timestamp CSV not found: {csv_file}')
    if args.interval <= 0:
        parser.error('--interval must be positive')

    with csv_file.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 't_s' not in reader.fieldnames:
            parser.error('Timestamp CSV must contain a t_s column')
        records = list(reader)
    if not records:
        parser.error('Timestamp CSV has no data rows')

    targets = {}
    for i, record in enumerate(records, 1):
        try:
            t = float(record['t_s'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid t_s in CSV row {i}') from exc
        if t < 0:
            raise ValueError(f'Negative t_s in CSV row {i}')
        key = round(t, 3)
        if key in targets:
            raise ValueError(f'Duplicate timestamp {key} in CSV')
        targets[key] = (i, record)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    found = {}
    width = max(2, len(str(len(records))))
    for t, sample_no, frame in sample_frames(str(video), args.interval):
        key = round(t, 3)
        if key in targets:
            i, record = targets[key]
            # Preserve filenames from extract_frames2.py for matching LabelImg labels.
            filename = record.get('image_filename', '').strip()
            if not filename:
                filename = f'frame_{i:0{width}d}_t_{key:010.3f}s.jpg'
            if Path(filename).name != filename or not filename.lower().endswith('.jpg'):
                raise ValueError(f'Invalid image filename in CSV row {i}: {filename!r}')
            if frame is None or not cv2.imwrite(str(output / filename), frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f'Failed to write {filename}')
            found[key] = (i, key, t, sample_no, filename)
            print(f'[{len(found):02d}/{len(targets)}] {filename} (pipeline sample {sample_no})')
            if len(found) == len(targets):
                break

    with (output / 'extraction_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_row', 'requested_t_s', 'pipeline_t_s',
                         'pipeline_sample_no', 'filename'])
        for key, (i, _) in sorted(targets.items(), key=lambda x: x[1][0]):
            if key in found:
                row_no, requested, actual, sample_no, filename = found[key]
                writer.writerow([row_no, f'{requested:.3f}', f'{actual:.6f}',
                                 sample_no, filename])

    missing = [f'{key:.3f}' for key in targets if key not in found]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} timestamps: {missing}. '
                           'Check source video and --interval.')
    print(f'\nSaved {len(found)} exact pipeline frames to {output.resolve()}')


if __name__ == '__main__':
    main()
