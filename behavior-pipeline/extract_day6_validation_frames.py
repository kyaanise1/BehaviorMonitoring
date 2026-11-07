"""Extract precisely the frames consumed by the Hybrid V2 pipeline.
Run from behavior-pipeline directory, using same --interval as the pipeline.
"""
import argparse
import csv
from pathlib import Path
import cv2
from common import sample_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--csv', required=True, help='30-frame manual validation CSV with t_s')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--output', default='day6_validation_frames_exact')
    args = ap.parse_args()
    with open(args.csv, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(not r.get('t_s') for r in rows):
        ap.error('CSV must have nonempty t_s values')
    targets = [(i, float(r['t_s'])) for i, r in enumerate(rows, 1)]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    matches = {}
    # Same sequential frame grabbing, stepping, and timestamp calculation as both pipelines.
    for t, sample_no, frame in sample_frames(args.video, args.interval):
        for row_no, requested in targets:
            if row_no not in matches and abs(t - requested) <= 0.001:
                filename = f'frame_{row_no:02d}_t_{requested:010.3f}s.jpg'
                if not cv2.imwrite(str(out / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Failed to write {filename}')
                matches[row_no] = (requested, t, sample_no, filename)
                print(f'[{len(matches):02d}/{len(targets)}] {filename} (pipeline t={t:.6f}s, sample={sample_no})')
        if len(matches) == len(targets):
            break
    with open(out / 'extraction_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['annotation_row','requested_t_s','pipeline_t_s','pipeline_sample_no','filename'])
        for row_no, _ in targets:
            if row_no in matches:
                requested, t, sample_no, filename = matches[row_no]
                w.writerow([row_no, f'{requested:.3f}', f'{t:.6f}', sample_no, filename])
    missing = [i for i, _ in targets if i not in matches]
    if missing:
        raise RuntimeError(f'No exact pipeline frame found for annotation rows: {missing}. Check video and --interval.')
    print(f'Saved {len(matches)} exact pipeline frames to {out}')

if __name__ == '__main__':
    main()
