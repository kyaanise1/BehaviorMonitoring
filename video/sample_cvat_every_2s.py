#!/usr/bin/env python3
"""Sample CVAT video-track XML at fixed wall-clock intervals (15 fps / 2 s by default).

Retains CVAT track IDs and linearly interpolates rectangular bounding boxes
between neighboring observations. Never interpolates across an 'outside' gap,
and never extrapolates before the first or after the last box.

Example:
  python sample_cvat_every_2s.py --input finald12_annotations_15fps_pts.xml --output day12_every_2s.csv

When annotations are edited later, re-export the CVAT for video XML and rerun.
"""
import argparse
import bisect
import csv
import xml.etree.ElementTree as ET

COORDS = ('xtl', 'ytl', 'xbr', 'ybr')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Exported CVAT for video XML from the 15-fps task')
    p.add_argument('--output', required=True, help='Path for sampled CSV')
    p.add_argument('--fps', type=int, default=15, help='Output video FPS (default 15)')
    p.add_argument('--interval', type=int, default=2, help='Sample interval in seconds (default 2)')
    args = p.parse_args()
    if args.fps <= 0 or args.interval <= 0:
        p.error('--fps and --interval must be positive')

    stride = args.fps * args.interval
    root = ET.parse(args.input).getroot()
    tracks = root.findall('./track')
    if not tracks:
        p.error('No CVAT video tracks were found in the XML')

    all_shapes = [s for t in tracks for s in t if s.tag == 'box' and s.get('frame') is not None]
    if not all_shapes:
        p.error('No tracked boxes found')
    max_frame = max(int(s.get('frame')) for s in all_shapes)
    sampled_frames = range(0, max_frame + 1, stride)
    fields = ['frame', 'time_s', 'track_id', 'label', 'xtl', 'ytl', 'xbr', 'ybr',
              'cx', 'cy', 'occluded', 'source']
    rows = 0
    active_tracks = 0
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            items = sorted(
                ((int(s.get('frame')), s) for s in track if s.tag == 'box' and s.get('frame') is not None),
                key=lambda x: x[0],
            )
            if not items:
                continue
            frames = [x[0] for x in items]
            if len(frames) != len(set(frames)):
                raise ValueError(f'Duplicate frame numbers in track {track.get("id")}; inspect XML')
            active_tracks += 1
            for frame in sampled_frames:
                j = bisect.bisect_right(frames, frame) - 1
                if j < 0:
                    continue
                left_frame, left = items[j]
                if frame == left_frame:
                    if left.get('outside', '0') == '1':
                        continue
                    xy = [float(left.get(k)) for k in COORDS]
                    source = 'keyframe' if left.get('keyframe') == '1' else 'cvat_interpolated'
                    occluded = int(left.get('occluded', '0'))
                else:
                    if j + 1 >= len(items) or left.get('outside', '0') == '1':
                        continue
                    right_frame, right = items[j + 1]
                    alpha = (frame - left_frame) / (right_frame - left_frame)
                    xy = [(1 - alpha) * float(left.get(k)) + alpha * float(right.get(k)) for k in COORDS]
                    source = 'resampled_interpolated'
                    occluded = int(left.get('occluded', '0'))
                xtl, ytl, xbr, ybr = xy
                writer.writerow({
                    'frame': frame, 'time_s': f'{frame / args.fps:.3f}',
                    'track_id': track.get('id'), 'label': track.get('label'),
                    'xtl': f'{xtl:.2f}', 'ytl': f'{ytl:.2f}',
                    'xbr': f'{xbr:.2f}', 'ybr': f'{ybr:.2f}',
                    'cx': f'{(xtl + xbr) / 2:.2f}',
                    'cy': f'{(ytl + ybr) / 2:.2f}',
                    'occluded': occluded, 'source': source,
                })
                rows += 1
    print(f'Saved {rows:,} visible-track rows from {active_tracks} tracks')
    print(f'Sampled frames 0, {stride}, {2 * stride}, ... {max(sampled_frames)}')
    print(f'Output: {args.output}')
    print('WARNING: Interpolated positions are not independent manual ground-truth observations.')


if __name__ == '__main__':
    main()
