#!/usr/bin/env python3
"""Map CVAT for video track frames from a VFR source onto a CFR video.

Uses actual presented frame timestamps (PTS), not stream average frame rates.
Requires ffprobe on PATH. Designed for the SAME footage, with the SAME starting
point and original resolution, transcoded with ffmpeg's fps filter.

Example (PowerShell):
python remap_cvat_xml_by_pts.py `
  --source "Day12_validation_35min_CVAT.mp4" `
  --target "Day12_validation_35min_CVAT_15fps.mp4" `
  --xml "finald12_annotations.xml" `
  --out "finald12_annotations_15fps_pts.xml"

WARNING: CVAT frame-step setting of the new task must be 1 for arbitrary
converted frame numbers. Review mapped annotations in CVAT before using as GT.
This is timestamp-proximity mapping, not pixel-perfect evidence that a source
frame survived the fps filter. Some source frames can be dropped by ffmpeg.
"""
import argparse
import bisect
import copy
import json
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def probe(path, *, frames=False):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if frames:
        command += ['-show_frames', '-show_entries', 'frame=best_effort_timestamp_time,pts_time', '-of', 'json']
    else:
        command += ['-show_streams', '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,r_frame_rate', '-of', 'json']
    command += [str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise SystemExit('ERROR: ffprobe not found on PATH. Install FFmpeg.')
    except subprocess.CalledProcessError as e:
        raise SystemExit('ffprobe failed for '+str(path)+': '+e.stderr[:1500])
    data = json.loads(result.stdout)
    if not frames:
        streams = data.get('streams', [])
        if len(streams) != 1:
            raise SystemExit(f'ERROR: Expected one selected video stream in {path}, got {len(streams)}')
        return streams[0]
    results = []
    for index, entry in enumerate(data.get('frames', [])):
        timestamp = entry.get('best_effort_timestamp_time', entry.get('pts_time'))
        if timestamp is None or timestamp == 'N/A':
            raise SystemExit(f'ERROR: Missing timestamp for video frame {index} in {path}')
        results.append(float(timestamp))
    if not results:
        raise SystemExit(f'ERROR: No timestamps found for {path}')
    if any(b <= a for a, b in zip(results, results[1:])):
        raise SystemExit(f'ERROR: Frame timestamps are not strictly increasing in {path}. Please investigate.')
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='Original video used in CVAT')
    ap.add_argument('--target', type=Path, required=True, help='New CFR video made with ffmpeg fps=15')
    ap.add_argument('--xml', type=Path, required=True, help='Original CVAT for video XML (never edits this)')
    ap.add_argument('--out', type=Path, required=True, help='New XML path')
    args = ap.parse_args()
    for p in (args.source, args.target, args.xml):
        if not p.is_file():
            ap.error(f'File missing: {p}')
    if args.out.resolve() == args.xml.resolve():
        ap.error('--out must be a different file from --xml')

    a = probe(args.source)
    b = probe(args.target)
    if (int(a['width']), int(a['height'])) != (int(b['width']), int(b['height'])):
        ap.error('Video resolution changed: box coordinates require spatial rescaling as well')
    print('Source:', args.source.name, a.get('avg_frame_rate'), 'avg FPS,', a.get('nb_frames'), 'reported frames')
    print('Target:', args.target.name, b.get('avg_frame_rate'), 'avg FPS,', b.get('nb_frames'), 'reported frames')

    tree = ET.parse(args.xml)
    root = tree.getroot()
    pairs = [(track, shape) for track in root.findall('.//track') for shape in track
             if shape.get('frame') is not None]
    if not pairs:
        ap.error('No track shapes with frame attribute found. Check XML format.')
    indices = [int(shape.get('frame')) for _, shape in pairs]
    if min(indices) < 0:
        ap.error('Negative frame number in XML')
    print('Reading actual original frame timestamps with ffprobe...')
    old_pts = probe(args.source, frames=True)
    if max(indices) >= len(old_pts):
        ap.error(f'CVAT XML refers to original frame {max(indices)}, but source has only '
                 f'{len(old_pts)} decoded frames. This XML may not use source-frame indices.')
    print('Reading actual output frame timestamps with ffprobe...')
    new_pts = probe(args.target, frames=True)
    if abs((old_pts[-1] - old_pts[0]) - (new_pts[-1] - new_pts[0])) > 1:
        ap.error('Source/output timestamp ranges differ by over 1 second: not the same time span')
    # ffmpeg fps filter usually normalizes initial PTS; compare elapsed video time.
    old_t0, new_t0 = old_pts[0], new_pts[0]
    errors, collisions = [], []
    count = 0
    for track in root.findall('.//track'):
        used = {}
        for shape in track:
            old_index = shape.get('frame')
            if old_index is None:
                continue
            oi = int(old_index)
            t = old_pts[oi] - old_t0
            absolute_t = t + new_t0
            insert_at = bisect.bisect_left(new_pts, absolute_t)
            options = [i for i in (insert_at-1, insert_at) if 0 <= i < len(new_pts)]
            ni = min(options, key=lambda i: (abs(new_pts[i] - absolute_t), i))
            error = abs(new_pts[ni] - absolute_t)
            if ni in used:
                collisions.append((track.get('id'), used[ni], oi, ni))
            used[ni] = oi
            errors.append(error)
            shape.set('frame', str(ni))
            count += 1
    print('Original timestamp span: %.3f s; output span %.3f s' %
          (old_pts[-1]-old_t0, new_pts[-1]-new_t0))
    print('Remapped %d shape elements across %d tracks' % (count, len(root.findall('.//track'))))
    print('Nearest output-frame timestamp discrepancy: median %.4f s, max %.4f s' %
          (statistics.median(errors), max(errors)))
    if collisions:
        print('ERROR: Same-track frame collisions detected (first 8):')
        for entry in collisions[:8]:
            print('  track=%s old_frames=%s and %s --> new_frame=%s' % entry)
        raise SystemExit('No output written; please investigate collisions before import.')
    if max(errors) > 0.1:
        raise SystemExit('No output written: >100 ms discrepancy; check source/target timebase alignment.')
    # Metadata in existing exports (if any) may refer to old task. CVAT import normally
    # uses the destination task settings. Leave task metadata unchanged; don't invent it.
    # Frame numbers are the only field updated here; all track IDs and labels preserved.
    ET.indent(tree, space='  ')
    tree.write(args.out, encoding='utf-8', xml_declaration=True)
    print('Saved:', args.out)
    print('Spot-check several annotations against both videos. Do not assume that all')
    print('original frames survived FFmpeg fps filtering; interpolated geometry needs review.')


if __name__ == '__main__':
    main()
