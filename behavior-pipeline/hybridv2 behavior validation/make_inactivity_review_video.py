"""Create a lightweight, sampled review video for a tracked chicken's inactivity audit.

Fixed edition: the information header is above the source frame, never over it.

Uses existing audit CSV only; does NOT re-run YOLO or Hybrid V2.
This video is for checking tracker identity and model movement measurements,
NOT independent ground-truth annotation. Review original continuous video as well.

Run from behavior-pipeline (after setting PYTHONPATH to that folder):
python ".\\hybridv2 behavior validation\\make_inactivity_review_video.py" \
    --source "../video/Day6_repaired.mp4" \
    --observations "inactivity_track_observations.csv" \
    --track-id 2 --until-seconds 1805 --out "day6_track2_review.mp4"
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
from common import sample_frames


def format_time(t):
    n = int(t)
    return f"{n//3600:02d}:{(n//60)%60:02d}:{n%60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Create visual track-ID review clip from existing audit data")
    ap.add_argument('--source', required=True, help='Same video used by audit_inactivity_v2.py')
    ap.add_argument('--observations', default='inactivity_track_observations.csv')
    ap.add_argument('--track-id', type=int, default=2)
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--until-seconds', type=float, default=1805)
    ap.add_argument('--playback-fps', type=float, default=5.0, help='5 fps with 2 s samples ≈ 10x speed')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--out', default='day6_track_review.mp4')
    args = ap.parse_args()
    if args.playback_fps <= 0 or args.interval <= 0 or args.width <= 0:
        ap.error('playback-fps, interval, and width must be positive')
    if not Path(args.source).is_file() or not Path(args.observations).is_file():
        ap.error('source video or observations CSV not found')
    rows = {}
    with open(args.observations, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['track_id']) == args.track_id and float(r['t_s']) < args.until_seconds:
                rows[round(float(r['t_s']), 3)] = r
    if not rows:
        ap.error(f'No observations found for track ID {args.track_id} before {args.until_seconds}s')

    writer = None
    count = found = 0
    trail = deque(maxlen=60)
    try:
        for t, sample_number, frame in sample_frames(args.source, args.interval):
            if t >= args.until_seconds:
                break
            if frame is None:
                continue
            h0, w0 = frame.shape[:2]
            scale = min(1.0, args.width / w0)
            w, h = int(round(w0*scale)), int(round(h0*scale))
            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))
            header_h = 124
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(args.out), fourcc, args.playback_fps, (w, h + header_h))
                if not writer.isOpened():
                    raise RuntimeError(f'Unable to open output writer: {args.out}')

            key = round(t, 3)
            r = rows.get(key)
            # Matching key comes from the same sample_frames source and uses 3-decimal t_s.
            if r:
                found += 1
                cx, cy = int(round(float(r['cx'])*scale)), int(round(float(r['cy'])*scale))
                trail.append((t, cx, cy))
                label = f'ID {args.track_id}  at ({r["cx"]}, {r["cy"]})'
                status = f'30-min window: {"READY" if r["span_ok"] == "1" and r["samples_ok"] == "1" else "NOT READY"}'
                motion = f'X range {r["range_x_px"]}px  Y range {r["range_y_px"]}px  limit {r["movement_limit_px"]}px'
                cv2.circle(frame, (cx, cy), 19, (255, 0, 255), 3)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                cv2.putText(frame, f'ID {args.track_id}', (cx+21, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                label = f'ID {args.track_id} NOT DETECTED in this sampled frame'
                status, motion = '30-min window: UNKNOWN', ''
            while trail and t-trail[0][0] > 120:
                trail.popleft()
            pts = [(x,y) for _,x,y in trail]
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j-1], pts[j], (255, 0, 255), 2, cv2.LINE_AA)

            # Keep the full source image visible: add an external header ABOVE the frame.
            # All trail/centroid annotations were already drawn in the original frame coordinates.
            canvas = cv2.copyMakeBorder(frame, header_h, 0, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for y, txt in [(27, f'Video time {format_time(t)}  |  sample #{sample_number}  |  ID {args.track_id}'),
                           (55, label), (82, status), (109, motion)]:
                if txt:
                    cv2.putText(canvas, txt, (14,y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.63, (255,255,255), 2, cv2.LINE_AA)
            writer.write(canvas)
            count += 1
            if count % 300 == 0:
                print(f'Wrote {count} sampled frames through {format_time(t)}')
    finally:
        if writer:
            writer.release()

    if not count:
        raise RuntimeError('No video samples decoded')
    print(f'Saved: {args.out} | {count} sampled frames, ID seen in {found} frames')
    print('IMPORTANT: Compressed sampled review. Use original full video for final manual labels;')
    print('this overlay is only for checking ID continuity and visualizing model tracks.')


if __name__ == '__main__':
    main()
