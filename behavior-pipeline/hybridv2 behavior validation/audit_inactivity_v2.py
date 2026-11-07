"""Audit why Hybrid V2 produces (or fails to produce) inactivity events.

Use as a separate diagnostic, not as a replacement for the behavior runner.
Place alongside gap_guard.py inside 'hybridv2 behavior validation'. Run from
behavior-pipeline after setting PYTHONPATH to that directory.

Outputs:
  inactivity_track_observations.csv  -- one row per observed ID/sample
  inactivity_track_summary.csv       -- why each session+ID did/didn't qualify

Identical default tracking and inactivity settings to run_pipeline_hybrid_v2_gapaware.py.
"""
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

from ultralytics import YOLO
from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior import estimate_body_length, prune_history
from gap_guard import iter_guarded_samples, parse_exclusions


def new_hybrid():
    return RecoveryHybrid(tight=0.5, wide=4.0, max_missed=4, wide_conf=0.3,
                          size_ratio=2.0, recovery_conf=0.7,
                          recovery_distance=1.5, duplicate_iou=0.5)


def new_stat(session, tid):
    return dict(session=session, track_id=tid, first_s=None, last_s=None,
                observations=0, current_streak_start=None,
                longest_consecutive_s=0.0, max_history_span_s=0.0,
                max_samples_in_window=0, span_ready_frames=0,
                sample_ready_frames=0, inactivity_frames=0,
                best_motion_ratio=None, best_motion_at_s=None,
                best_dx_px=None, best_dy_px=None, best_limit_px=None)


def update_stat(stat, t, interval, history_span, samples_in_window,
                span_ok, samples_ok, inactive, dx=None, dy=None, limit=None):
    if stat['first_s'] is None:
        stat['first_s'] = t
        stat['current_streak_start'] = t
    if stat['last_s'] is not None and t - stat['last_s'] > 1.5 * interval:
        stat['current_streak_start'] = t
    stat['last_s'] = t
    stat['observations'] += 1
    stat['longest_consecutive_s'] = max(
        stat['longest_consecutive_s'], t - stat['current_streak_start'])
    stat['max_history_span_s'] = max(stat['max_history_span_s'], history_span)
    stat['max_samples_in_window'] = max(
        stat['max_samples_in_window'], samples_in_window)
    stat['span_ready_frames'] += bool(span_ok)
    stat['sample_ready_frames'] += bool(span_ok and samples_ok)
    stat['inactivity_frames'] += bool(inactive)
    if span_ok and samples_ok and limit and limit > 0:
        ratio = max(dx, dy) / limit
        if stat['best_motion_ratio'] is None or ratio < stat['best_motion_ratio']:
            stat['best_motion_ratio'] = ratio
            stat['best_motion_at_s'] = t
            stat['best_dx_px'] = dx
            stat['best_dy_px'] = dy
            stat['best_limit_px'] = limit


def main():
    ap = argparse.ArgumentParser(description='Per-track Hybrid V2 inactivity audit')
    ap.add_argument('--source', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tracker', default='configs/bytetrack_broiler.yaml')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--window', type=float, default=1800.0)
    ap.add_argument('--move-frac', type=float, default=0.5)
    ap.add_argument('--min-fill', type=float, default=0.5)
    ap.add_argument('--freeze-frames', type=int, default=3)
    ap.add_argument('--freeze-diff', type=float, default=0.25)
    ap.add_argument('--max-time-step', type=float, default=6.0)
    ap.add_argument('--warmup-frames', type=int, default=3)
    ap.add_argument('--min-visible', type=int, default=1)
    ap.add_argument('--exclude', action='append', default=[])
    ap.add_argument('--until-seconds', type=float, default=None,
                    help='Stop before this video timestamp; e.g. 2462 for Day 6 pre-gap')
    ap.add_argument('--observations-csv', default='inactivity_track_observations.csv')
    ap.add_argument('--summary-csv', default='inactivity_track_summary.csv')
    args = ap.parse_args()
    if not Path(args.source).exists():
        ap.error(f'Cannot find video: {args.source}')
    if args.interval <= 0 or args.window <= 0 or not 0 < args.min_fill <= 1:
        ap.error('interval and window must be positive, min-fill must be in (0,1]')
    exclusions = parse_exclusions(args.exclude)
    if Path(args.source).is_dir():
        print('CAUTION: Image-folder times are synthetic; gaps between images are unknown.')

    tracker_model = YOLO(args.weights, task='detect')
    detector_model = YOLO(args.weights, task='detect')
    hybrid = new_hybrid()
    history = defaultdict(deque)
    body_lengths = deque(maxlen=500)
    required_samples = max(1, int(args.min_fill * args.window / args.interval))
    need_reset = False
    warmup_left = 0
    session = 0
    stats = {}
    nframes = 0
    ninvalid = 0
    ninactive = 0

    with open(args.observations_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t_s','session','status','track_id','cx','cy',
                    'history_span_s','samples_in_window','required_samples',
                    'span_ok','samples_ok','range_x_px','range_y_px',
                    'movement_limit_px','movement_ok','inactive'])
        src = sample_frames(args.source, args.interval)
        checked = iter_guarded_samples(src, interval=args.interval,
                                       freeze_frames=args.freeze_frames,
                                       freeze_diff=args.freeze_diff,
                                       max_step=args.max_time_step,
                                       exclusions=exclusions)
        for item in checked:
            t = float(item.time)
            if args.until_seconds is not None and t >= args.until_seconds:
                break
            nframes += 1
            if item.status == 'GAP':
                need_reset = True
                ninvalid += 1
                continue
            if need_reset:
                # Same reset strategy as run_pipeline_hybrid_v2_gapaware.py
                del tracker_model
                tracker_model = YOLO(args.weights, task='detect')
                hybrid = new_hybrid()
                history.clear()
                body_lengths.clear()
                warmup_left = args.warmup_frames
                need_reset = False
                session += 1

            bt = tracker_model.track(item.frame, tracker=args.tracker,
                                     conf=args.conf, imgsz=args.imgsz,
                                     persist=True, verbose=False)[0]
            if bt.boxes is None or bt.boxes.id is None:
                byte_ids, byte_boxes, byte_scores = [], [], []
            else:
                byte_ids = bt.boxes.id.int().tolist()
                byte_boxes = bt.boxes.xywh.tolist()
                byte_scores = bt.boxes.conf.tolist()
            raw = detector_model.predict(item.frame, conf=args.conf,
                                         imgsz=args.imgsz, verbose=False)[0]
            if raw.boxes is None:
                raw_boxes, raw_scores = [], []
            else:
                raw_boxes = raw.boxes.xywh.tolist()
                raw_scores = raw.boxes.conf.tolist()
            ids, boxes, _, _ = hybrid.update(
                byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores)
            if len(ids) < args.min_visible:
                need_reset = True
                ninvalid += 1
                continue
            for tid, (cx,cy,bw,bh) in zip(ids,boxes):
                history[tid].append((t,cx,cy))
            prune_history(history, t, args.window, args.interval)
            if boxes:
                body_lengths.append(estimate_body_length([(b[2],b[3]) for b in boxes]))
            body_len = (sorted(body_lengths)[len(body_lengths)//2]
                        if body_lengths else None)
            recovering = warmup_left > 0
            if recovering:
                warmup_left -= 1
            status = 'RECOVERING' if recovering else 'VALID'
            for tid, (cx,cy,_,_) in zip(ids,boxes):
                pts = history[tid]
                span = t - pts[0][0]
                recent = [p for p in pts if t-p[0] <= args.window]
                count = len(recent)
                span_ok = span >= args.window
                samples_ok = count >= required_samples
                dx = max(p[1] for p in recent)-min(p[1] for p in recent)
                dy = max(p[2] for p in recent)-min(p[2] for p in recent)
                limit = (args.move_frac * body_len if body_len else None)
                motion_ok = bool(limit and dx < limit and dy < limit)
                inactive = bool(not recovering and span_ok and samples_ok and motion_ok)
                ninactive += inactive
                key = (session, int(tid))
                if key not in stats:
                    stats[key] = new_stat(*key)
                update_stat(stats[key], t, args.interval, span, count,
                            span_ok, samples_ok, inactive, dx, dy, limit)
                w.writerow([f'{t:.3f}', session, status, tid,
                            f'{cx:.2f}',f'{cy:.2f}',f'{span:.3f}', count,
                            required_samples, int(span_ok), int(samples_ok),
                            f'{dx:.2f}',f'{dy:.2f}',
                            f'{limit:.2f}' if limit is not None else '',
                            int(motion_ok), int(inactive)])
            if item.number is not None and item.number % 200 == 0:
                print(f'frame={item.number} time={t:.1f}s IDs={len(ids)} '
                      f'candidate_30m_frames={sum(x["sample_ready_frames"] for x in stats.values())}')

    fields = ['session','track_id','first_s','last_s','observations',
              'longest_consecutive_s','max_history_span_s','max_samples_in_window',
              'span_ready_frames','sample_ready_frames','inactivity_frames',
              'best_motion_ratio','best_motion_at_s','best_dx_px','best_dy_px',
              'best_limit_px','main_reason']
    summaries = []
    for st in stats.values():
        s = dict(st)
        if s['inactivity_frames']:
            s['main_reason'] = 'inactivity_triggered'
        elif not s['span_ready_frames']:
            s['main_reason'] = 'never_spanned_full_30min_window'
        elif not s['sample_ready_frames']:
            s['main_reason'] = 'not_enough_observations_in_window'
        else:
            s['main_reason'] = 'centroid_movement_exceeded_threshold'
        for col in ('first_s','last_s','longest_consecutive_s','max_history_span_s',
                    'best_motion_ratio','best_motion_at_s','best_dx_px',
                    'best_dy_px','best_limit_px'):
            if s[col] is not None:
                s[col] = round(s[col], 3)
        summaries.append(s)
    summaries.sort(key=lambda x: (-x['sample_ready_frames'],
                                  -x['max_history_span_s'], x['session'],x['track_id']))
    with open(args.summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(summaries)
    print(f'Finished: {nframes} sampled frames ({ninvalid} GAP), '
          f'{len(stats)} session-specific track IDs')
    print(f'Observed inactivity-positive track/frame pairs: {ninactive}')
    print(f'Tracks with enough history and samples: '
          f'{sum(bool(s["sample_ready_frames"]) for s in summaries)}')
    print(f'Saved {args.observations_csv} and {args.summary_csv}')
    print('NOTE: This audits the rule, not biological inactivity or correct physical identity.')

if __name__ == '__main__':
    main()
