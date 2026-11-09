"""Per-ID inactivity audit for run_pipeline_hybrid_v2.py (NON-GAP-AWARE).

This is a diagnostic ONLY. It does not modify Hybrid V2 or its behavior rules.
The tracker, history, body-length calculation, and update_inactivity() call match
run_pipeline_hybrid_v2.py. It intentionally does NOT detect gaps or reset
tracking/history after frozen frames or camera recording discontinuities.

From behavior-pipeline, example:
    python ".\\hybridv2 behavior validation\\audit_inactivity_v2_nongap.py" \
        --source "../video/Day12_repaired.mp4" \
        --weights "../runs/broiler_yolo11n/weights/best.pt" \
        --tracker "configs/bytetrack_broiler.yaml" \
        --interval 2 --window 1800 \
        --observations-csv "day12_nongap_observations.csv" \
        --summary-csv "day12_nongap_summary.csv" \
        --focus-id 15 --compare-pred-csv "behavior_day12test2.csv"

To compare tracker IDs reliably, use the EXACT same video/images and settings
as those used for the comparison behavior CSV. This diagnostic by itself does
not provide independent ground-truth labels.
"""

import argparse
import csv
import sys
from collections import defaultdict, deque
from pathlib import Path

# Same import resolution strategy as run_pipeline_hybrid_v2.py.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ultralytics import YOLO
from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior import (estimate_body_length, prune_history, update_inactivity)


def new_hybrid():
    return RecoveryHybrid(tight=0.5, wide=4.0, max_missed=4,
                          wide_conf=0.3, size_ratio=2.0,
                          recovery_conf=0.7, recovery_distance=1.5,
                          duplicate_iou=0.5)


def new_stat(tid):
    return dict(session=0, track_id=tid, first_s=None, last_s=None,
                observations=0, current_streak_start=None,
                longest_consecutive_s=0.0, max_history_span_s=0.0,
                max_samples_in_window=0, span_ready_frames=0,
                sample_ready_frames=0, inactivity_frames=0,
                first_inactivity_s=None, last_inactivity_s=None,
                best_motion_ratio=None, best_motion_at_s=None,
                best_dx_px=None, best_dy_px=None, best_limit_px=None)


def update_stat(stat, t, interval, history_span, samples_in_window,
                span_ok, samples_ok, inactive, dx, dy, limit):
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
    stat['max_samples_in_window'] = max(stat['max_samples_in_window'], samples_in_window)
    stat['span_ready_frames'] += int(span_ok)
    stat['sample_ready_frames'] += int(span_ok and samples_ok)
    stat['inactivity_frames'] += int(inactive)
    if inactive:
        if stat['first_inactivity_s'] is None:
            stat['first_inactivity_s'] = t
        stat['last_inactivity_s'] = t
    if span_ok and samples_ok and limit is not None and limit > 0:
        ratio = max(dx, dy) / limit
        if stat['best_motion_ratio'] is None or ratio < stat['best_motion_ratio']:
            stat['best_motion_ratio'] = ratio
            stat['best_motion_at_s'] = t
            stat['best_dx_px'] = dx
            stat['best_dy_px'] = dy
            stat['best_limit_px'] = limit


def load_predictions(path):
    """Optional frame-level consistency check against the main runner's CSV."""
    data = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {'t_s', 'n_inactive', 'inactive_ids'}.issubset(reader.fieldnames):
            raise ValueError('Prediction CSV requires t_s,n_inactive,inactive_ids columns')
        for row in reader:
            ts = f"{float(row['t_s']):.3f}"
            ids = set(int(x) for x in row['inactive_ids'].split('|') if x.strip())
            data[ts] = (int(row['n_inactive']), ids)
    return data


def main():
    ap = argparse.ArgumentParser(description='NON-GAP-AWARE Hybrid V2 per-ID inactivity audit')
    ap.add_argument('--source', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tracker', default='configs/bytetrack_broiler.yaml')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--window', type=float, default=1800.0)
    ap.add_argument('--move-frac', type=float, default=0.5)
    ap.add_argument('--min-fill', type=float, default=0.5)
    ap.add_argument('--until-seconds', type=float, default=None,
                    help='Optional cutoff time in video seconds (exclusive)')
    ap.add_argument('--focus-id', type=int, default=None,
                    help='Print a focused result for one tracking ID; still audit all IDs')
    ap.add_argument('--compare-pred-csv', default='',
                    help='Optional CSV from run_pipeline_hybrid_v2.py to check identical inactivity predictions')
    ap.add_argument('--observations-csv', default='inactivity_nongap_observations.csv')
    ap.add_argument('--summary-csv', default='inactivity_nongap_summary.csv')
    args = ap.parse_args()

    if not Path(args.source).exists():
        ap.error(f'Cannot find video/images: {args.source}')
    if args.interval <= 0 or args.window <= 0 or not 0 < args.min_fill <= 1:
        ap.error('interval and window must be positive; min-fill must be in (0,1]')
    if args.until_seconds is not None and args.until_seconds <= 0:
        ap.error('--until-seconds must be positive')
    if Path(args.observations_csv).resolve() == Path(args.summary_csv).resolve():
        ap.error('Use different filenames for observations and summary')
    if args.compare_pred_csv and not Path(args.compare_pred_csv).exists():
        ap.error(f'Prediction CSV not found: {args.compare_pred_csv}')
    if args.compare_pred_csv and Path(args.compare_pred_csv).resolve() in {
        Path(args.observations_csv).resolve(), Path(args.summary_csv).resolve()
    }:
        ap.error('Do not overwrite the prediction CSV with an audit output')

    if Path(args.source).is_dir():
        print('CAUTION: Image-folder timestamps are synthetic; true recording gaps are not known.')
    print('NON-GAP-AWARE MODE: tracking/history is never reset for frozen or missing video.')
    predictions = load_predictions(args.compare_pred_csv) if args.compare_pred_csv else None

    # Match the behavior runner: distinct YOLO instances, identical Hybrid V2 params.
    tracker_model = YOLO(args.weights, task='detect')
    detector_model = YOLO(args.weights, task='detect')
    hybrid = new_hybrid()
    history = defaultdict(deque)
    body_lens = deque(maxlen=500)
    required_samples = max(1, int(args.min_fill * args.window / args.interval))

    stats = {}
    nframes = 0
    positive_pairs = 0
    comparison_checked = 0
    comparison_missing = 0
    mismatches = []

    with open(args.observations_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['t_s', 'session', 'status', 'track_id', 'cx', 'cy',
                         'history_span_s', 'samples_in_window', 'required_samples',
                         'span_ok', 'samples_ok', 'range_x_px', 'range_y_px',
                         'movement_limit_px', 'movement_ok', 'inactive'])

        for t, n, frame in sample_frames(args.source, args.interval):
            t = float(t)
            if args.until_seconds is not None and t >= args.until_seconds:
                break
            nframes += 1

            bt = tracker_model.track(frame, tracker=args.tracker,
                                     conf=args.conf, imgsz=args.imgsz,
                                     persist=True, verbose=False)[0]
            if bt.boxes is None or bt.boxes.id is None:
                byte_ids, byte_boxes, byte_scores = [], [], []
            else:
                byte_ids = bt.boxes.id.int().tolist()
                byte_boxes = bt.boxes.xywh.tolist()
                byte_scores = bt.boxes.conf.tolist()

            raw = detector_model.predict(frame, conf=args.conf,
                                         imgsz=args.imgsz, verbose=False)[0]
            if raw.boxes is None:
                raw_boxes, raw_scores = [], []
            else:
                raw_boxes = raw.boxes.xywh.tolist()
                raw_scores = raw.boxes.conf.tolist()

            ids, boxes, scores, recovered = hybrid.update(
                byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores)
            for tid, (cx, cy, bw, bh) in zip(ids, boxes):
                history[tid].append((t, cx, cy))
            prune_history(history, t, args.window, args.interval)
            if boxes:
                body_lens.append(estimate_body_length([(b[2], b[3]) for b in boxes]))
            body_len = sorted(body_lens)[len(body_lens) // 2] if body_lens else None

            # CRITICAL: call the EXACT same inactivity rule as the main runner.
            # No duplicated or altered implementation of the classification logic.
            inactive_ids = set(update_inactivity(
                history, t, args.window,
                args.move_frac * body_len, required_samples)) if body_len else set()
            positive_pairs += len(inactive_ids)

            if predictions is not None:
                key = f'{t:.3f}'
                if key not in predictions:
                    comparison_missing += 1
                else:
                    comparison_checked += 1
                    pred_count, pred_ids = predictions[key]
                    if len(inactive_ids) != pred_count or inactive_ids != pred_ids:
                        mismatches.append((key, sorted(pred_ids), sorted(inactive_ids)))

            limit = args.move_frac * body_len if body_len else None
            for tid, (cx, cy, bw, bh) in zip(ids, boxes):
                pts = history[tid]
                span = t - pts[0][0]
                recent = [p for p in pts if t - p[0] <= args.window]
                count = len(recent)
                span_ok = span >= args.window
                samples_ok = count >= required_samples
                dx = max(p[1] for p in recent) - min(p[1] for p in recent)
                dy = max(p[2] for p in recent) - min(p[2] for p in recent)
                motion_ok = bool(limit and dx < limit and dy < limit)
                inactive = tid in inactive_ids

                if tid not in stats:
                    stats[tid] = new_stat(int(tid))
                update_stat(stats[tid], t, args.interval, span, count,
                            span_ok, samples_ok, inactive, dx, dy, limit)
                writer.writerow([
                    f'{t:.3f}', 0, 'VALID', tid, f'{cx:.2f}', f'{cy:.2f}',
                    f'{span:.3f}', count, required_samples,
                    int(span_ok), int(samples_ok), f'{dx:.2f}', f'{dy:.2f}',
                    f'{limit:.2f}' if limit is not None else '',
                    int(motion_ok), int(inactive)
                ])

            if n % 200 == 0:
                print(f'frame={n} t={t:.1f}s IDs={len(ids)} '
                      f'current_inactive={len(inactive_ids)}')

    fields = ['session', 'track_id', 'first_s', 'last_s', 'observations',
              'longest_consecutive_s', 'max_history_span_s', 'max_samples_in_window',
              'span_ready_frames', 'sample_ready_frames', 'inactivity_frames',
              'first_inactivity_s', 'last_inactivity_s', 'best_motion_ratio',
              'best_motion_at_s', 'best_dx_px', 'best_dy_px', 'best_limit_px',
              'main_reason']
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
        for col in ('first_s', 'last_s', 'longest_consecutive_s',
                    'max_history_span_s', 'first_inactivity_s',
                    'last_inactivity_s', 'best_motion_ratio',
                    'best_motion_at_s', 'best_dx_px', 'best_dy_px', 'best_limit_px'):
            if s[col] is not None:
                s[col] = round(s[col], 3)
        summaries.append(s)
    summaries.sort(key=lambda x: (-x['inactivity_frames'],
                                  -x['sample_ready_frames'],
                                  -x['max_history_span_s'], x['track_id']))

    with open(args.summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)

    print(f'Finished: {nframes} sampled frames, {len(stats)} tracked IDs')
    print(f'Inactivity-positive ID/frame pairs: {positive_pairs}')
    print(f'IDs with sufficient 30-minute history and samples: '
          f'{sum(bool(s["sample_ready_frames"]) for s in summaries)}')
    if args.focus_id is not None:
        match = next((s for s in summaries if s['track_id'] == args.focus_id), None)
        if match is None:
            print(f'ID {args.focus_id}: NOT OBSERVED in this run')
        else:
            print(f'ID {args.focus_id}: first={match["first_s"]}s '
                  f'last={match["last_s"]}s '
                  f'positive_frames={match["inactivity_frames"]} '
                  f'first_positive={match["first_inactivity_s"]}s '
                  f'last_positive={match["last_inactivity_s"]}s '
                  f'reason={match["main_reason"]}')
    if predictions is not None:
        print(f'Comparison: matched timestamps={comparison_checked}, '
              f'missing in reference CSV={comparison_missing}, '
              f'inactivity prediction mismatches={len(mismatches)}')
        for t, expected, got in mismatches[:10]:
            print(f'  t={t}s expected={expected} audit={got}')
        if len(mismatches) > 10:
            print(f'  ... plus {len(mismatches) - 10} additional mismatches')
        if mismatches or comparison_missing:
            print('CHECK: verify same source and settings; tracking IDs may differ between runs.')
        else:
            print('Per-frame inactivity ID predictions match the reference CSV.')
    print(f'Saved {args.observations_csv} and {args.summary_csv}')
    print('NOTE: Audit is not ground truth; non-gap-aware history may span corrupted footage.')


if __name__ == '__main__':
    main()
