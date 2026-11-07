r"""Rebuild the Day 6 inactivity diagnostic summary without rerunning video inference.

Reads inactivity_track_observations.csv from audit_inactivity_v2.py and writes
inactivity_track_summary.csv. Requires only the Python standard library.

Usage from behavior-pipeline:
    python ".\hybridv2 behavior validation\finish_inactivity_summary.py"
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


FIELDS = [
    'session', 'track_id', 'first_s', 'last_s', 'observations',
    'longest_consecutive_s', 'max_history_span_s', 'max_samples_in_window',
    'span_ready_frames', 'sample_ready_frames', 'inactivity_frames',
    'best_motion_ratio', 'best_motion_at_s', 'best_dx_px', 'best_dy_px',
    'best_limit_px', 'main_reason',
]


def truth(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def maybe_float(value):
    if value is None or value == '':
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild the per-ID inactivity summary from an already-saved observations CSV'
    )
    parser.add_argument('--observations-csv', default='inactivity_track_observations.csv')
    parser.add_argument('--summary-csv', default='inactivity_track_summary.csv')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Expected sampling interval in seconds (default: 2)')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    src = Path(args.observations_csv)
    if not src.is_file():
        parser.error(f'Cannot find {src}. Check where the audit script saved its observations file.')

    stats = {}
    rows_read = 0
    with src.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {
            't_s', 'session', 'track_id', 'history_span_s', 'samples_in_window',
            'span_ok', 'samples_ok', 'inactive', 'range_x_px', 'range_y_px',
            'movement_limit_px',
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error(f'Observations CSV is missing columns: {", ".join(sorted(missing))}')

        for row in reader:
            rows_read += 1
            key = (int(row['session']), int(row['track_id']))
            t = float(row['t_s'])
            span = float(row['history_span_s'])
            count = int(row['samples_in_window'])
            span_ok, samples_ok = truth(row['span_ok']), truth(row['samples_ok'])
            inactive = truth(row['inactive'])
            dx, dy, limit = (
                maybe_float(row['range_x_px']),
                maybe_float(row['range_y_px']),
                maybe_float(row['movement_limit_px']),
            )

            if key not in stats:
                stats[key] = dict(
                    session=key[0], track_id=key[1], first_s=t, last_s=t,
                    observations=0, current_streak_start=t,
                    longest_consecutive_s=0.0, max_history_span_s=0.0,
                    max_samples_in_window=0, span_ready_frames=0,
                    sample_ready_frames=0, inactivity_frames=0,
                    best_motion_ratio=None, best_motion_at_s=None,
                    best_dx_px=None, best_dy_px=None, best_limit_px=None,
                )
            st = stats[key]
            if t - st['last_s'] > 1.5 * args.interval:
                st['current_streak_start'] = t
            st['last_s'] = t
            st['observations'] += 1
            st['longest_consecutive_s'] = max(
                st['longest_consecutive_s'], t - st['current_streak_start'])
            st['max_history_span_s'] = max(st['max_history_span_s'], span)
            st['max_samples_in_window'] = max(st['max_samples_in_window'], count)
            st['span_ready_frames'] += int(span_ok)
            st['sample_ready_frames'] += int(span_ok and samples_ok)
            st['inactivity_frames'] += int(inactive)
            if span_ok and samples_ok and limit is not None and limit > 0 and dx is not None and dy is not None:
                ratio = max(dx, dy) / limit
                if st['best_motion_ratio'] is None or ratio < st['best_motion_ratio']:
                    st['best_motion_ratio'] = ratio
                    st['best_motion_at_s'] = t
                    st['best_dx_px'] = dx
                    st['best_dy_px'] = dy
                    st['best_limit_px'] = limit

    if rows_read == 0:
        parser.error(f'{src} has headers but no observation rows; cannot build summary.')

    summaries = []
    for st in stats.values():
        if st['inactivity_frames']:
            reason = 'inactivity_triggered'
        elif not st['span_ready_frames']:
            reason = 'never_spanned_full_30min_window'
        elif not st['sample_ready_frames']:
            reason = 'not_enough_observations_in_window'
        else:
            reason = 'centroid_movement_exceeded_threshold'
        result = {field: st.get(field) for field in FIELDS}
        result['main_reason'] = reason
        for col in (
            'first_s', 'last_s', 'longest_consecutive_s', 'max_history_span_s',
            'best_motion_ratio', 'best_motion_at_s', 'best_dx_px', 'best_dy_px',
            'best_limit_px',
        ):
            if result[col] is not None:
                result[col] = round(result[col], 3)
        summaries.append(result)

    summaries.sort(key=lambda x: (
        -x['sample_ready_frames'], -x['max_history_span_s'],
        x['session'], x['track_id']))
    with open(args.summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    reasons = Counter(st['main_reason'] for st in summaries)
    print(f'Read {rows_read} track/frame observations for {len(summaries)} session-specific IDs.')
    print(f'Tracks with enough history and samples: '
          f'{sum(bool(st["sample_ready_frames"]) for st in summaries)}')
    print('Main reasons:')
    for name, count in reasons.most_common():
        print(f'  {name}: {count}')
    print(f'Saved {args.summary_csv}')
    print('Note: This is an algorithm audit, not independent validation of biological inactivity.')


if __name__ == '__main__':
    main()
