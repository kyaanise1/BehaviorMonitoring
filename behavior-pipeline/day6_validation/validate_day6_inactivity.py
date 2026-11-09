#!/usr/bin/env python3
"""Validate Hybrid V2 flock-level inactivity alerts against CVAT Day 6 labels.

Input files:
  1. day6_cvat_ground_truth_boxes.csv (one visible CVAT track per sampled frame)
  2. behavior_day6test2.csv (Hybrid V2 flock-level log, including t_s/n_inactive)

This is a SHORT-SEGMENT, FLOCK-LEVEL comparison, NOT a per-bird evaluation
and NOT a 30-minute inactivity rule validation. CVAT track IDs are unrelated
to ByteTrack IDs. Do not count 26 x 133 individual samples as independent TNs.

Only uses Python's standard library (Python 3.9+). No sklearn needed.

Example (PowerShell; files all in same folder):
    python validate_day6_inactivity.py `
        --ground-truth day6_cvat_ground_truth_boxes.csv `
        --predictions behavior_day6test2.csv `
        --output-dir day6_validation_results

If your CVAT trim started at a different time in the full recording, set
--time-offset-seconds accordingly; do NOT assume clips are aligned.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        return fields, list(reader)


def save_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_binary(value, description):
    if str(value).strip() not in ('0', '1'):
        raise ValueError(f'{description} must be 0 or 1, not {value!r}')
    return int(value)


def finite_num(value, description):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'Invalid {description}: {value!r}') from None
    if not math.isfinite(number):
        raise ValueError(f'{description} must be finite: {value!r}')
    return number


def read_gt_boxes(path: Path):
    fields, raw = load_csv(path)
    required = {'cvat_frame_index', 'estimated_video_time_seconds',
                'cvat_track_id', 'inactive_gt_manual'}
    missing = required - set(fields)
    if missing:
        raise ValueError('Ground truth must be the DETAILED boxes CSV, not the '
                         f'summary CSV. Missing columns: {sorted(missing)}')
    if not raw:
        raise ValueError('Ground-truth CSV is empty.')

    frames = defaultdict(list)
    birds = set()
    labeled_track = {}
    for row_n, row in enumerate(raw, start=2):
        f = int(row['cvat_frame_index'])
        bird = row['cvat_track_id'].strip()
        if not bird:
            raise ValueError(f'Empty CVAT track ID at CSV row {row_n}')
        y = parse_binary(row['inactive_gt_manual'], f'GT label at row {row_n}')
        t = finite_num(row['estimated_video_time_seconds'],
                       f'GT estimated timestamp at row {row_n}')
        if bird in labeled_track and labeled_track[bird] != y:
            raise ValueError(f'Conflicting ground-truth labels for CVAT chicken {bird}.')
        labeled_track[bird] = y
        birds.add(bird)
        frames[f].append((bird, t, y, row))

    output = []
    for f in sorted(frames):
        observations = frames[f]
        times = [t for _, t, _, _ in observations]
        if max(times) - min(times) > 0.001:
            raise ValueError(f'Inconsistent estimated timestamp at CVAT frame {f}')
        ids = [bird for bird, *_ in observations]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate track IDs at CVAT frame {f}')
        labels = [y for _, _, y, _ in observations]
        n_eligible = sum(
            str(row.get('eligible_30min_for_this_track', '')).strip() == '1'
            for _, _, _, row in observations)
        output.append({
            'cvat_frame_index': f,
            'gt_time_s': times[0],
            'gt_birds': len(ids),
            'gt_inactive_count': sum(labels),
            'gt_binary': int(any(labels)),
            'eligible_track_boxes_30min': n_eligible,
        })
    # Check time ordering independently of frame indices.
    for before, after in zip(output, output[1:]):
        if after['gt_time_s'] <= before['gt_time_s']:
            raise ValueError('CVAT estimated timestamps must increase with frame index.')
    return output, birds, labeled_track


def read_predictions(path: Path):
    fields, raw = load_csv(path)
    required = {'t_s', 'n_inactive'}
    missing = required - set(fields)
    if missing:
        raise ValueError(f'System prediction CSV is missing {sorted(missing)}')
    data = []
    for i, row in enumerate(raw, start=2):
        t = finite_num(row['t_s'], f'prediction timestamp CSV row {i}')
        raw_n = row['n_inactive'].strip()
        try:
            inactive_count = int(raw_n)
        except (ValueError, TypeError):
            raise ValueError(f'Invalid n_inactive at row {i}: {raw_n!r}') from None
        if inactive_count < 0:
            raise ValueError(f'Negative n_inactive at row {i}')
        data.append({
            't_s': t,
            'n_inactive': inactive_count,
            'binary': int(inactive_count > 0),
            'n_birds': row.get('n_birds', ''),
            'inactive_ids': row.get('inactive_ids', ''),
            'source_row': i,
        })
    if not data:
        raise ValueError('System predictions CSV is empty.')
    data.sort(key=lambda x: x['t_s'])
    return data


def confusion_metrics(cm):
    tp, fp, fn, tn = (cm[k] for k in ('TP', 'FP', 'FN', 'TN'))
    total = tp + fp + fn + tn
    actual_pos = tp + fn
    predicted_pos = tp + fp
    actual_neg = tn + fp
    accuracy = (tp + tn) / total if total else None
    precision = tp / predicted_pos if predicted_pos else None
    recall = tp / actual_pos if actual_pos else None
    specificity = tn / actual_neg if actual_neg else None
    # Positive-class F1 is uninformative without actual positives.
    # With actual positives but no correct detections, F1 = 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if actual_pos else None
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'balanced_accuracy': ((recall + specificity) / 2
                              if recall is not None and specificity is not None else None),
    }


def pretty(value):
    return f'{value:.4f} ({value * 100:.2f}%)' if value is not None else 'N/A'


def main():
    parser = argparse.ArgumentParser(
        description='Compare CVAT Day 6 manual no-inactivity ground truth against Hybrid V2 flock-level predictions.')
    parser.add_argument('--ground-truth', '--gt', type=Path,
                        default=Path('day6_cvat_ground_truth_boxes.csv'),
                        help='Detailed CVAT ground truth boxes CSV (not the summary).')
    parser.add_argument('--predictions', '--pred', type=Path,
                        default=Path('behavior_day6test2.csv'),
                        help='Hybrid V2 output CSV with t_s, n_inactive.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('day6_validation_results'))
    parser.add_argument('--max-time-difference', type=float, default=1.5,
                        help='Maximum absolute timestamp error (seconds) for matched frames; default 1.5.')
    parser.add_argument('--time-offset-seconds', type=float, default=0.0,
                        help='Offset added to CVAT times to align with the full video.')
    parser.add_argument('--required-inactivity-seconds', type=float, default=1800.0,
                        help='Required 30-minute inactivity window in seconds.')
    args = parser.parse_args()
    if args.max_time_difference < 0 or args.required_inactivity_seconds <= 0:
        parser.error('Invalid time/tolerance argument.')
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    gt, chicken_ids, bird_labels = read_gt_boxes(args.ground_truth)
    pred = read_predictions(args.predictions)
    pred_times = [p['t_s'] for p in pred]
    gt_start = gt[0]['gt_time_s'] + args.time_offset_seconds
    gt_end = gt[-1]['gt_time_s'] + args.time_offset_seconds
    span = gt_end - gt_start
    full_window_eligible = span >= args.required_inactivity_seconds

    # Audit EVERY Hybrid V2 prediction inside the manually labeled interval,
    # not only those chosen by nearest-neighbor frame matching. This catches
    # alerts that might happen between two CVAT sampled frames.
    overlap = []
    for p in pred:
        if gt_start <= p['t_s'] <= gt_end:
            overlap.append({
                'system_time_s': p['t_s'],
                'system_detected_birds': p['n_birds'],
                'system_n_inactive': p['n_inactive'],
                'system_inactive_ids': p['inactive_ids'],
                'system_any_inactive': p['binary'],
                'system_source_csv_row': p['source_row'],
            })
    save_csv(outdir / 'system_predictions_entire_gt_interval.csv',
             ['system_time_s', 'system_detected_birds', 'system_n_inactive',
              'system_inactive_ids', 'system_any_inactive', 'system_source_csv_row'], overlap)

    paired = []
    cm = Counter({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
    matched_system_row_counts = Counter()
    matched = 0
    for fr in gt:
        t = fr['gt_time_s'] + args.time_offset_seconds
        insert_at = bisect.bisect_left(pred_times, t)
        candidates = []
        if insert_at < len(pred):
            candidates.append(pred[insert_at])
        if insert_at > 0:
            candidates.append(pred[insert_at - 1])
        closest = min(candidates, key=lambda p: (abs(p['t_s'] - t), p['t_s']))
        dt = abs(closest['t_s'] - t)
        use_match = dt <= args.max_time_difference
        row = {
            'cvat_frame_index': fr['cvat_frame_index'],
            'gt_time_s': round(t, 6),
            'gt_visible_birds': fr['gt_birds'],
            'gt_inactive_birds': fr['gt_inactive_count'],
            'ground_truth_inactive': fr['gt_binary'],
            'system_time_s': '',
            'absolute_time_difference_s': '',
            'system_detected_birds': '',
            'system_inactive_count': '',
            'system_inactive_ids': '',
            'predicted_inactive': '',
            'confusion_outcome': 'UNMATCHED',
            'eligible_30min': int(full_window_eligible),
        }
        if use_match:
            actual = fr['gt_binary']
            predicted = closest['binary']
            category = ('TP' if actual and predicted else
                        'FN' if actual and not predicted else
                        'FP' if not actual and predicted else 'TN')
            cm[category] += 1
            matched += 1
            matched_system_row_counts[closest['source_row']] += 1
            row.update({
                'system_time_s': closest['t_s'],
                'absolute_time_difference_s': round(dt, 6),
                'system_detected_birds': closest['n_birds'],
                'system_inactive_count': closest['n_inactive'],
                'system_inactive_ids': closest['inactive_ids'],
                'predicted_inactive': predicted,
                'confusion_outcome': category,
            })
        paired.append(row)

    matched_fields = [
        'cvat_frame_index', 'gt_time_s', 'gt_visible_birds', 'gt_inactive_birds',
        'ground_truth_inactive', 'system_time_s', 'absolute_time_difference_s',
        'system_detected_birds', 'system_inactive_count', 'system_inactive_ids',
        'predicted_inactive', 'confusion_outcome', 'eligible_30min',
    ]
    save_csv(outdir / 'frame_level_prediction_vs_ground_truth.csv', matched_fields, paired)

    save_csv(outdir / 'confusion_matrix.csv',
             ['Ground truth / Prediction', 'Predicted inactive (1)', 'Predicted not inactive (0)'],
             [
                 {'Ground truth / Prediction': 'Actual inactive (1)',
                  'Predicted inactive (1)': cm['TP'], 'Predicted not inactive (0)': cm['FN']},
                 {'Ground truth / Prediction': 'Actual not inactive (0)',
                  'Predicted inactive (1)': cm['FP'], 'Predicted not inactive (0)': cm['TN']},
             ])

    metrics = confusion_metrics(cm)
    notes = {
        'accuracy': 'Short-duration, sampled-frame flock-level agreement; NOT validated 30-minute performance.',
        'precision': 'N/A when system predicted no positive inactivity alerts.',
        'recall': 'N/A without any ground-truth inactive events.',
        'f1_score': 'N/A without any ground-truth inactive events.',
        'specificity': 'Based on manually labeled negative CVAT frames only.',
        'balanced_accuracy': 'N/A unless both ground-truth classes are present.',
    }
    save_csv(outdir / 'metrics_summary.csv', ['metric', 'value', 'percent', 'interpretation'], [
        {'metric': name,
         'value': '' if value is None else round(value, 6),
         'percent': '' if value is None else round(100 * value, 3),
         'interpretation': notes[name]}
        for name, value in metrics.items()
    ])

    alerts_in_overlap = sum(row['system_any_inactive'] for row in overlap)
    duplicate_pred_matches = sum(n - 1 for n in matched_system_row_counts.values() if n > 1)
    warnings = []
    if not full_window_eligible:
        warnings.append('CVAT annotations span less than 1800 seconds: cannot validate the actual 30-minute inactivity rule.')
    if all(v == 0 for v in bird_labels.values()):
        warnings.append('Ground truth is NEGATIVE ONLY: precision, recall and F1 cannot establish positive-class detection performance.')
    if matched < len(gt):
        warnings.append(f'{len(gt) - matched} CVAT frames had no prediction within the time tolerance and were not scored.')
    if duplicate_pred_matches:
        warnings.append(f'{duplicate_pred_matches} prediction rows were reused by nearest-neighbor matches. Repeated matches may not be independent.')
    if alerts_in_overlap > 0:
        warnings.append(f'{alerts_in_overlap} system alerts appear within the annotated interval; inspect all predictions audit even if nearest-frame scores miss some.')
    warnings += [
        'CVAT timestamps are ESTIMATES from the original ground-truth conversion. Verify FPS and clip alignment before publication.',
        'The system log contains flock-level counts, not coordinates of predicted chicken IDs. No per-chicken ID match or individual accuracy is claimed.',
        'Repeated sampled frames are correlated; the count of TN frames is not the number of independent chickens or independent 30-minute trials.',
        'Interpolated CVAT boxes require visual review for accurate tracking/identity.'
    ]

    report = {
        'scope': 'short-duration, flock-level, sampled-frame binary no-inactivity comparison',
        'ground_truth_file': str(args.ground_truth),
        'prediction_file': str(args.predictions),
        'ground_truth_tracks': len(chicken_ids),
        'ground_truth_labeled_inactive_tracks': sum(bird_labels.values()),
        'ground_truth_labeled_not_inactive_tracks': sum(1 - x for x in bird_labels.values()),
        'cvat_frame_count': len(gt),
        'matched_frame_count': matched,
        'unmatched_frame_count': len(gt) - matched,
        'matched_prediction_row_reuses': duplicate_pred_matches,
        'gt_first_timestamp_s_aligned': round(gt_start, 6),
        'gt_last_timestamp_s_aligned': round(gt_end, 6),
        'gt_annotated_span_s': round(span, 6),
        'time_offset_s': args.time_offset_seconds,
        'max_time_difference_s': args.max_time_difference,
        'required_inactivity_s': args.required_inactivity_seconds,
        'eligible_for_full_30min_rule_validation': full_window_eligible,
        'system_predictions_full_file': len(pred),
        'system_predictions_in_gt_interval': len(overlap),
        'system_alert_rows_in_gt_interval': alerts_in_overlap,
        'system_alert_rows_full_file': sum(p['binary'] for p in pred),
        'confusion_matrix': dict(cm),
        'metrics': metrics,
        'warnings': warnings,
    }
    with (outdir / 'validation_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    report_txt = [
        'DAY 6: CVAT GROUND TRUTH vs HYBRID V2 INACTIVITY',
        '=' * 55,
        'Scope: Short-duration FLOCK-LEVEL sampled-frame comparison',
        f'CVAT physical chicken tracks: {len(chicken_ids)}',
        f'CVAT frames: {len(gt)}; matched frames: {matched}',
        f'Annotated interval: {gt_start:.3f} to {gt_end:.3f} s (span {span:.3f} s)',
        f'System rows in annotated interval: {len(overlap)}',
        f'System inactivity alert rows in annotated interval: {alerts_in_overlap}',
        f'30-minute rule evaluable: {"YES" if full_window_eligible else "NO"}',
        '',
        f'CONFUSION MATRIX (matched CVAT frames, n={matched})',
        '                            Predicted 1  Predicted 0',
        f'Actual inactive (1)             {cm["TP"]:>5}        {cm["FN"]:>5}',
        f'Actual not inactive (0)         {cm["FP"]:>5}        {cm["TN"]:>5}',
        '',
        'METRICS (sampled-frame, flock-level):',
        *[f'  {k:20s} {pretty(v)}' for k, v in metrics.items()],
        '',
        'LIMITATIONS AND WARNINGS:',
        *[f'  - {s}' for s in warnings],
        '',
        'These metrics are NOT a validated 30-minute inactivity detection accuracy.',
    ]
    text_report = '\n'.join(report_txt) + '\n'
    (outdir / 'validation_report.txt').write_text(text_report, encoding='utf-8')
    print(text_report)
    print(f'Files saved to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
