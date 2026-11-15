#!/usr/bin/env python3
"""Day 12: independent CVAT ground truth vs nongap Hybrid V2 inactivity alerts.

Python 3.10+, standard library only. Does NOT modify Hybrid V2 or its inputs.

Default evaluation:
  - Ground truth describes whether a *physical chicken* was inactive during
    the FIRST 30 minutes (t=0..1800 s).
  - System detections are checked just after the 30-minute threshold,
    t=1800..1810 s (change --alert-grace-s if justified).
  - CVAT ID 1 is an annotator-confirmed positive; it is position-matched to
    Hybrid V2 ID 15. All other CVAT IDs are initially PROVISIONAL, never
    silently promoted to confirmed ground truth.
  - Negative candidates need visible movement beyond the half-body-length
    threshold at independently set CVAT keyframes. Human identity review
    remains necessary.

Run from the folder containing the three Day 12 inputs:
    python validate_inactivity_day12.py

Or supply paths:
    python validate_inactivity_day12.py \
        --xml day12_annotations_validations.xml \
        --observations day12_nongap_observations.csv \
        --summary day12_nongap_summary.csv \
        --fps 15

The first run creates OUTPUT/day12_review_labels.csv. Edit 'review_status' to
'confirmed' only AFTER independently reviewing each bird/track. Re-run to
recompute confirmed metrics. The label file is NEVER overwritten.

Timing: CVAT XML contains original video frame indices (step=30), but not the
video FPS. The Day 12 recording (Day_12_2026-09-08_0600_to_0700.mkv) is 15 FPS
(1280x720, 53,949 frames), and the CVAT boxes line up with it only at 15 FPS:
mean best IoU against detections 0.733 at 15 FPS, 0.310 at the 11 FPS this
script used to infer from stop_frame 23100 / 2100 s. At 15 FPS the annotation
covers 23100 / 15 = 1540 s (25.7 min), so it cannot validate the 30-minute rule
by itself; the script says so instead of scoring. Pass --fps to override.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys
import xml.etree.ElementTree as ET


@dataclass
class Box:
    frame: int
    cx: float
    cy: float
    long_side: float
    keyframe: bool
    outside: bool
    occluded: bool


@dataclass
class Audit:
    cvat_id: int
    sampled_boxes: int
    valid_boxes: int
    manual_keyframes: int
    manual_dx: float | None
    manual_dy: float | None
    half_body_px: float | None
    outside_first30: int
    occluded_first30: int
    last_manual_s: float | None
    candidate: str
    explanation: str


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f'Missing input file: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = required - cols
        if missing:
            raise ValueError(f'{path.name}: missing columns: {sorted(missing)}')
        return list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits=2):
    return '' if v is None else f'{v:.{digits}f}'


def parse_cvat(path: Path) -> tuple[dict[int, list[Box]], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f'Missing XML: {path}')
    root = ET.parse(path).getroot()
    job = root.find('meta/job')
    if job is None:
        raise ValueError('Expected CVAT for video 1.1 XML, with <meta><job>.')
    meta = {k: job.findtext(k, default='') for k in ('size', 'start_frame', 'stop_frame', 'frame_filter')}
    tracks: dict[int, list[Box]] = {}
    for track in root.findall('track'):
        if track.get('label') != 'broiler':
            continue
        track_id = int(track.attrib['id'])
        boxes = []
        for b in track.findall('box'):
            x1, y1 = float(b.attrib['xtl']), float(b.attrib['ytl'])
            x2, y2 = float(b.attrib['xbr']), float(b.attrib['ybr'])
            boxes.append(Box(
                frame=int(b.attrib['frame']), cx=(x1+x2)/2, cy=(y1+y2)/2,
                long_side=max(x2-x1, y2-y1), keyframe=b.get('keyframe') == '1',
                outside=b.get('outside') == '1', occluded=b.get('occluded') == '1'
            ))
        tracks[track_id] = sorted(boxes, key=lambda b: b.frame)
    if not tracks:
        raise ValueError('No tracks labeled "broiler" were found.')
    return tracks, meta


def audit_one(track_id: int, boxes: list[Box], fps: float,
              sample_step: int, window_s: float) -> Audit:
    max_src_frame = window_s * fps
    first = [b for b in boxes if b.frame <= max_src_frame + 1e-6]
    visible = [b for b in first if not b.outside]
    kfs = [b for b in visible if b.keyframe and not b.occluded]
    outside = sum(b.outside for b in first)
    occluded = sum(b.occluded and not b.outside for b in first)
    valid = [b for b in visible if not b.occluded]
    limit = median(b.long_side for b in valid) / 2 if valid else None
    dx = (max(b.cx for b in kfs) - min(b.cx for b in kfs)) if len(kfs) >= 2 else None
    dy = (max(b.cy for b in kfs) - min(b.cy for b in kfs)) if len(kfs) >= 2 else None
    latest = max((b.frame / fps for b in kfs), default=None)
    expected = int(max_src_frame / sample_step) + 1
    enough = (len(visible) >= expected and first and not first[0].outside and
              len(visible) == len(first) and first[-1].frame >= max_src_frame - sample_step)
    if not enough:
        candidate = 'INSUFFICIENT_COVERAGE'
        note = f'Need {expected} visible sampled boxes covering 0–{window_s:g}s; got {len(visible)}.'
    elif dx is None or dy is None or limit is None:
        candidate = 'NEEDS_MANUAL_REVIEW'
        note = 'Fewer than two usable manually set keyframes; cannot infer a negative.'
    elif dx > limit or dy > limit:
        candidate = 'NEGATIVE_CANDIDATE'
        note = 'Manually placed keyframes show movement exceeding half median body length.'
    else:
        candidate = 'POSSIBLE_POSITIVE_REVIEW'
        note = 'Movement at keyframes alone does not establish 30 minutes of inactivity.'
    if outside or occluded:
        note += f' First-window obstruction flags: outside={outside}, occluded={occluded}.'
    return Audit(track_id, len(first), len(visible), len(kfs), dx, dy,
                 limit, outside, occluded, latest, candidate, note)


def create_labels(path: Path, audits: dict[int, Audit]):
    if path.exists():
        return False
    rows = []
    for id_, a in sorted(audits.items()):
        if id_ == 1:
            gt, status, system_id = '1', 'confirmed', '15'
            note = ('Annotator confirmed this physical bird inactive for 30 min. '
                    'System 15 is the positional match; verify identity match against video if needed.')
        elif a.candidate == 'NEGATIVE_CANDIDATE':
            gt, status, system_id = '0', 'provisional', ''
            note = 'Movement evidence only. Set review_status=confirmed after checking physical identity.'
        else:
            gt, status, system_id = '', 'exclude', ''
            note = a.explanation
        rows.append({'cvat_id': id_, 'ground_truth_inactive': gt,
                     'review_status': status, 'system_track_id': system_id,
                     'notes': note})
    write_csv(path, ['cvat_id', 'ground_truth_inactive', 'review_status',
                     'system_track_id', 'notes'], rows)
    return True


def load_labels(path: Path, audits: dict[int, Audit]):
    rows = read_csv(path, {'cvat_id', 'ground_truth_inactive', 'review_status', 'system_track_id'})
    result = {}
    mapped = set()
    for row in rows:
        id_ = int(row['cvat_id'])
        if id_ not in audits:
            raise ValueError(f'Labels CSV refers to CVAT ID {id_}, not present in XML.')
        if id_ in result:
            raise ValueError(f'Duplicate label for CVAT ID {id_}')
        gt = row['ground_truth_inactive'].strip()
        status = row['review_status'].strip().lower()
        raw_system = row['system_track_id'].strip()
        if status not in {'confirmed', 'provisional', 'exclude'}:
            raise ValueError(f'CVAT {id_}: review_status must be confirmed, provisional, or exclude.')
        if gt not in {'', '0', '1'}:
            raise ValueError(f'CVAT {id_}: ground_truth_inactive must be blank, 0 or 1.')
        if status != 'exclude' and gt == '':
            raise ValueError(f'CVAT {id_}: assign a 0/1 GT or set review_status=exclude.')
        if raw_system:
            sys_id = int(raw_system)
            if sys_id in mapped:
                raise ValueError(f'System ID {sys_id} mapped to multiple CVAT birds.')
            mapped.add(sys_id)
        result[id_] = {**row, 'gt': int(gt) if gt else None,
                       'status': status, 'sys_id': int(raw_system) if raw_system else None}
    for id_ in audits:
        if id_ not in result:
            raise ValueError(f'Labels CSV is missing CVAT ID {id_}.')
    return result


def load_system(observations: Path, summary: Path, end_s: float,
                min_s: float, max_s: float, matching_ids: set[int]):
    rows = read_csv(observations, {'t_s', 'track_id', 'status', 'cx', 'cy', 'inactive'})
    summaries = read_csv(summary, {'track_id', 'inactivity_frames', 'first_inactivity_s'})
    summary_alerts = {int(r['track_id']): r for r in summaries
                      if float(r.get('inactivity_frames') or 0) > 0}
    all_alerts: dict[int, list[float]] = defaultdict(list)
    scoring_alerts: dict[int, list[float]] = defaultdict(list)
    positions: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    n_in_window = 0
    for r in rows:
        if r['status'] != 'VALID':
            continue
        t = float(r['t_s'])
        sys_id = int(r['track_id'])
        if t <= end_s + 1e-6:
            n_in_window += 1
            if int(float(r['inactive'])) == 1:
                all_alerts[sys_id].append(t)
                if min_s - 1e-6 <= t <= max_s + 1e-6:
                    scoring_alerts[sys_id].append(t)
        if sys_id in matching_ids and 0 <= t <= min_s:
            positions[sys_id].append((t, float(r['cx']), float(r['cy'])))
    for id_ in positions:
        positions[id_].sort()
    if n_in_window == 0:
        raise ValueError('No VALID system observations in the requested Day 12 window.')
    return all_alerts, scoring_alerts, summary_alerts, positions, n_in_window


def spatial_match(boxes: list[Box], fps: float,
                  positions: list[tuple[float, float, float]],
                  end_s: float = 120.0, tol_s: float = 1.6):
    if not positions:
        return 0, None
    times = [x[0] for x in positions]
    distances = []
    for b in boxes:
        if b.outside or b.occluded or b.frame / fps > end_s:
            continue
        t = b.frame / fps
        idx = bisect_left(times, t)
        choices = [j for j in (idx-1, idx) if 0 <= j < len(times)]
        if not choices:
            continue
        nearest = min(choices, key=lambda j: abs(times[j]-t))
        ts, cx, cy = positions[nearest]
        if abs(ts-t) <= tol_s:
            distances.append(((b.cx-cx)**2 + (b.cy-cy)**2)**0.5)
    return len(distances), median(distances) if distances else None


def safe_div(n: float, d: float):
    return n / d if d else None


def format_metric(n: float | None):
    return 'N/A' if n is None else f'{100*n:.2f}%'


def score(rows: list[dict], statuses: set[str]) -> dict:
    eligible = [r for r in rows if r['review_status'] in statuses and r['result'] in ('TP','TN','FP','FN')]
    counts = {k: sum(r['result'] == k for r in eligible) for k in ('TP','TN','FP','FN')}
    tp, tn, fp, fn = (counts[k] for k in ('TP','TN','FP','FN'))
    prec = safe_div(tp, tp+fp)
    rec = safe_div(tp, tp+fn)
    f1 = safe_div(2*tp, 2*tp+fp+fn)  # undefined if no positive cases
    return {**counts, 'evaluated_cases': len(eligible),
            'accuracy': safe_div(tp+tn, tp+tn+fp+fn),
            'precision': prec, 'recall': rec, 'f1': f1,
            'specificity': safe_div(tn, tn+fp)}


def main():
    parser = argparse.ArgumentParser(description='Audit/validate Day12 30-minute inactivity from CVAT XML and Hybrid V2 CSVs')
    parser.add_argument('--xml', type=Path, default=Path('day12_annotations_validations.xml'))
    parser.add_argument('--observations', type=Path, default=Path('day12_nongap_observations.csv'))
    parser.add_argument('--summary', type=Path, default=Path('day12_nongap_summary.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('day12_validation_results'))
    parser.add_argument('--labels', type=Path, default=None,
                        help='Editable review labels CSV; defaults to output_dir/day12_review_labels.csv')
    parser.add_argument('--video-duration-s', type=float, default=None,
                        help='CVAT video duration in seconds (default: stop_frame / fps)')
    parser.add_argument('--fps', type=float, default=15.0,
                        help='ACTUAL FPS of the CVAT-uploaded video (default 15: the Day 12 recording, checked by box alignment)')
    parser.add_argument('--window-s', type=float, default=1800.0, help='GT inactivity duration in seconds (default 1800)')
    parser.add_argument('--alert-grace-s', type=float, default=10.0,
                        help='Check system alerts from window_s to window_s+grace (default 10s)')
    args = parser.parse_args()
    if (args.video_duration_s is not None and args.video_duration_s <= 0) or args.window_s <= 0 or args.alert_grace_s < 0:
        parser.error('Duration/window must be positive and grace >= 0.')
    try:
        tracks, metadata = parse_cvat(args.xml)
        raw_filter = metadata['frame_filter'].strip()
        if not raw_filter.startswith('step='):
            raise ValueError(f'Unsupported frame_filter {raw_filter!r}; expected step=N.')
        step = int(raw_filter.split('=', 1)[1])
        if step <= 0:
            raise ValueError('Invalid CVAT frame step.')
        stop = int(metadata['stop_frame'])
        fps = args.fps
        if fps <= 0:
            raise ValueError('Video FPS must be positive.')
        if args.video_duration_s is None:
            args.video_duration_s = (stop + step) / fps
        if args.window_s * fps > stop + step:
            raise ValueError(
                f'The CVAT annotation covers {(stop + step) / fps:.0f} s ({(stop + step) / fps / 60:.1f} min) at {fps:g} FPS, '
                f'less than the {args.window_s:.0f} s observation window: the {args.window_s / 60:g}-minute rule cannot be '
                'validated on it. Annotate at least that long, or pass a shorter --window-s for an exploratory check.')
        if args.window_s + args.alert_grace_s > args.video_duration_s + 1:
            raise ValueError('Prediction check extends beyond video duration.')
        audits = {id_: audit_one(id_, boxes, fps, step, args.window_s)
                  for id_, boxes in tracks.items()}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = args.labels or args.output_dir / 'day12_review_labels.csv'
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        created = create_labels(labels_path, audits)
        labels = load_labels(labels_path, audits)
        mapped = {row['sys_id']: id_ for id_, row in labels.items() if row['sys_id'] is not None}
        all_alerts, scoring_alerts, summary_alerts, positions, valid_rows = load_system(
            args.observations, args.summary, args.video_duration_s,
            args.window_s, args.window_s + args.alert_grace_s, set(mapped))
        unmatched = sorted(set(scoring_alerts) - set(mapped))
        if any(id_ not in summary_alerts for id_ in scoring_alerts):
            print('WARNING: some alerts in observations are absent from summary.csv', file=sys.stderr)

        audit_rows = []
        case_rows = []
        for id_, a in sorted(audits.items()):
            gt = labels[id_]
            sysid = gt['sys_id']
            match_n, match_median = spatial_match(tracks[id_], fps, positions.get(sysid, [])) if sysid is not None else (0, None)
            warnings = []
            if a.outside_first30 or a.occluded_first30:
                warnings.append('Occlusion/outside in GT window')
            if a.candidate == 'INSUFFICIENT_COVERAGE':
                warnings.append('Partial GT window')
            if sysid is not None and (match_n < 5 or (match_median is not None and match_median > 70)):
                warnings.append('System–CVAT spatial match needs review')
            if gt['status'] == 'confirmed' and a.candidate == 'INSUFFICIENT_COVERAGE':
                warnings.append('Confirmed GT label lacks full CVAT coverage')
            if gt['status'] == 'exclude' or gt['gt'] is None:
                pred, result = '', 'EXCLUDE'
            elif a.candidate == 'INSUFFICIENT_COVERAGE':
                pred, result = '', 'EXCLUDE_INSUFFICIENT_CVAT_COVERAGE'
            elif unmatched:
                pred, result = '', 'UNRESOLVED_SYSTEM_ALERT'
            else:
                pred = int(sysid in scoring_alerts) if sysid is not None else 0
                truth = gt['gt']
                result = ('TP' if pred else 'FN') if truth == 1 else ('FP' if pred else 'TN')
            audit_rows.append({
                'cvat_id': id_, 'candidate': a.candidate,
                'samples_first_30m': a.sampled_boxes, 'visible_samples_first_30m': a.valid_boxes,
                'manual_keyframes_first_30m': a.manual_keyframes,
                'manual_dx_px': fmt(a.manual_dx), 'manual_dy_px': fmt(a.manual_dy),
                'half_body_length_px': fmt(a.half_body_px),
                'outside_first_30m': a.outside_first30,
                'occluded_first_30m': a.occluded_first30,
                'last_manual_keyframe_s': fmt(a.last_manual_s),
                'notes': a.explanation})
            case_rows.append({
                'cvat_id': id_, 'ground_truth_inactive': gt['ground_truth_inactive'],
                'review_status': gt['status'], 'system_track_id': sysid if sysid is not None else '',
                'system_pred_inactive': pred, 'result': result,
                'manual_dx_px': fmt(a.manual_dx), 'manual_dy_px': fmt(a.manual_dy),
                'half_body_length_px': fmt(a.half_body_px),
                'spatial_match_samples_first_2m': match_n,
                'spatial_match_median_px_first_2m': fmt(match_median),
                'warnings': '; '.join(warnings), 'review_notes': gt.get('notes', '')})

        first_window_metrics = score(case_rows, {'confirmed'})
        provisional_metrics = score(case_rows, {'confirmed','provisional'})
        metric_rows = []
        for label, scores, caveat in [
            ('CONFIRMED_ONLY', first_window_metrics, 'Human-confirmed CVAT labels only.'),
            ('INCLUDING_PROVISIONAL', provisional_metrics,
             'Research audit only; movement-based negatives need identity review.')
        ]:
            metric_rows.append({
                'tier': label, **{k: fmt(v, 6) if isinstance(v, float) else v for k,v in scores.items()},
                'caveat': caveat
            })
        alert_rows = []
        for sysid in sorted(set(all_alerts) | set(summary_alerts)):
            alert_rows.append({
                'system_track_id': sysid, 'matched_cvat_id': mapped.get(sysid, ''),
                'first_alert_in_35min_s': fmt(min(all_alerts[sysid]),3) if all_alerts[sysid] else '',
                'alert_samples_in_35min': len(all_alerts[sysid]),
                'first_alert_in_scoring_window_s': fmt(min(scoring_alerts[sysid]),3) if scoring_alerts[sysid] else '',
                'alert_samples_in_scoring_window': len(scoring_alerts[sysid]),
                'summary_inactivity_frames_full_session': summary_alerts.get(sysid, {}).get('inactivity_frames','')
            })
        write_csv(args.output_dir/'day12_cvat_audit.csv',
                  ['cvat_id','candidate','samples_first_30m','visible_samples_first_30m',
                   'manual_keyframes_first_30m','manual_dx_px','manual_dy_px',
                   'half_body_length_px','outside_first_30m','occluded_first_30m',
                   'last_manual_keyframe_s','notes'], audit_rows)
        write_csv(args.output_dir/'day12_validation_cases.csv',
                  ['cvat_id','ground_truth_inactive','review_status','system_track_id',
                   'system_pred_inactive','result','manual_dx_px','manual_dy_px',
                   'half_body_length_px','spatial_match_samples_first_2m',
                   'spatial_match_median_px_first_2m','warnings','review_notes'], case_rows)
        write_csv(args.output_dir/'day12_validation_metrics.csv',
                  ['tier','evaluated_cases','TP','TN','FP','FN','accuracy','precision','recall',
                   'f1','specificity','caveat'], metric_rows)
        write_csv(args.output_dir/'day12_system_alerts.csv',
                  ['system_track_id','matched_cvat_id','first_alert_in_35min_s',
                   'alert_samples_in_35min','first_alert_in_scoring_window_s',
                   'alert_samples_in_scoring_window','summary_inactivity_frames_full_session'], alert_rows)
        lines = [
            'DAY 12 — INACTIVITY VALIDATION (INITIAL 30-MINUTE WINDOW)',
            '=======================================================',
            f'CVAT tracks: {len(tracks)}; sampled frames: {metadata["size"]}; frame step: {step}',
            f'CVAT FPS used: {fps:g} (15 = the Day 12 recording, checked by box alignment; override with --fps)',
            f'GT window: 0.0–{args.window_s:.1f} seconds',
            f'System alert scoring window: {args.window_s:.1f}–{args.window_s+args.alert_grace_s:.1f} seconds',
            f'System VALID observation rows within first {args.video_duration_s:g} s: {valid_rows}',
            f'System IDs alerting in first {args.video_duration_s:g} s: {sorted(all_alerts)}',
            f'System IDs alerting within scoring window: {sorted(scoring_alerts)}',
            f'Unmatched alert system IDs in scoring window: {unmatched}',
            f'CVAT-to-system ID mapping from review file: { {v:k for k,v in mapped.items()} }',
            '',
        ]
        for tier, metrics in [('CONFIRMED ONLY', first_window_metrics),
                              ('CONFIRMED + PROVISIONAL', provisional_metrics)]:
            lines += [
                tier,
                f'Cases: {metrics["evaluated_cases"]}, TP={metrics["TP"]}, TN={metrics["TN"]}, '
                f'FP={metrics["FP"]}, FN={metrics["FN"]}',
                'Accuracy: ' + format_metric(metrics['accuracy']),
                'Precision: ' + format_metric(metrics['precision']),
                'Recall: ' + format_metric(metrics['recall']),
                'F1-score: ' + format_metric(metrics['f1']),
                'Specificity: ' + format_metric(metrics['specificity']),
                ''
            ]
        lines += [
            'INTERPRETATION / CAUTIONS',
            '- Day 12 CVAT ID 1 is CONFIRMED inactive by annotator; matched to system ID 15 using position.',
            '- The other 21 initial-window negatives start PROVISIONAL; do NOT present provisional 100% as final thesis accuracy.',
            '- Changing review_status to confirmed in day12_review_labels.csv requires independent physical-ID review.',
            '- A CVAT interpolation-only segment is not proof of observed stationary behavior.',
            '- Outside/occluded intervals are unknown positions, never stationary evidence.',
            '- If an alerting system ID is unmatched, bird-wise scores are withheld rather than invented.',
            '- This evaluates a single initial 30-minute chicken-window per eligible bird, not arbitrary overlapping windows.',
            '- Distinguish ID-matching confidence from behavioral GT confirmation.',
            '- Only one confirmed positive event: results do not quantify performance across diverse positives.',
        ]
        report_path = args.output_dir/'day12_validation_report.txt'
        report_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines))
        if created:
            print(f'CREATED editable review labels: {labels_path}')
        else:
            print(f'REUSED existing review labels: {labels_path}')
        print(f'CSV reports and text summary saved to: {args.output_dir.resolve()}')
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
