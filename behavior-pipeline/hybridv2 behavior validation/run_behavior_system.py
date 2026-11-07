"""One shared Hybrid V2 behavior controller, WITHOUT gap-aware processing.

The inactivity rule is from the uploaded run_pipeline_hybrid_v2(3).py; the
feeder-aware huddling module is unchanged. Both use the SAME tracked birds.

This runner deliberately does not detect frozen frames / corrupted samples,
reset tracker histories, or mark missing footage. Use only on verified
continuous footage for reliable long-duration inactivity monitoring.

RecoveryHybrid retains two YOLO calls per sampled frame (ByteTrack + raw
recovery). This is a validation prototype, not optimized Raspberry Pi firmware.
"""
import argparse
import csv
import json
import sys
from collections import deque
from contextlib import ExitStack
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ultralytics import YOLO
from common import sample_frames
from hybrid_tracker_v2 import RecoveryHybrid
from behavior_huddling_v2 import HuddlingModule, load_feeding_zones
from behavior_inactivity_v2 import InactivityModule


def main():
    ap = argparse.ArgumentParser(
        description='Hybrid V2: shared tracker + feeder-aware huddling + original inactivity (no gap guard)')
    ap.add_argument('--source', required=True, help='Continuous video file; time must be valid')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tracker', default='configs/bytetrack_broiler.yaml')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--window', type=float, default=1800.0)
    ap.add_argument('--move-frac', type=float, default=0.5)
    ap.add_argument('--min-fill', type=float, default=0.5)
    ap.add_argument('--min-huddle', type=int, default=3)
    ap.add_argument('--zones', default='configs/feeding_zones.json')
    ap.add_argument('--huddle-method', choices=['connected', 'complete', 'average', 'contact'], default='complete')
    ap.add_argument('--feeder-margin-px', type=float, default=15.0, help='Margin used when checking full-cluster overlap with feeder zones (reference pixels)')
    ap.add_argument('--hard-feeder-exclusion', action='store_true', help='Legacy mode: exclude every bird whose centroid is inside an expanded feeder zone')
    ap.add_argument('--cluster-zone-overlap', type=float, default=0.15)
    ap.add_argument('--overlap-sweep', default='', help='Comma-separated zone-overlap limits, e.g. 0.15,0.30')
    ap.add_argument('--local-recovery', action='store_true')
    ap.add_argument('--temporal-recovery', action='store_true',
                    help='Huddling recovery requiring two earlier/two later samples')
    ap.add_argument('--clusters-csv', default='', help='Save predicted huddling clusters')
    ap.add_argument('--timestamps-csv', default='', help='CSV with t_s; limit clusters to these timestamps')
    ap.add_argument('--temporal-audit-csv', default='', help='Log temporal huddling additions')
    ap.add_argument('--all-detections-csv', default='', help='Per-detection audit CSV')
    ap.add_argument('--inactivity-observations-csv', default='', help='Per-ID inactivity history audit')
    ap.add_argument('--log-csv', default='behavior_system_log.csv')
    args = ap.parse_args()

    if args.feeder_margin_px < 0:
        ap.error('--feeder-margin-px must be nonnegative')
    if args.interval <= 0 or args.window <= 0:
        ap.error('--interval and --window must be positive')
    if not 0 < args.min_fill <= 1 or args.min_huddle < 2:
        ap.error('--min-fill must be in (0,1] and --min-huddle >= 2')
    if not 0 <= args.cluster_zone_overlap <= 1:
        ap.error('--cluster-zone-overlap must be in [0,1]')
    if args.local_recovery and (args.huddle_method not in ('average', 'contact') or args.min_huddle != 3):
        ap.error('--local-recovery requires --huddle-method average/contact --min-huddle 3')
    if args.temporal_recovery:
        if not args.local_recovery or args.huddle_method not in ('average', 'contact') or args.min_huddle != 3:
            ap.error('--temporal-recovery requires average/contact linkage, --local-recovery, --min-huddle 3')
        if abs(args.cluster_zone_overlap - .30) > 1e-8:
            ap.error('--temporal-recovery requires --cluster-zone-overlap 0.30')
        if not args.zones or args.overlap_sweep:
            ap.error('--temporal-recovery requires zones and does not allow --overlap-sweep')
    if args.temporal_audit_csv and not args.temporal_recovery:
        ap.error('--temporal-audit-csv requires --temporal-recovery')
    if args.timestamps_csv and not args.clusters_csv:
        ap.error('--timestamps-csv requires --clusters-csv')
    if not Path(args.source).exists():
        ap.error(f'Source not found: {args.source}')

    if args.zones:
        try:
            feeding_zones, ref_size = load_feeding_zones(args.zones)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            ap.error(f'Invalid zones JSON: {e}')
    else:
        feeding_zones, ref_size = [], None
    if args.temporal_recovery and len(feeding_zones) != 4:
        print('WARNING: the temporal-recovery development experiment used four zones')

    thresholds = []
    if args.overlap_sweep:
        try:
            thresholds = [float(s.strip()) for s in args.overlap_sweep.split(',')]
        except ValueError:
            ap.error('--overlap-sweep requires comma-separated floats')
        if (not thresholds or any(not 0 < v <= 1 for v in thresholds)
                or len(set(thresholds)) != len(thresholds)):
            ap.error('--overlap-sweep values must be distinct and in (0,1]')
        if not args.clusters_csv:
            ap.error('--overlap-sweep requires --clusters-csv')

    selected_times = None
    if args.timestamps_csv:
        with open(args.timestamps_csv, newline='', encoding='utf-8-sig') as f:
            rows = csv.DictReader(f)
            if 't_s' not in (rows.fieldnames or []):
                ap.error('--timestamps-csv requires a t_s column')
            selected_times = {round(float(row['t_s']), 3) for row in rows}
        if not selected_times:
            ap.error('No timestamps in --timestamps-csv')

    if Path(args.source).is_dir():
        print('WARNING: image folder timestamps are synthetic; not suitable for 30-minute inactivity')
    print('NOTICE: no recording-gap checks or automatic history resets; verify footage continuity yourself.')
    print('Feeder handling: ' + ('hard centroid exclusion (legacy)' if args.hard_feeder_exclusion
                                   else 'cluster-level overlap filtering (no automatic centroid exclusion)'))

    # Identical settings to run_pipeline_hybrid_v2(3).py.
    tracker_model = YOLO(args.weights, task='detect')
    detector_model = YOLO(args.weights, task='detect')
    hybrid = RecoveryHybrid(tight=0.5, wide=4.0, max_missed=4, wide_conf=0.3,
                            size_ratio=2.0, recovery_conf=0.7, recovery_distance=1.5,
                            duplicate_iou=0.5)
    inactivity_module = InactivityModule(args.window, args.interval, args.move_frac, args.min_fill)
    huddling_module = HuddlingModule(feeding_zones, ref_size, args.min_huddle,
                                      args.huddle_method, args.cluster_zone_overlap,
                                      args.local_recovery, bool(thresholds), args.feeder_margin_px)
    temporal_buffer = deque()
    seq = 0
    temporal_added = 0
    recovered_total = 0
    samples_processed = 0

    # First eight columns match the uploaded standalone runner and eval_behavior.py.
    log_fields = [
        't_s', 'n_birds', 'n_huddles', 'n_inactive', 'body_len_px',
        'recovered', 'inactive_ids', 'huddle_ids', 'n_feeding',
        'tracked_histories', 'frame_no',
    ]
    cluster_fields = [
        't_s', 'frame_no', 'cluster_no', 'track_ids',
        'member_boxes_xywh_json', 'cluster_center_x', 'cluster_center_y',
        'n_members', 'n_clusters_in_frame',
    ]
    detections_fields = [
        't_s', 'frame_no', 'source', 'track_id', 'confidence',
        'cx', 'cy', 'w', 'h', 'feeder_excluded', 'huddle_cluster_no',
        'body_len_px', 'n_hybrid_birds', 'frame_width', 'frame_height',
        'feeder_nearby',
    ]
    # session=0 is included only so finish_inactivity_summary.py stays usable.
    inactivity_fields = [
        't_s', 'session', 'track_id', 'cx', 'cy',
        'history_span_s', 'samples_in_window', 'required_samples',
        'span_ok', 'samples_ok', 'range_x_px', 'range_y_px',
        'movement_limit_px', 'movement_ok', 'inactive',
    ]

    with ExitStack() as stack:
        def output_handle(path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return stack.enter_context(open(p, 'w', newline='', encoding='utf-8'))

        log_writer = csv.DictWriter(output_handle(args.log_csv), fieldnames=log_fields)
        log_writer.writeheader()
        detection_writer = None
        if args.all_detections_csv:
            detection_writer = csv.writer(output_handle(args.all_detections_csv))
            detection_writer.writerow(detections_fields)
        inactivity_writer = None
        if args.inactivity_observations_csv:
            inactivity_writer = csv.writer(output_handle(args.inactivity_observations_csv))
            inactivity_writer.writerow(inactivity_fields)
        temporal_audit_writer = None
        if args.temporal_audit_csv:
            temporal_audit_writer = csv.writer(output_handle(args.temporal_audit_csv))
            temporal_audit_writer.writerow([
                't_s', 'stage', 'track_ids', 'zone_overlap',
                'max_box_gap_bodylength', 'mean_box_gap_bodylength',
                'max_centroid_distance_bodylength', 'temporal_support',
                'session',
            ])
        cluster_writers = {}
        for threshold in thresholds or ([args.cluster_zone_overlap] if args.clusters_csv else []):
            path = Path(args.clusters_csv)
            if thresholds:
                path = path.with_name(path.stem + f'_overlap_{round(threshold * 100):02d}pct' + (path.suffix or '.csv'))
            writer = csv.writer(output_handle(path))
            writer.writerow(cluster_fields + (['origin', 'session'] if args.temporal_recovery else ['session']))
            cluster_writers[threshold] = writer

        def export_clusters(writer, record, groups, temporal):
            t, n, ids, boxes = record['t'], record['frame_no'], record['ids'], record['boxes']
            if not groups:
                writer.writerow([f'{t:.3f}', n, '', '', '[]', '', '', 0, 0] +
                                (['', 0] if temporal else [0]))
                return
            for number, (cluster, origin) in enumerate(groups, 1):
                members = [
                    {'track_id': int(ids[j]), 'cx': round(float(boxes[j][0]), 2),
                     'cy': round(float(boxes[j][1]), 2), 'w': round(float(boxes[j][2]), 2),
                     'h': round(float(boxes[j][3]), 2)}
                    for j in cluster
                ]
                writer.writerow([
                    f'{t:.3f}', n, number,
                    '|'.join(str(member['track_id']) for member in members),
                    json.dumps(members, separators=(',', ':')),
                    round(sum(member['cx'] for member in members) / len(members), 2),
                    round(sum(member['cy'] for member in members) / len(members), 2),
                    len(members), len(groups),
                ] + ([origin, 0] if temporal else [0]))

        def emit(record, context):
            nonlocal temporal_added
            if args.temporal_recovery:
                clusters, added = huddling_module.resolve_temporal(record, context)
                temporal_added += len(added)
                if temporal_audit_writer is not None:
                    for line in added:
                        temporal_audit_writer.writerow([*line, 0])
            else:
                clusters = [(list(group), 'primary') for group in record['huddles']]
            ids, boxes = record['ids'], record['boxes']
            t, number = record['t'], record['frame_no']
            huddles = [group for group, _origin in clusters]
            log_writer.writerow(dict(
                t_s=f'{t:.3f}', n_birds=len(ids), n_huddles=len(huddles),
                n_inactive=len(record['inactive']),
                body_len_px=f"{record['body_len']:.2f}" if record['body_len'] else '',
                recovered=record['recovered'],
                inactive_ids='|'.join(map(str, sorted(record['inactive']))),
                huddle_ids=';'.join('|'.join(map(str, sorted(ids[j] for j in group)))
                                    for group in huddles),
                n_feeding=record['n_feeding'], tracked_histories=record['tracked_histories'],
                frame_no=number,
            ))
            if cluster_writers and (selected_times is None or round(t, 3) in selected_times):
                if args.temporal_recovery:
                    export_clusters(cluster_writers[args.cluster_zone_overlap], record, clusters, True)
                else:
                    for threshold, writer in cluster_writers.items():
                        filtered = huddling_module.filter_candidates(
                            record['candidates'], record['overlaps'], threshold,
                            record['keep_indices'], record['boxes'],
                            record['body_len'], record['size'])
                        export_clusters(writer, record, [(group, 'primary') for group in filtered], False)
            if detection_writer is not None:
                excluded = set(range(len(boxes))) - set(record['keep_indices'])
                membership = {j: k for k, group in enumerate(huddles, 1) for j in group}
                def write_det(source, tid, score, box, exclude='', group='', j=None):
                    detection_writer.writerow([
                        f'{t:.3f}', number, source, tid,
                        f'{score:.5f}' if score is not None else '',
                        *[f'{v:.2f}' for v in box], exclude, group,
                        f"{record['body_len']:.2f}" if record['body_len'] else '',
                        len(boxes), *record['size'],
                        (int(j in record['feeding_indices']) if j is not None else ''),
                    ])
                for j, (tid, box) in enumerate(zip(ids, boxes)):
                    write_det('hybrid', tid, record['scores'][j] if j < len(record['scores']) else None,
                              box, int(j in excluded), membership.get(j, ''))
                for tid, box, score in zip(record['byte_ids'], record['byte_boxes'], record['byte_scores']):
                    write_det('bytetrack', tid, score, box)
                for box, score in zip(record['raw_boxes'], record['raw_scores']):
                    write_det('raw_yolo', '', score, box)

        for t, n, frame in sample_frames(args.source, args.interval):
            t = float(t)
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
            recovered_total += recovered
            samples_processed += 1

            activity = inactivity_module.observe(
                t, ids, boxes, audit=inactivity_writer is not None)
            if inactivity_writer is not None:
                for row in activity.audit_rows:
                    inactivity_writer.writerow([f'{t:.3f}', 0, *row])
            body_len = activity.body_len
            spatial = huddling_module.analyze(
                ids, boxes, body_len, (frame.shape[1], frame.shape[0]))
            seq += 1
            record = {
                'seq': seq, 't': t, 'frame_no': n, 'session': 0,
                'tracked_histories': activity.tracked_histories,
                'ids': list(ids), 'boxes': list(boxes), 'body_len': body_len,
                'keep_indices': spatial['keep_indices'],
                'feeding_indices': spatial['feeding_indices'],
                'huddles': spatial['huddles'], 'inactive': activity.inactive,
                'recovered': recovered, 'n_feeding': spatial['n_feeding'],
                'size': (frame.shape[1], frame.shape[0]),
                'candidates': spatial['candidates'], 'overlaps': spatial['overlaps'],
                'scores': list(scores) if detection_writer is not None else [],
                'byte_ids': list(byte_ids) if detection_writer is not None else [],
                'byte_boxes': list(byte_boxes) if detection_writer is not None else [],
                'byte_scores': list(byte_scores) if detection_writer is not None else [],
                'raw_boxes': list(raw_boxes) if detection_writer is not None else [],
                'raw_scores': list(raw_scores) if detection_writer is not None else [],
            }
            if args.temporal_recovery:
                temporal_buffer.append(record)
                if len(temporal_buffer) >= 3:
                    emit(temporal_buffer[-3], list(temporal_buffer))
                    if len(temporal_buffer) > 4:
                        temporal_buffer.popleft()
            else:
                emit(record, [])
            if n is not None and (n % 100 == 0 or activity.inactive):
                print(f'[{t:8.1f}s] frame={n}, birds={len(ids)}, '
                      f'huddles={len(spatial["huddles"])}, inactive={len(activity.inactive)}, '
                      f'in_feeder_zones={spatial["n_feeding"]}')
        if args.temporal_recovery:
            for record in list(temporal_buffer)[-min(2, len(temporal_buffer)):]:
                emit(record, list(temporal_buffer))

    print(f'Saved combined log: {args.log_csv} ({samples_processed} sampled frames)')
    print(f'Recovered detections: {recovered_total}; temporal huddling additions: {temporal_added}')
    print('Inactivity is centroid immobility, NOT a health diagnosis; confirm against time-based annotations.')
    print('No gap checks: missing/frozen footage is not automatically excluded and histories are not reset.')


if __name__ == '__main__':
    main()
