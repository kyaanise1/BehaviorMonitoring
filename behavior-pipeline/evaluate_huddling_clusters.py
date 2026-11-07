"""Compare manually annotated YOLO huddle boxes with cluster-export CSVs.

Use --timestamps-csv to evaluate ALL sampled frames. A sampled frame without
a corresponding YOLO TXT file is treated as having zero ground-truth huddles.
No video inference required. Python standard library only.
"""
import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

TIME_RE = re.compile(r'_t_(\d+(?:\.\d+)?)s(?:\.txt)?$', re.I)


def timestamp_from_name(name):
    match = TIME_RE.search(Path(name).name)
    return round(float(match.group(1)), 3) if match else None


def load_annotations(path, width, height, class_id, allow_empty=False):
    source = Path(path)
    if source.is_file() and source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as z:
            entries = [(name, z.read(name).decode('utf-8-sig')) for name in z.namelist()
                       if name.lower().endswith('.txt') and not name.lower().endswith('classes.txt')]
    elif source.is_dir():
        entries = [(str(p), p.read_text(encoding='utf-8-sig')) for p in source.rglob('*.txt')
                   if p.name.lower() != 'classes.txt']
    else:
        raise ValueError('Annotations must be a ZIP or a directory of YOLO .txt files')
    labels = {}
    for name, content in entries:
        t = timestamp_from_name(name)
        if t is None:
            raise ValueError(f'Annotation filename missing _t_...s timestamp: {name}')
        if t in labels:
            raise ValueError(f'Duplicate annotation timestamp {t}: {name}')
        boxes = []
        for line_no, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'{name}:{line_no}: expected class cx cy w h')
            cls, cx, cy, w, h = map(float, parts)
            if int(cls) != class_id:
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f'{name}:{line_no}: invalid normalized box')
            boxes.append(((cx-w/2)*width, (cy-h/2)*height,
                          (cx+w/2)*width, (cy+h/2)*height))
        labels[t] = boxes
    if not labels and not allow_empty:
        raise ValueError('No timestamped annotation TXT files found')
    return labels


def load_validation_timestamps(path):
    """Return the complete preselected validation frame set from extractor CSV."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 't_s' not in (reader.fieldnames or []):
            raise ValueError(f'{path}: validation timestamp CSV needs a t_s column')
        timestamps = []
        for row_no, row in enumerate(reader, 2):
            try:
                t = round(float(row['t_s']), 3)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{row_no}: invalid t_s value') from exc
            if 'image_filename' in row and row['image_filename']:
                from_filename = timestamp_from_name(Path(row['image_filename']).with_suffix('.txt').name)
                if from_filename is not None and from_filename != t:
                    raise ValueError(f'{path}:{row_no}: image filename timestamp differs from t_s')
            timestamps.append(t)
    if not timestamps:
        raise ValueError(f'{path}: no validation timestamps')
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f'{path}: duplicated t_s values (rounded to milliseconds)')
    return timestamps


def load_predictions(path):
    pred = defaultdict(list)
    seen = set()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'t_s', 'member_boxes_xywh_json'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path} must contain {sorted(required)}')
        for row in reader:
            t = round(float(row['t_s']), 3)
            seen.add(t)
            members = json.loads(row['member_boxes_xywh_json'] or '[]')
            if not members:
                continue
            pred[t].append((min(float(b['cx'])-float(b['w'])/2 for b in members),
                            min(float(b['cy'])-float(b['h'])/2 for b in members),
                            max(float(b['cx'])+float(b['w'])/2 for b in members),
                            max(float(b['cy'])+float(b['h'])/2 for b in members)))
    return pred, seen


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    union = aa+bb-intersection
    return intersection/union if union else 0.0


def match_count(truth, predicted, threshold):
    # Maximum-cardinality one-to-one matching; sort higher IoUs first for stable ties.
    edges = [sorted((j for j in range(len(truth)) if iou(p, truth[j]) >= threshold),
                    key=lambda j: -iou(p, truth[j])) for p in predicted]
    assigned = {}
    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or augment(assigned[j], visited):
                assigned[j] = i
                return True
        return False
    for i in sorted(range(len(predicted)), key=lambda i: -max((iou(predicted[i], g) for g in truth), default=0)):
        augment(i, set())
    return len(assigned)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', required=True, help='ZIP or folder with YOLO TXT labels')
    ap.add_argument('--timestamps-csv', help='Complete selected-frame CSV (t_s). Missing TXT means no huddles for that frame')
    ap.add_argument('--baseline', default='baseline_clusters.csv')
    ap.add_argument('--feeder-aware', default='feeder_aware_clusters.csv')
    ap.add_argument('--strict', default='strict_huddling_clusters.csv')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--class-id', type=int, default=2, help='Huddling class index in classes.txt')
    ap.add_argument('--output', default='huddling_evaluation.csv')
    ap.add_argument('--per-frame', default='huddling_evaluation_per_frame.csv')
    args = ap.parse_args()
    gt = load_annotations(args.annotations, args.width, args.height, args.class_id,
                          allow_empty=bool(args.timestamps_csv))
    if args.timestamps_csv:
        timestamps = load_validation_timestamps(args.timestamps_csv)
        unexpected = sorted(set(gt) - set(timestamps))
        if unexpected:
            raise ValueError(
                f'Annotation timestamps absent from {args.timestamps_csv}: {unexpected}. '
                'Check you selected the correct validation timestamp CSV.'
            )
        unlabeled = sorted(set(timestamps) - set(gt))
        for t in unlabeled:
            gt[t] = []  # Explicit convention: no TXT = zero ground-truth huddles.
        print(f'Validation timestamps: {len(timestamps)} frames; '
              f'{len(unlabeled)} missing annotation TXT files treated as no huddles.')
    methods = [('Baseline', args.baseline), ('Feeder-aware', args.feeder_aware), ('Strict', args.strict)]
    summary, details = [], []
    for method, path in methods:
        predictions, seen = load_predictions(path)
        if not args.timestamps_csv:
            # Legacy mode: each annotated timestamp must appear in the prediction CSV.
            missing = sorted(set(gt) - seen)
            if missing:
                raise ValueError(f'{method}: missing predictions for {len(missing)} annotated timestamps: {missing}')
        # With a full validation manifest, no prediction row means zero predicted
        # clusters, which is normal for cluster-only CSV output.
        for threshold in (0.50, 0.30):
            tp = fp = fn = 0
            for t in sorted(gt):
                actual, predicted = gt[t], predictions[t]
                hits = match_count(actual, predicted, threshold)
                a, b, c = hits, len(predicted)-hits, len(actual)-hits
                tp += a; fp += b; fn += c
                details.append({'method': method, 'iou_threshold': threshold, 't_s': f'{t:.3f}',
                                'manual_clusters': len(actual), 'predicted_clusters': len(predicted),
                                'TP': a, 'FP': b, 'FN': c})
            precision = tp/(tp+fp) if tp+fp else 0
            recall = tp/(tp+fn) if tp+fn else 0
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
            summary.append({'method': method, 'iou_threshold': threshold, 'frames': len(gt),
                            'manual_clusters': sum(map(len, gt.values())), 'TP': tp, 'FP': fp, 'FN': fn,
                            'precision': f'{precision:.4f}', 'recall': f'{recall:.4f}', 'F1': f'{f1:.4f}'})
    for filename, rows in [(args.output, summary), (args.per_frame, details)]:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(f'Validated {len(gt)} frames; {sum(map(len, gt.values()))} manually labeled clusters')
    print(f'{"Method":<15} {"IoU":>5} {"TP":>4} {"FP":>4} {"FN":>4} {"Precision":>10} {"Recall":>8} {"F1":>8}')
    for r in summary:
        print(f'{r["method"]:<15} {r["iou_threshold"]:>5.2f} {r["TP"]:>4} {r["FP"]:>4} {r["FN"]:>4} {r["precision"]:>10} {r["recall"]:>8} {r["F1"]:>8}')
    print(f'Saved {args.output} and {args.per_frame}')


if __name__ == '__main__':
    main()
