"""Draw ground-truth and predicted tracking IDs side by side on existing frames.

Example (from behavior-pipeline):
  python visualize_track_frames.py --frames tracking_frames \
      --gt annotated_tracking_frames/gt/gt.txt --pred pred_hybrid.txt \
      --start 4 --end 9 --gt-id 7

Requires: pip install opencv-python
Does NOT modify existing predictions or trackers.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def load_mot(path):
    """MOT format: frame, id, left, top, width, height, ..."""
    by_frame = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 6 or row[0].startswith("#"):
                continue
            try:
                fr = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, w, h = map(float, row[2:6])
                if w <= 0 or h <= 0:
                    continue
            except ValueError:
                continue
            # MOT GT files use a confidence/validity field at column 7;
            # ignore explicitly excluded entries when present.
            if len(row) >= 7 and path == load_mot.gt_path:
                try:
                    if float(row[6]) <= 0:
                        continue
                except ValueError:
                    pass
            by_frame[fr].append((tid, x, y, w, h))
    return by_frame


load_mot.gt_path = ""


def draw_boxes(image, records, prefix, base_color, highlight_id=None):
    h_img, w_img = image.shape[:2]
    scale = max(0.42, min(0.80, w_img / 950.0))
    line_h = int(22 * scale) + 5
    for tid, x, y, w, h in records:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x+w)), int(round(y+h))
        x1, x2 = max(0,x1), min(w_img-1,x2)
        y1, y2 = max(0,y1), min(h_img-1,y2)
        if x1 >= x2 or y1 >= y2:
            continue
        color = (0, 255, 255) if tid == highlight_id else base_color
        thickness = 3 if tid == highlight_id else 1
        cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
        tag = f"{prefix}{tid}"
        (tw,th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        text_y = y1 - 4 if y1 >= th + 8 else min(h_img-3, y1 + th + line_h)
        bg_top = max(0, text_y - th - 3)
        bg_bottom = min(h_img - 1, text_y + baseline + 2)
        cv2.rectangle(image, (x1,bg_top), (min(w_img-1,x1+tw+5),bg_bottom), color, -1)
        cv2.putText(image, tag, (x1+2, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    ap = argparse.ArgumentParser(description="Visualize tracking results alongside annotations")
    ap.add_argument("--frames", default="tracking_frames")
    ap.add_argument("--gt", default="annotated_tracking_frames/gt/gt.txt")
    ap.add_argument("--pred", default="pred_hybrid.txt")
    ap.add_argument("--start", type=int, default=4)
    ap.add_argument("--end", type=int, default=9)
    ap.add_argument("--gt-id", type=int, default=None, help="Optional GT ID to highlight in yellow")
    ap.add_argument("--pred-id", type=int, default=None, help="Optional predicted ID to highlight in yellow")
    ap.add_argument("--outdir", default="switch_review")
    args = ap.parse_args()
    if args.start < 1 or args.end < args.start:
        ap.error("Expected 1 <= --start <= --end")

    files = sorted(list(Path(args.frames).glob("*.jpg")) + list(Path(args.frames).glob("*.png")))
    if not files:
        ap.error(f"No JPG/PNG frames found in {args.frames}")
    load_mot.gt_path = args.gt
    gt = load_mot(args.gt)
    pred = load_mot(args.pred)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for fr in range(args.start, args.end + 1):
        if fr > len(files):
            print(f"Frame {fr} exceeds image count ({len(files)}); skipping")
            continue
        img = cv2.imread(str(files[fr - 1]))
        if img is None:
            print(f"Could not read {files[fr - 1]}; skipping")
            continue
        left = draw_boxes(img.copy(), gt.get(fr, []), "GT", (80, 190, 70), args.gt_id)
        right = draw_boxes(img.copy(), pred.get(fr, []), "P", (40, 145, 240), args.pred_id)
        h, w = img.shape[:2]
        cv2.rectangle(left, (0, 0), (w-1, 30), (40, 105, 40), -1)
        cv2.rectangle(right, (0, 0), (w-1, 30), (75, 80, 125), -1)
        cv2.putText(left, f"Frame {fr}: GROUND TRUTH ({len(gt.get(fr, []))})", (10,21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(right, f"Frame {fr}: HYBRID PREDICTION ({len(pred.get(fr, []))})", (10,21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        both = cv2.hconcat([left, right])
        dest = out / f"frame_{fr:04d}.jpg"
        cv2.imwrite(str(dest), both, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"Saved {dest} | source image: {files[fr-1].name}")
        count += 1
    print(f"Done: {count} side-by-side review images in '{out}'")


if __name__ == "__main__":
    main()
