"""Behavior F1 (inactivity, huddling) against manual annotation.

Inputs
  --pred : state_log.csv written by run_pipeline.py
           columns: t_s, n_birds, n_huddles, n_inactive, body_len_px
  --gt   : your manual annotation, CSV with columns:  type,start_s,end_s
           type is "huddling" or "inactivity"; times are video seconds.
           Example:
               type,start_s,end_s
               huddling,120,310
               inactivity,0,1900

Method: on a fixed grid (default every 5 s, as in the proposal) each class is a
binary label per time step. Pipeline state = the latest log row at or before
that time. Precision / recall / F1 are computed per class.

Note: inactivity can only be true after one full window of footage, so for a
fair comparison run the pipeline and annotate with the same --window.
"""
import argparse
import csv


def load_pred(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t_s"]), int(r["n_huddles"]) > 0, int(r["n_inactive"]) > 0))
    return rows


def load_gt(path):
    gt = {"huddling": [], "inactivity": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[r["type"].strip()].append((float(r["start_s"]), float(r["end_s"])))
    return gt


def in_any(t, intervals):
    return any(s <= t <= e for s, e in intervals)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="state_log.csv")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--step", type=float, default=5.0)
    args = ap.parse_args()

    pred = load_pred(args.pred)
    gt = load_gt(args.gt)
    t_end = pred[-1][0]

    counts = {"huddling": [0, 0, 0], "inactivity": [0, 0, 0]}   # tp, fp, fn
    t, idx = 0.0, 0
    while t <= t_end:
        while idx + 1 < len(pred) and pred[idx + 1][0] <= t:
            idx += 1
        _, huddle_p, inactive_p = pred[idx]
        for cls, p in (("huddling", huddle_p), ("inactivity", inactive_p)):
            g = in_any(t, gt[cls])
            c = counts[cls]
            if p and g: c[0] += 1
            elif p and not g: c[1] += 1
            elif g and not p: c[2] += 1
        t += args.step

    print(f"{'class':<12}{'precision':>10}{'recall':>9}{'F1':>7}   (tp/fp/fn)")
    for cls, (tp, fp, fn) in counts.items():
        p, r, f1 = prf(tp, fp, fn)
        print(f"{cls:<12}{p:>10.3f}{r:>9.3f}{f1:>7.3f}   ({tp}/{fp}/{fn})")
    print("\nTarget from the proposal: F1 >= 0.85 per class")


if __name__ == "__main__":
    main()