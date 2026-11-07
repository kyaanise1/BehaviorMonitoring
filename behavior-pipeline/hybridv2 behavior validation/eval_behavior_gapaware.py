"""Time-grid behavior F1 that EXCLUDES GAP and RECOVERING samples.

Ground truth format: type,start_s,end_s with type huddling or inactivity.
This is binary ANY-event evaluation, NOT per-chicken or per-group scoring.
"""
import argparse
import csv
import math
from bisect import bisect_right


def load_gt(path):
    out = {"huddling": [], "inactivity": []}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            kind = row["type"].strip().lower()
            if kind not in out:
                raise ValueError(f"Unknown GT type: {kind}")
            a, b = float(row["start_s"]), float(row["end_s"])
            if b < a:
                raise ValueError(f"GT end before start: {row}")
            out[kind].append((a,b))
    return out


def load_pred(path):
    rows=[]
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        if "status" not in rd.fieldnames:
            raise ValueError("Expected gap-aware CSV with status column")
        for row in rd:
            t = float(row["t_s"])
            status = row["status"].strip().upper()
            if status not in ("VALID", "GAP", "RECOVERING"):
                raise ValueError(f"Unexpected status {status!r} at t={t}")
            hp = int(row["n_huddles"]) > 0 if status=="VALID" else False
            ip = int(row["n_inactive"]) > 0 if status=="VALID" else False
            rows.append((t,status,hp,ip))
    if not rows:
        raise ValueError("Prediction CSV is empty")
    if any(rows[i][0] >= rows[i+1][0] for i in range(len(rows)-1)):
        raise ValueError("Timestamps must increase strictly")
    return rows


def compute(pred, gt, step):
    times=[r[0] for r in pred]
    counts={"huddling": [0,0,0,0], "inactivity": [0,0,0,0]}
    valid=0
    skipped=0
    total = int(math.floor(pred[-1][0]/step))+1
    for i in range(total):
        t=i*step
        idx=bisect_right(times,t)-1
        if idx < 0 or pred[idx][1] != "VALID":
            skipped += 1
            continue
        valid+=1
        for kind,pi in (("huddling",2),("inactivity",3)):
            actual=any(a<=t<=b for a,b in gt[kind])
            predicted=pred[idx][pi]
            c=counts[kind]
            if actual: c[3]+=1
            if actual and predicted: c[0]+=1
            elif predicted: c[1]+=1
            elif actual: c[2]+=1
    return counts, valid, skipped


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred",required=True)
    ap.add_argument("--gt",required=True)
    ap.add_argument("--step",type=float,default=5.0)
    args=ap.parse_args()
    if args.step<=0: ap.error("step must be positive")
    pred=load_pred(args.pred)
    gt=load_gt(args.gt)
    counts,valid,skipped=compute(pred,gt,args.step)
    print(f"Scored {valid} valid time points; excluded {skipped} GAP/RECOVERING time points")
    print("class         precision    recall        F1  TP FP FN (GT-positive points)")
    for name, (tp,fp,fn,npos) in counts.items():
        p=tp/(tp+fp) if tp+fp else None
        r=tp/(tp+fn) if tp+fn else None
        f1=2*p*r/(p+r) if p is not None and r is not None and p+r else None
        fmt=lambda x: f"{x:>9.3f}" if x is not None else f"{'N/A':>9}"
        print(f"{name:<12} {fmt(p)} {fmt(r)} {fmt(f1)}   {tp} {fp} {fn} ({npos})")
    print("N/A means a metric is undefined, such as recall with no ground-truth positives.")
    print("Binary presence only: separately validate chicken IDs and huddle membership.")

if __name__=="__main__":
    main()
