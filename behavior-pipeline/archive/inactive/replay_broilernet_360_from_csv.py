#!/usr/bin/env python3
"""Replay Broiler-Net-inspired 30-min cumulative activity on EXISTING tracking CSVs.

This reuses the tracked positions already saved by run_behavior_system.py,
so it is an algorithm-only comparison without re-running YOLO or changing
Hybrid V2 tracking. Does not modify the original files.

Default names: inactivity_observations.csv, behavior_system_log.csv
Outputs: an activity audit CSV and a revised behavior-system log.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

# Original project's behavior.py lives one level above the validation scripts.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from behavior_inactivity_broilernet import InactivityModule, AUDIT_FIELDS


def read_csv(path, cols):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        missing=set(cols)-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader), reader.fieldnames


def main():
    ap=argparse.ArgumentParser(description='Replay 360px cumulative 30-min activity on stored Hybrid V2 observations')
    ap.add_argument('--observations',type=Path,default=Path('inactivity_observations.csv'))
    ap.add_argument('--log',type=Path,default=Path('behavior_system_log.csv'))
    ap.add_argument('--out-dir',type=Path,default=Path('broilernet_360_replay'))
    ap.add_argument('--window',type=float,default=1800.0)
    ap.add_argument('--interval',type=float,default=2.0)
    ap.add_argument('--activity-threshold-px',type=float,default=360.0)
    ap.add_argument('--min-fill',type=float,default=0.95)
    ap.add_argument('--max-gap-s',type=float,default=None)
    args=ap.parse_args()
    try:
        obs,_ = read_csv(args.observations,('t_s','track_id','cx','cy'))
        logs,log_fields=read_csv(args.log,('t_s','n_birds','n_inactive','inactive_ids'))
        if not logs: raise ValueError('No system-log rows')
        bytime=defaultdict(list)
        for row in obs:
            bytime[round(float(row['t_s']),3)].append(row)
        mod=InactivityModule(window=args.window,interval=args.interval,
                             activity_threshold_px=args.activity_threshold_px,
                             min_fill=args.min_fill,max_gap_s=args.max_gap_s)
        args.out_dir.mkdir(exist_ok=True,parents=True)
        audit_file=args.out_dir/'broilernet_360_inactivity_observations.csv'
        log_file=args.out_dir/'broilernet_360_behavior_system_log.csv'
        with audit_file.open('w',newline='',encoding='utf-8-sig') as fo, log_file.open('w',newline='',encoding='utf-8-sig') as fl:
            audit=csv.writer(fo); audit.writerow(['t_s','session',*AUDIT_FIELDS])
            outlog=csv.DictWriter(fl,fieldnames=[*log_fields,'activity_threshold_px','inactivity_method'],extrasaction='ignore')
            outlog.writeheader()
            npos=0; initial=None; positive_ids=set(); log_times=set()
            for r in sorted(logs,key=lambda x:float(x['t_s'])):
                t=float(r['t_s']); key=round(t,3)
                if key in log_times: raise ValueError(f'Duplicate system sample timestamp {t:.3f}')
                log_times.add(key)
                rows=bytime.pop(key,[])
                ids=[int(s['track_id']) for s in rows]
                boxes=[[float(s['cx']),float(s['cy']),1.,1.] for s in rows]
                body=float(r['body_len_px']) if r.get('body_len_px') else None
                sample=mod.observe(t,ids,boxes,audit=True,body_len_override=body)
                for record in sample.audit_rows:
                    audit.writerow([f'{t:.3f}',0,*record])
                predicted=set(sample.inactive)
                newr=dict(r)
                newr['n_inactive']=len(predicted)
                newr['inactive_ids']='|'.join(str(i) for i in sorted(predicted))
                newr['activity_threshold_px']=f'{args.activity_threshold_px:g}'
                newr['inactivity_method']='broilernet_inspired_cumulative_px_30min'
                outlog.writerow(newr)
                if predicted:
                    npos+=1; positive_ids.update(predicted)
                    if initial is None:initial=(t,predicted)
        unmatched=sum(len(v) for v in bytime.values())
        print(f'Processed {len(logs)} sampled time rows, {len(obs)} per-bird observations')
        print(f'Provisional activity threshold: {args.activity_threshold_px:g} px; window={args.window:g}s; min-fill={args.min_fill:g}')
        print(f'Samples with at least one inactive alert: {npos}; IDs alerted: {sorted(positive_ids)}')
        print(f'First alert: {initial}' if initial else 'First alert: NONE')
        print(f'Unmatched observation CSV rows: {unmatched}')
        print(f'Saved: {audit_file.resolve()}')
        print(f'Saved: {log_file.resolve()}')
    except (OSError, ValueError) as e:
        ap.exit(2,f'ERROR: {e}\n')


if __name__=='__main__':
    main()
