#!/usr/bin/env python3
"""Re-evaluate Option A inactivity from SAVED Hybrid V2 centroid observations.

No YOLO inference, ID reassignment, or huddling recomputation.
Inputs: previous Option A observations.csv + system_log.csv.
Outputs in --out-dir:
  smoothed_inactivity_observations.csv   (raw, median3, median5 activity)
  smoothed_system_log.csv               (unchanged huddling + new inactivity)
  eligible_track_summary.csv            (eligible scores by ID)
  candidate_threshold_sweep.csv         (descriptive; NOT validation)

Requires Python 3.9+ and behavior.py from the original behavior-pipeline root.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0,str(PIPELINE_ROOT))
# Prefer the matching updated module in this folder.
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0,str(SCRIPT_DIR))

from behavior_inactivity_broilernet_smoothed import InactivityModule, AUDIT_FIELDS


def read_csv(path,required):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        missing=set(required)-set(rd.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing {sorted(missing)}')
        rows=list(rd)
        return rows,rd.fieldnames


def main():
    p=argparse.ArgumentParser(description='Recompute Option A raw/median-filtered activity from saved centroid CSVs')
    p.add_argument('--observations',required=True,type=Path)
    p.add_argument('--log',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--window',type=float,default=1800)
    p.add_argument('--interval',type=float,default=2)
    p.add_argument('--activity-threshold-px',type=float,default=450,
                   help='PROVISIONAL score threshold; must be separately calibrated')
    p.add_argument('--smoothing',choices=('raw','median3','median5','block30'),default='median3')
    p.add_argument('--activity-threshold-bl',type=float,default=None,
                   help='block30 threshold in body lengths (default 4.0 with block30)')
    p.add_argument('--min-fill',type=float,default=None,help='default 0.8 with block30, 0.95 otherwise')
    p.add_argument('--max-gap-s',type=float,default=None)
    p.add_argument('--diagnostics-only',action='store_true',
                   help='Do not assign inactive 0/1; export scores for calibration')
    p.add_argument('--compare-thresholds',default='350,400,450,500,550',
                   help='Candidate values for descriptive sweep; do NOT call these validated')
    args=p.parse_args()
    if args.smoothing == 'block30':
        if args.activity_threshold_bl is None:
            args.activity_threshold_bl = 4.0
        if args.min_fill is None:
            args.min_fill = 0.8
        if args.max_gap_s is None:
            args.max_gap_s = 60.0
    elif args.min_fill is None:
        args.min_fill = 0.95
    if args.activity_threshold_px <= 0:
        p.error('--activity-threshold-px must be > 0')
    try:
        candidates=[float(x) for x in args.compare_thresholds.split(',') if x.strip()]
        if not candidates or any(not math.isfinite(x) or x<=0 for x in candidates):
            raise ValueError('all candidate thresholds must be finite and positive')
        obs,_=read_csv(args.observations,('t_s','track_id','cx','cy'))
        logs,log_fields=read_csv(args.log,('t_s','n_birds','n_inactive','inactive_ids'))
        if not logs:
            raise ValueError('empty system log')
        bytime=defaultdict(list)
        for r in obs:
            bytime[round(float(r['t_s']),3)].append(r)
        logs=sorted(logs,key=lambda x:float(x['t_s']))
        if len(set(round(float(r['t_s']),3) for r in logs)) != len(logs):
            raise ValueError('duplicate system-log timestamps')
        mod=InactivityModule(window=args.window,interval=args.interval,
                             activity_threshold_px=(None if args.diagnostics_only
                                                    else args.activity_threshold_px),
                             min_fill=args.min_fill,max_gap_s=args.max_gap_s,
                             smoothing=args.smoothing,
                             activity_threshold_bl=(None if args.diagnostics_only
                                                    else args.activity_threshold_bl))
        args.out_dir.mkdir(parents=True,exist_ok=True)
        obs_out=args.out_dir/'smoothed_inactivity_observations.csv'
        log_out=args.out_dir/'smoothed_system_log.csv'
        summary_out=args.out_dir/'eligible_track_summary.csv'
        sweep_out=args.out_dir/'candidate_threshold_sweep.csv'
        if obs_out.resolve()==args.observations.resolve() or log_out.resolve()==args.log.resolve():
            raise ValueError('output must not overwrite an input')
        extra=['inactivity_method','smoothing_method','activity_threshold_px',
               'cumulative_activity_levels_by_id_px']
        out_fields=list(dict.fromkeys([*log_fields,*extra]))
        eligible=defaultdict(list)
        first_alert=None
        with obs_out.open('w',encoding='utf-8-sig',newline='') as fo,\
             log_out.open('w',encoding='utf-8-sig',newline='') as fl:
            wo=csv.writer(fo)
            wo.writerow(['t_s','session',*AUDIT_FIELDS])
            wl=csv.DictWriter(fl,fieldnames=out_fields)
            wl.writeheader()
            for line in logs:
                t=float(line['t_s']); key=round(t,3)
                rows=bytime.pop(key,[])
                ids=[int(r['track_id']) for r in rows]
                boxes=[[float(r['cx']),float(r['cy']),1.,1.] for r in rows]
                length=float(line['body_len_px']) if line.get('body_len_px') else None
                result=mod.observe(t,ids,boxes,audit=True,body_len_override=length)
                for rec in result.audit_rows:
                    wo.writerow([f'{t:.3f}',0,*rec])
                    data=dict(zip(AUDIT_FIELDS,rec))
                    if data['eligible']==1 or data['eligible']=='1':
                        eligible[int(data['track_id'])].append((t,float(data['cumulative_activity_px'])))
                alerts=sorted(result.inactive)
                row=dict(line)
                row.update({
                    'n_inactive':('' if args.diagnostics_only else len(alerts)),
                    'inactive_ids':('' if args.diagnostics_only else '|'.join(map(str,alerts))),
                    'inactivity_method':'optionA_cumulative_euclidean',
                    'smoothing_method':args.smoothing,
                    'activity_threshold_px':f'{args.activity_threshold_px:g}',
                    'cumulative_activity_levels_by_id_px':'|'.join(
                        f'{tid}:{score:.3f}'
                        for tid,score in sorted(result.eligible_activity_by_id_px.items())
                    ),
                })
                wl.writerow(row)
                if first_alert is None and alerts:
                    first_alert=(t,alerts)
        if bytime:
            raise ValueError(f'{sum(len(v) for v in bytime.values())} observations have timestamps not in log')
        with summary_out.open('w',newline='',encoding='utf-8-sig') as fh:
            w=csv.writer(fh)
            w.writerow(['track_id','eligible_samples','first_eligible_t_s','last_eligible_t_s',
                        'min_activity_px','median_activity_px','max_activity_px',
                        'alerts_at_selected_threshold'])
            for tid,values in sorted(eligible.items()):
                a=[v for _,v in values]
                w.writerow([tid,len(values),f'{values[0][0]:.3f}',f'{values[-1][0]:.3f}',
                            f'{min(a):.3f}',f'{median(a):.3f}',f'{max(a):.3f}',
                            sum(v<args.activity_threshold_px for v in a)])
        with sweep_out.open('w',newline='',encoding='utf-8-sig') as fh:
            w=csv.writer(fh)
            w.writerow(['candidate_threshold_px','track_id','eligible_samples','would_flag_samples',
                        'fraction_of_eligible','note'])
            for threshold in candidates:
                for tid,values in sorted(eligible.items()):
                    n=sum(v<threshold for _,v in values)
                    w.writerow([f'{threshold:g}',tid,len(values),n,
                                f'{n/len(values):.3f}',
                                'DESCRIPTIVE ONLY; no independent validation label'])
        print(f'Processed: {len(logs)} sampled timestamps, {len(obs)} centroid observations')
        print(f'Mode: {args.smoothing}; first alert: {first_alert if first_alert else "none"}')
        print(f'Eligible IDs: {sorted(eligible)}')
        for tid in (15,16):
            if tid in eligible:
                print(f'ID {tid}, first eligible: {eligible[tid][0][1]:.3f} px '
                      f'at {eligible[tid][0][0]:.3f}s')
        for path in (obs_out,log_out,summary_out,sweep_out):
            print('Wrote:',path.resolve())
    except (OSError,ValueError,KeyError) as e:
        p.exit(2,f'ERROR: {e}\n')


if __name__=='__main__':
    main()
