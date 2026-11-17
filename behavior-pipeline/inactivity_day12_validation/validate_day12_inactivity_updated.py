#!/usr/bin/env python3
r"""Validate UPDATED Day 12 Hybrid V2 inactivity against verified CVAT video XML.

Python 3.9+ standard library only.  Neither the pipeline nor inputs are changed.

Inputs (same folder as this script, by default):
  day12_annotations_validations.xml : independently reviewed CVAT chicken tracks
  behavior_system_log.csv           : new Hybrid V2 sampled flock behavior log
  inactivity_observations.csv       : new Hybrid V2 per-system-ID observations

PowerShell:
    python .\validate_day12_inactivity_updated.py

Or explicit paths:
    python .\validate_day12_inactivity_updated.py `
      --xml .\day12_annotations_validations.xml `
      --log .\behavior_system_log.csv `
      --observations .\inactivity_observations.csv `
      --output-dir .\day12_updated_validation

What is evaluated:
  A. One initial 30-minute chicken-window per VERIFIED CVAT bird (at t=1800).
     The system decision is its first sampled row after 1800 seconds.  Every
     positive system ID MUST match a CVAT physical bird by trajectory.
     A GT-negative bird with no assigned alert is a TN at the *bird-alert*
     level, even if the detector failed to track it. This is NOT an IDF1 or
     tracking-recall evaluation. If any positive ID cannot be attributed,
     the bird-level confusion matrix is withheld, not silently set to 100%.
  B. Sampled-time flock-level binary alerts from 30:00 to 35:00, compared
     against sliding 30-min CVAT labels. These highly correlated samples
     are NOT independent detection events or birds.

Important: CVAT XML uses SOURCE FRAME numbers, NOT CVAT display indices.
Its FPS is not embedded: the default 11.0 FPS is inferred from the
35-minute clip and XML stop_frame 23100; check with ffprobe and set --fps
if different. The user independently verified the CVAT tracks.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
from statistics import median
import sys
import xml.etree.ElementTree as ET


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f'Cannot find input: {path}')
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
        return list(rd)


def save_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def parse_id_set(s: str) -> set[int]:
    # Inactivity IDs generally use '|'; accept CSV/semicolon delimiters too.
    if not s or not s.strip():
        return set()
    import re
    return {int(v) for v in re.split(r'[|;,\s]+', s.strip()) if v}


def parse_xml(path: Path):
    if not path.is_file():
        raise ValueError(f'Cannot find CVAT XML: {path}')
    root = ET.parse(path).getroot()
    job = root.find('meta/job')
    if job is None:
        raise ValueError('CVAT for video XML with meta/job is required.')
    fmt = job.findtext('frame_filter', '')
    if not fmt.startswith('step='):
        raise ValueError(f'Expected CVAT frame_filter step=N, got {fmt!r}')
    step = int(fmt.split('=', 1)[1]); stop = int(job.findtext('stop_frame'))
    if step < 1 or stop < 1:
        raise ValueError('Invalid CVAT frame step / stop frame.')
    tracks = {}
    for t in root.findall('track'):
        if t.get('label', '').lower() != 'broiler':
            continue
        tid = int(t.get('id'))
        boxes = {}
        for box in t.findall('box'):
            frame = int(box.get('frame'))
            x1,y1,x2,y2 = (float(box.get(k)) for k in ('xtl','ytl','xbr','ybr'))
            boxes[frame//step] = {
                'src_frame': frame, 'cx': (x1+x2)/2, 'cy': (y1+y2)/2,
                'length': max(x2-x1, y2-y1),
                'outside': box.get('outside')=='1',
                'occluded': box.get('occluded')=='1',
                'keyframe': box.get('keyframe')=='1',
            }
            if frame%step:
                raise ValueError(f'CVAT ID {tid} has frame {frame} not divisible by step {step}')
        if boxes:
            tracks[tid] = boxes
    if not tracks:
        raise ValueError('No CVAT tracks with label broiler.')
    return tracks, step, stop, int(job.findtext('size', '0'))


def gt_at(tracks, end_index: int, needed_steps: int) -> dict[int, dict]:
    """Independent verified-CVAT label for ONE 30-minute moving bird-window."""
    start = end_index-needed_steps
    result = {}
    for tid, by_index in tracks.items():
        if start < 0:
            result[tid] = dict(label=None, dx=None, dy=None, limit=None, n=0,
                               reason='Video has not reached 30 min')
            continue
        window = [by_index.get(i) for i in range(start, end_index+1)]
        if any(b is None or b['outside'] or b['occluded'] for b in window):
            result[tid] = dict(label=None, dx=None, dy=None, limit=None,
                               n=sum(b is not None and not b['outside'] and not b['occluded'] for b in window),
                               reason='Insufficient full-window visible CVAT coverage')
            continue
        xs = [b['cx'] for b in window]
        ys = [b['cy'] for b in window]
        limit = median(b['length'] for b in window)/2
        dx,dy = max(xs)-min(xs),max(ys)-min(ys)
        result[tid] = dict(label=int(dx<=limit and dy<=limit), dx=dx,dy=dy,
                           limit=limit,n=len(window),reason='Verified CVAT centroid rule')
    return result


def confusion_metrics(cm: Counter) -> dict:
    tp,tn,fp,fn = (cm[k] for k in ('TP','TN','FP','FN'))
    def ratio(a,b): return a/b if b else None
    p=ratio(tp,tp+fp); r=ratio(tp,tp+fn)
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,n=tp+tn+fp+fn,
                accuracy=ratio(tp+tn,tp+tn+fp+fn),
                precision=p, recall=r,
                f1=ratio(2*tp,2*tp+fp+fn),
                specificity=ratio(tn,tn+fp))


def display(v):
    return 'N/A' if v is None else f'{v*100:.2f}%'


def locate_cvat_track(system_id, observations, tracks, fps, step, time_offset,
                      window_s, max_match_median_px):
    """Track-to-track positional attribution; never match IDs by number alone.

    Each system observation is paired to its nearest CVAT sample (time error
    at most 0.6 * CVAT sample interval). Confidence comes from trajectory
    median error across the first 30 min and separation from other tracks.
    """
    samples = [(float(o['t_s']),float(o['cx']),float(o['cy'])) for o in observations
               if int(o['track_id'])==system_id and 0<=float(o['t_s'])<=window_s]
    if not samples:
        return None,0,None,None,'No system history in first window'
    diffs = []
    max_delta = (step/fps)*0.60
    for cvat_id, byidx in tracks.items():
        ds=[]
        for t,x,y in samples:
            idx = int(round((t-time_offset)*fps/step))
            box = byidx.get(idx)
            if box is None or box['outside'] or box['occluded']:
                continue
            gt_t = idx*step/fps+time_offset
            if abs(t-gt_t)>max_delta:
                continue
            ds.append(math.hypot(x-box['cx'],y-box['cy']))
        if len(ds) >= 20:
            diffs.append((median(ds), -len(ds), cvat_id, len(ds)))
    if not diffs:
        return None,0,None,None,'No sufficiently sampled CVAT positional comparison'
    diffs.sort()
    best, _, id_, n = diffs[0]
    second = diffs[1][0] if len(diffs)>1 else math.inf
    # Require 20+ temporal matches and unambiguous spatial alignment.
    if best > max_match_median_px or second - best < 20:
        return None,n,best,second,'Ambiguous or distant positional match'
    return id_,n,best,second,'Matched by CVAT/system trajectory; not numeric ID'


def post_threshold_continuity(system_id, cvat_id, observations, tracks, fps, step,
                              time_offset, threshold_s):
    """Check that the alerting system track still occupies its CVAT bird after 30m."""
    byindex=tracks[cvat_id]
    distances=[]
    for row in observations:
        if int(row['track_id']) != system_id:
            continue
        t=float(row['t_s'])
        if t < time_offset+threshold_s:
            continue
        idx=int(round((t-time_offset)*fps/step))
        b=byindex.get(idx)
        if b is None or b['outside'] or b['occluded']:
            continue
        if abs(t-(time_offset+idx*step/fps)) > 0.6*step/fps:
            continue
        distances.append(math.hypot(float(row['cx'])-b['cx'],
                                    float(row['cy'])-b['cy']))
    if not distances:
        return 0,None,None
    distances.sort()
    return len(distances), median(distances), distances[int(0.90*(len(distances)-1))]

def main():
    ap=argparse.ArgumentParser(description='Updated Day 12 30-minute inactivity validation (CVAT XML vs new Hybrid V2 CSVs)')
    ap.add_argument('--xml',type=Path,default=Path('day12_annotations_validations.xml'))
    ap.add_argument('--log',type=Path,default=Path('behavior_system_log.csv'))
    ap.add_argument('--observations',type=Path,default=Path('inactivity_observations.csv'))
    ap.add_argument('--output-dir',type=Path,default=Path('day12_updated_validation'))
    ap.add_argument('--fps',type=float,default=None,help='Actual FPS of CVAT-uploaded MP4; else infer from duration')
    ap.add_argument('--duration-s',type=float,default=2100.0,help='Duration of CVAT video in seconds (default: 35 min)')
    ap.add_argument('--window-s',type=float,default=1800.0,help='Inactivity window seconds (default: 1800)')
    ap.add_argument('--time-offset-s',type=float,default=0.0,help='CVAT clip start offset relative to system log')
    ap.add_argument('--match-max-median-px',type=float,default=45.0)
    ap.add_argument('--initial-alert-tolerance-s',type=float,default=5.0,
                    help='Max delay after 1800 s allowed for first system decision')
    args=ap.parse_args()
    if min(args.duration_s,args.window_s,args.match_max_median_px,args.initial_alert_tolerance_s)<=0:
        ap.error('FPS/window/duration/match threshold/initial tolerance must be positive.')
    try:
        tracks, step, stop, count = parse_xml(args.xml)
        fps = args.fps if args.fps is not None else stop / args.duration_s
        if fps<=0: raise ValueError('FPS must be positive')
        frame_interval=step/fps
        needed_steps=round(args.window_s/frame_interval)
        if abs(needed_steps*frame_interval-args.window_s)>frame_interval/2:
            raise ValueError('30-minute window cannot be represented within half a CVAT sample; check FPS')
        last_index=max(max(d) for d in tracks.values())
        if last_index < needed_steps:
            raise ValueError('CVAT annotations are shorter than the required inactivity window')

        logs=load_csv(args.log,{'t_s','n_inactive','inactive_ids'})
        observations=load_csv(args.observations,{'t_s','track_id','cx','cy','inactive','session'})
        if not logs or not observations:
            raise ValueError('System log or observations CSV is empty.')
        log_records=[]
        by_time=defaultdict(set)
        history_by_id=defaultdict(list)
        for row in observations:
            t=float(row['t_s']); sid=int(row['track_id'])
            history_by_id[sid].append(row)
            if int(row['inactive']): by_time[round(t,3)].add(sid)
        for row in logs:
            t=float(row['t_s']); ids=parse_id_set(row['inactive_ids'])
            n=int(row['n_inactive'])
            if len(ids)!=n:
                raise ValueError(f'Log at {t:.3f}s: n_inactive={n} but inactive_ids={ids}')
            actual=by_time.get(round(t,3),set())
            if ids!=actual:
                raise ValueError(f'Log vs observations disagree at {t:.3f}s: log={ids}, observations={actual}')
            log_records.append((t,ids,n))
        log_records.sort()
        if log_records[0][0]>args.time_offset_s+frame_interval:
            raise ValueError('System log starts after the CVAT initial window; check clip offset.')
        if log_records[-1][0] < args.time_offset_s+args.window_s:
            raise ValueError('System log ends before the 30-minute inactivity threshold.')

        # 1. Use the verified XML, not a system-generated label, for GT.
        initial_gt=gt_at(tracks,needed_steps,needed_steps)
        bird_rows=[]
        for cid,gt in sorted(initial_gt.items()):
            bird_rows.append(dict(cvat_id=cid,gt_inactive='' if gt['label'] is None else gt['label'],
                                  gt_eligible=int(gt['label'] is not None),
                                  covered_cvat_samples=gt['n'],
                                  centroid_range_x_px='' if gt['dx'] is None else round(gt['dx'],3),
                                  centroid_range_y_px='' if gt['dy'] is None else round(gt['dy'],3),
                                  half_body_length_px='' if gt['limit'] is None else round(gt['limit'],3),
                                  reason=gt['reason']))
        known_ids={cid:gt['label'] for cid,gt in initial_gt.items() if gt['label'] is not None}
        early_alerts=[(t,ids) for t,ids,n in log_records if t<args.time_offset_s+args.window_s and ids]
        first_system=next((e for e in log_records if e[0]>=args.time_offset_s+args.window_s),None)
        if first_system is None: raise ValueError('No system decision after the inactivity threshold.')
        decision_t,first_positive_ids,_=first_system
        if decision_t > args.time_offset_s+args.window_s+args.initial_alert_tolerance_s:
            raise ValueError('No system sample sufficiently near the first 30-minute decision.')
        # Map all predicted-positive IDs at the decision; unknown IDs block bird metrics.
        matched={};mapping_rows=[];unmatched=[]
        all_ids_alerted=sorted(set.union(set(),*(ids for t,ids,n in log_records if t>=args.time_offset_s+args.window_s)))
        for sid in all_ids_alerted:
            cvat,n,error,runnerup,reason=locate_cvat_track(
                sid,history_by_id[sid],tracks,fps,step,args.time_offset_s,args.window_s+args.time_offset_s,args.match_max_median_px)
            cont_n,cont_median,cont_p90=(0,None,None)
            if cvat is not None:
                cont_n,cont_median,cont_p90=post_threshold_continuity(
                    sid,cvat,history_by_id[sid],tracks,fps,step,args.time_offset_s,args.window_s)
                if cont_n>=20 and cont_median>args.match_max_median_px:
                    reason+='; post-threshold identity drift: UNSAFE to attribute alerts'
                    cvat=None
            mapping_rows.append(dict(system_id=sid,cvat_id='' if cvat is None else cvat,
                                     paired_observations=n,median_center_distance_px='' if error is None else round(error,3),
                                     second_best_median_distance_px='' if runnerup is None or math.isinf(runnerup) else round(runnerup,3),
                                     post30_samples=cont_n,
                                     post30_median_distance_px='' if cont_median is None else round(cont_median,3),
                                     post30_p90_distance_px='' if cont_p90 is None else round(cont_p90,3),
                                     mapping_result=reason,
                                     at_first_threshold=int(sid in first_positive_ids)))
            if cvat is None:
                unmatched.append(sid)
            elif cvat in matched.values():
                unmatched.append(sid)  # an ambiguous duplicate physical assignment
            else:
                matched[sid]=cvat
        if unmatched:
            all_mapped=False
        else:
            all_mapped=all(cvat in initial_gt and initial_gt[cvat]['label'] is not None
                           for cvat in matched.values())
        # Also prohibit unexpected alerting IDs that cannot be assigned to known GT.
        # One bird-window decision at the FIRST system sample after the threshold.
        per_bird=[]; cm_bird=Counter()
        for cid,gt in sorted(initial_gt.items()):
            pred=int(cid in {matched.get(sid) for sid in first_positive_ids}) if all_mapped else None
            label=gt['label']
            if label is None: status='EXCLUDED_GT_COVERAGE'
            elif not all_mapped: status='UNRESOLVED_ALERT_ATTRIBUTION'
            else:
                status='TP' if label==1 and pred==1 else 'FN' if label==1 else 'FP' if pred==1 else 'TN'
                cm_bird[status]+=1
            per_bird.append(dict(cvat_id=cid,ground_truth_inactive='' if label is None else label,
                                system_pred_inactive='' if pred is None or label is None else pred,
                                result=status,system_ids_alerting_this_bird='|'.join(str(sid) for sid,c in matched.items() if c==cid),
                                note='One verified CVAT bird-window; missing detector tracks are not tracking recall.'))

        # 2. Sampled-frame flock-level GT for each system sample after 1800 s.
        # This is an additional diagnostic (many correlated samples, NOT trials).
        timeline=[];cm_flock=Counter()
        last_gt_by_idx={}
        for t,ids,n in log_records:
            if t < args.time_offset_s+args.window_s-0.01 or t > args.time_offset_s+args.duration_s+0.01:
                continue
            gt_idx=int(round((t-args.time_offset_s)/frame_interval))
            if gt_idx not in last_gt_by_idx:
                last_gt_by_idx[gt_idx]=gt_at(tracks,gt_idx,needed_steps)
            states=last_gt_by_idx[gt_idx]
            pos_ids={cid for cid,g in states.items() if g['label']==1}
            unknown_ids={cid for cid,g in states.items() if g['label'] is None}
            if pos_ids:
                actual=1
            elif unknown_ids:
                actual=None  # cannot certify flock negative if some birds are unobserved
            else:
                actual=0
            predicted=int(bool(ids))
            if actual is None or not all_mapped:
                status='UNSCORED_GT_UNKNOWN' if actual is None else 'UNRESOLVED_ALERT_ATTRIBUTION'
            else:
                status=('TP' if predicted else 'FN') if actual==1 else ('FP' if predicted else 'TN')
                cm_flock[status]+=1
            timeline.append(dict(system_time_s=round(t,3),closest_cvat_sample_index=gt_idx,
                                 ground_truth_any_inactive='' if actual is None else actual,
                                 cvat_positive_ids='|'.join(map(str,sorted(pos_ids))),
                                 system_n_inactive=n,system_inactive_ids='|'.join(map(str,sorted(ids))),
                                 system_any_inactive=predicted,result=status))

        args.output_dir.mkdir(parents=True,exist_ok=True)
        save_csv(args.output_dir/'day12_gt_first30_birds.csv',
                 ['cvat_id','gt_inactive','gt_eligible','covered_cvat_samples','centroid_range_x_px',
                  'centroid_range_y_px','half_body_length_px','reason'],bird_rows)
        save_csv(args.output_dir/'day12_system_to_cvat_matches.csv',
                 ['system_id','cvat_id','paired_observations','median_center_distance_px',
                  'second_best_median_distance_px','post30_samples','post30_median_distance_px',
                  'post30_p90_distance_px','mapping_result','at_first_threshold'],mapping_rows)
        save_csv(args.output_dir/'day12_bird_window_decisions.csv',
                 ['cvat_id','ground_truth_inactive','system_pred_inactive','result',
                  'system_ids_alerting_this_bird','note'],per_bird)
        save_csv(args.output_dir/'day12_flock_timeline.csv',
                 ['system_time_s','closest_cvat_sample_index','ground_truth_any_inactive',
                  'cvat_positive_ids','system_n_inactive','system_inactive_ids','system_any_inactive','result'],timeline)
        metrics_bird=confusion_metrics(cm_bird) if all_mapped else None
        metrics_flock=confusion_metrics(cm_flock) if all_mapped else None
        metrics=[]
        for scope,vals,unit in [
            ('INITIAL_30MIN_BIRD_ALERT_DECISION',metrics_bird,'One independently labeled physical chicken, first 30-min window'),
            ('SLIDING_30MIN_FLOCK_SAMPLE',metrics_flock,'Correlated system samples over 30–35 min (not independent events)')]:
            if vals is not None:
                metrics.append({'scope':scope,'unit':unit,**{k:round(v,6) if isinstance(v,float) else ('' if v is None else v) for k,v in vals.items()}})
            else:
                metrics.append({'scope':scope,'unit':'UNSCORED: unmatched/ambiguous positive alert; '+unit})
        save_csv(args.output_dir/'day12_metrics.csv',
                 ['scope','unit','TP','TN','FP','FN','n','accuracy','precision','recall','f1','specificity'],metrics)
        n_gtpositive=sum(x==1 for x in known_ids.values())
        n_gtnegative=sum(x==0 for x in known_ids.values())
        alerts_late=[(t,ids) for t,ids,n in log_records if t>=args.time_offset_s+args.window_s]
        first_alert=next(((t,ids) for t,ids,n in log_records if ids),None)
        last_alert=next(((t,ids) for t,ids,n in reversed(log_records) if ids),None)
        report=[
            'DAY 12: UPDATED HYBRID V2 INACTIVITY VALIDATION',
            '='*52,
            'Independent ground truth: user-verified CVAT video XML (physical chicken tracks).',
            f'CVAT tracks: {len(tracks)}; sampled frames: {count}; frame step: {step}; FPS: {fps:.6f}',
            f'FPS method: {"supplied --fps" if args.fps is not None else "inferred from CVAT stop_frame / 35-minute duration (verify with ffprobe)"}',
            f'30-min window spans {needed_steps+1} CVAT samples, covering {needed_steps*frame_interval:.3f} s',
            f'GT initial 30-min: {n_gtpositive} positive / {n_gtnegative} negative / {len(tracks)-len(known_ids)} excluded',
            f'System log rows: {len(log_records)}; system observation rows: {len(observations)}',
            f'System log coverage: {log_records[0][0]:.3f}–{log_records[-1][0]:.3f}s',
            f'System first sample after 30min: {decision_t:.3f}s; predicted active alert IDs: {sorted(first_positive_ids)}',
            f'First alert: {first_alert[0]:.3f}s ID(s) {sorted(first_alert[1])}' if first_alert else 'First alert: none',
            f'Last alert: {last_alert[0]:.3f}s ID(s) {sorted(last_alert[1])}' if last_alert else 'Last alert: none',
            f'Positive sampled log rows from 30–35min: {sum(bool(ids) for t,ids in alerts_late)} / {len(alerts_late)}',
            f'All system positive IDs across 30–35min: {all_ids_alerted}',
            f'Unexpected PRE-30min alert samples: {len(early_alerts)}',
            f'CVAT attribution of positive system IDs: {matched}',
            f'Unmatched/ambiguous positive IDs: {unmatched}',
            '',
        ]
        if early_alerts:
            report += ['WARNING: system flagged inactivity before 30min; inspect rule and timestamps.', '']
        if not all_mapped:
            report += ['WARNING: Bird-level and flock-level metrics are UNSCORED because positive system IDs',
                       'cannot all be attributed to independently verified CVAT birds.', '']
        for header,data in [('INITIAL 30-MIN BIRD-WINDOW DECISIONS',metrics_bird),
                            ('30–35 MIN FLOCK-LEVEL SAMPLE AGREEMENT (CORRELATED)',metrics_flock)]:
            report.append(header)
            if data is None:
                report.append('UNSCORED')
            else:
                report.append(f'CM: TP={data["TP"]}, TN={data["TN"]}, FP={data["FP"]}, FN={data["FN"]}; n={data["n"]}')
                report += [f'{k.capitalize()}: {display(data[k])}' for k in ('accuracy','precision','recall','f1','specificity')]
            report.append('')
        report += [
            'METHODOLOGY / IMPORTANT LIMITATIONS',
            '- Binary per-bird scoring uses the FIRST 30-minute window only, not every frame as a fresh trial.',
            '- Full-window GT labels are computed from the independently verified CVAT boxes (including reviewed interpolation).',
            '- Half-body-length threshold is applied to each coordinate range independently.',
            '- CVAT and system IDs are NEVER assumed to be numerically identical; positive IDs need geometric matching.',
            '- TN means the system issued no inactivity alert for that GT-negative bird at the first 30-min decision;',
            '  it does not mean its tracker ID was accurate, persistent, or even visible.',
            '- The 30–35-minute flock timeline contains correlated 2-second observations and cannot be interpreted',
            '  as hundreds of independent successes. Its specificity is N/A if all GT samples are positive.',
            '- One positive physical bird cannot establish generalization to multiple inactivity events.',
            '- Check the actual CVAT video FPS and verify that the CVAT trim and system log start at the same moment.',
        ]
        text='\n'.join(report)+'\n'
        (args.output_dir/'day12_validation_report.txt').write_text(text,encoding='utf-8')
        print(text)
        print('Results written to:',args.output_dir.resolve())
    except (ValueError, OSError, ET.ParseError) as ex:
        ap.exit(2,f'ERROR: {ex}\n')


if __name__=='__main__':
    main()
