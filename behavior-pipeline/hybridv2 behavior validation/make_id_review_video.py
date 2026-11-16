#!/usr/bin/env python3
"""Overlay an EXISTING system track ID on its original video.

Uses inactivity observation CSVs from Hybrid V2 or Broiler-Net replay.
No model inference, retracking, or track-ID reassignment.

Examples (PowerShell, executed from behavior-pipeline):
    python ".\\hybridv2 behavior validation\\make_id_review_video.py" `
      --source "..\\video\\Day12_repaired.mp4" `
      --observations ".\\inactivity_observations.csv" `
      --id 15 --start 28:00 --end 35:00 `
      --output ".\\results\\ID15_review_28to35.mp4"

    # Focus on the reported 33:30-33:51 potential ID switch:
    python ".\\hybridv2 behavior validation\\make_id_review_video.py" `
      --source "..\\video\\Day12_repaired.mp4" `
      --observations ".\\inactivity_observations.csv" `
      --id 15 --start 33:20 --end 34:05 `
      --output ".\\results\\ID15_switch_review.mp4"

The CSV contains sampled CENTROIDS, NOT bounding box dimensions; the overlay
is a circle and trajectory, not a true detected bounding box. Realtime mode
retains normal playback speed but only updates markers at available CSV times.
The overlay does NOT establish physical chicken identity if IDs switch.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import math
from pathlib import Path

import cv2


def parse_time(s: str) -> float:
    p = s.strip().split(':')
    if not 1 <= len(p) <= 3:
        raise argparse.ArgumentTypeError('Time must be seconds, MM:SS, or HH:MM:SS')
    try:
        values = [float(v) for v in p]
    except ValueError as e:
        raise argparse.ArgumentTypeError('Invalid time: '+s) from e
    if any(v<0 for v in values):
        raise argparse.ArgumentTypeError('Time must be nonnegative')
    if len(values)==3: return values[0]*3600+values[1]*60+values[2]
    if len(values)==2: return values[0]*60+values[1]
    return values[0]


def clock(t:float)->str:
    t=max(0,t)
    h=int(t//3600);m=int(t%3600//60);s=t%60
    return f'{h:02d}:{m:02d}:{s:05.2f}'


def load_id_csv(path:Path, target:int):
    if not path.is_file():raise ValueError('CSV not found: '+str(path))
    rows=[]
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        needed={'t_s','track_id','cx','cy'}
        missing=needed.difference(reader.fieldnames or [])
        if missing:raise ValueError('CSV is missing columns: '+', '.join(sorted(missing)))
        for r in reader:
            if int(r['track_id'])!=target:continue
            rows.append((float(r['t_s']),float(r['cx']),float(r['cy']),r.get('inactive','')))
    rows.sort()
    if not rows:raise ValueError(f'No rows found for system track ID {target} in {path}')
    # Fail rather than silently mix different session/video segments.
    times=[r[0] for r in rows]
    if len(set(times))<len(times):
        raise ValueError('Duplicate timestamps for the target ID; CSV may contain multiple sessions. Use a single-session CSV.')
    return rows,times


def text_shadow(img,s,xy,scale=0.72,color=(255,255,255),thick=2):
    cv2.putText(img,s,(xy[0]+2,xy[1]+2),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+3,cv2.LINE_AA)
    cv2.putText(img,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)


def annotate(frame,t,row,rows,times,pos,args):
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,75),(20,20,20),-1)
    text_shadow(frame,f'SYSTEM TRACK ID {args.id} | video {clock(t)}',(18,30),0.75)
    text_shadow(frame,'CSV centroid is sampled: markers are NOT frame-by-frame tracking',(18,59),0.50,(190,230,255),1)
    cv2.rectangle(frame,(0,h-38),(w,h),(20,20,20),-1)
    if row is None:
        text_shadow(frame,'ID NOT OBSERVED within sample tolerance',(15,h-12),0.60,(120,150,255),1)
        return
    ts,cx,cy,inactive=row
    px,py=int(round(cx)),int(round(cy))
    sample_gap=abs(t-ts)
    # Recent sampled trajectory; very large jumps are shown, not smoothed away.
    tail=[]
    for j in range(max(0,pos-9),pos+1):
        if 0 <= j < len(rows):
            r=rows[j]
            if r[0] <= ts and ts-r[0]<=24:
                tail.append(r)
    for a,b in zip(tail,tail[1:]):
        axy=(int(round(a[1])),int(round(a[2])))
        bxy=(int(round(b[1])),int(round(b[2])))
        if math.hypot(b[1]-a[1],b[2]-a[2])>18:
            cv2.line(frame,axy,bxy,(30,60,240),2,cv2.LINE_AA)
        else:
            cv2.line(frame,axy,bxy,(30,205,240),2,cv2.LINE_AA)
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(frame,(px,py),23,(0,0,0),6,cv2.LINE_AA)
        cv2.circle(frame,(px,py),23,(0,240,50),3,cv2.LINE_AA)
        cv2.circle(frame,(px,py),6,(0,0,0),-1,cv2.LINE_AA)
        cv2.circle(frame,(px,py),4,(0,240,50),-1,cv2.LINE_AA)
        label_y=max(97,min(h-50,py-32))
        label_x=max(12,min(w-150,px+25))
        text_shadow(frame,f'ID {args.id}',(label_x,label_y),0.78,(0,255,70),2)
    flag=(f'inactive={inactive}' if inactive.strip() else 'inactive field unavailable')
    text_shadow(frame,f'CSV t={ts:.2f}s | off={sample_gap:.2f}s | ({cx:.1f},{cy:.1f}) | {flag}',
                (15,h-13),0.49,(200,245,255),1)
    # The previously flagged interval deserves special visual attention.
    if args.id==15 and 2010 <= t <= 2031:
        text_shadow(frame,'CHECK PHYSICAL CHICKEN: reported ID switch 33:30-33:51',
                    (18,95),0.57,(40,180,255),2)


def main():
    ap=argparse.ArgumentParser(description='Make review video highlighting a system ID from an existing inactivity observations CSV.')
    ap.add_argument('--source',type=Path,required=True,help='Exact original video used to produce the CSV')
    ap.add_argument('--observations',type=Path,required=True,help='Original or Broiler-Net replay per-ID CSV')
    ap.add_argument('--id',type=int,default=15,help='System track ID, not CVAT track ID (default 15)')
    ap.add_argument('--start',type=parse_time,default=1680,help='Start time, seconds or MM:SS (default 28:00)')
    ap.add_argument('--end',type=parse_time,default=2100,help='End time, seconds or MM:SS (default 35:00)')
    ap.add_argument('--output',type=Path,default=Path('ID15_review.mp4'))
    ap.add_argument('--mode',choices=('realtime','fast'),default='realtime',
                    help='realtime: video with sampled marker. fast: one snapshot per sampled CSV position')
    ap.add_argument('--tolerance-s',type=float,default=1.3,help='Maximum difference between frame and CSV sample for marker (default 1.3 s)')
    ap.add_argument('--fast-fps',type=float,default=10.0,help='Playback FPS of fast sampled-video mode (default 10)')
    args=ap.parse_args()
    if args.end<=args.start or args.tolerance_s<=0 or args.fast_fps<=0:ap.error('Invalid time range, tolerance or fast FPS')
    try:
        rows,times=load_id_csv(args.observations,args.id)
        if args.output.resolve() == args.source.resolve():raise ValueError('Output must be different from original input video')
        cap=cv2.VideoCapture(str(args.source))
        if not cap.isOpened():raise ValueError('Video cannot be opened: '+str(args.source))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if fps<=0:raise ValueError('Could not read video FPS')
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width<=0 or height<=0:raise ValueError('Cannot read video dimensions')
        if count>0 and args.start>count/fps+1:raise ValueError('Start time exceeds video duration; verify source matches CSV')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        out_fps = fps if args.mode=='realtime' else args.fast_fps
        writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),out_fps,(width,height))
        if not writer.isOpened():raise ValueError('Could not create MP4 output: '+str(args.output))
        n=0;n_visible=0
        def chosen(t):
            j=bisect.bisect_left(times,t)
            options=[k for k in (j-1,j) if 0 <= k < len(times)]
            if not options:return None,-1
            k=min(options,key=lambda x:abs(times[x]-t))
            return (rows[k],k) if abs(times[k]-t)<=args.tolerance_s else (None,-1)
        if args.mode=='realtime':
            # Work sequentially through the source footage, retaining real playback speed.
            start_frame=max(0,int(round(args.start*fps)))
            last_frame=int(round(args.end*fps))
            if count>0:last_frame=min(last_frame,count)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
            for i in range(start_frame,last_frame):
                ok,frame=cap.read()
                if not ok:break
                # POS_MSEC may be inaccurate or absent on some AVI/MKV files;
                # sequential frame time is stable for normal constant-frame-rate videos.
                t=i/fps
                row,k=chosen(t)
                annotate(frame,t,row,rows,times,k,args)
                writer.write(frame);n+=1;n_visible+=int(row is not None)
                if n%max(1,int(fps*60))==0:print(f'Processed {n} frames ({n/fps:.0f}s output)')
        else:
            for row in rows:
                t=row[0]
                if t<args.start or t>args.end:continue
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
                ok,frame=cap.read()
                if not ok:continue
                k=bisect.bisect_left(times,t)
                annotate(frame,t,row,rows,times,k,args)
                writer.write(frame);n+=1;n_visible+=1
        writer.release();cap.release()
        if not n:raise ValueError('No video frames written: check file, times and observations CSV')
        print(f'SAVED: {args.output.resolve()}')
        print(f'Frames written: {n}; marker shown on {n_visible}; playback duration ~{n/out_fps:.1f}s')
        print('IMPORTANT: The centroid marker uses logged sample positions, not continuous rerun tracking.')
    except (ValueError,OSError) as exc:
        ap.exit(2,'ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
