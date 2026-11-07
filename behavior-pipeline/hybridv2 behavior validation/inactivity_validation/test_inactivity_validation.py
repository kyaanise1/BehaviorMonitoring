"""Self-contained synthetic tests. These are unit tests, NOT measured thesis results."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
TOOL = BASE / "validate_inactivity_day6.py"


def write_csv(path, fieldnames, records):
    with path.open('w', newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader();w.writerows(records)


def run_tests():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        video = root/'video.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5.0,(320,240))
        assert writer.isOpened()
        for f in range(60):
            frame=np.zeros((240,320,3),dtype=np.uint8)
            cv2.circle(frame,(80+f//5,90),15,(200,200,200),-1)
            cv2.circle(frame,(230,135),15,(220,220,220),-1)
            writer.write(frame)
        writer.release()
        obs=[]
        beh=[]
        for t in (0,2,4,6,8,10):
            for tid,x,y,flag in [(1,80+t,90,0),(2,230,135,0)]:
                obs.append(dict(t_s=f'{t:.3f}',session='0',status='VALID',track_id=str(tid),cx=x,cy=y,
                                inactive=str(flag),movement_limit_px='40'))
            beh.append(dict(t_s=f'{t:.3f}',status='VALID',n_inactive='0',inactive_ids=''))
        op=root/'observations.csv';bp=root/'behavior.csv'
        write_csv(op,list(obs[0]),obs);write_csv(bp,list(beh[0]),beh)
        out=root/'review'
        subprocess.run([sys.executable,str(TOOL),'prepare','--observations',str(op),'--source',str(video),
            '--behavior-csv',str(bp),'--outdir',str(out),'--window','8','--interval','2',
            '--samples','2','--seed','42'],check=True,capture_output=True,text=True)
        assert (out/'review.html').exists()
        assert len(list((out/'images').glob('*.jpg')))==9
        manifest=list(csv.DictReader((out/'manifest.csv').open(encoding='utf-8-sig')))
        assert len(manifest)==2 and all(r['pred_inactive']=='0' for r in manifest),manifest
        assert all(abs(float(r['end_s'])-8)<1e-6 for r in manifest)
        # One negative and one manual positive, when system predicted zero -> TN=1, FN=1.
        anno=[dict(case_id=manifest[0]['case_id'],identity_visible='yes',recording_continuous='yes',movement='exceeded',tracker_stable='yes',notes=''),
              dict(case_id=manifest[1]['case_id'],identity_visible='yes',recording_continuous='yes',movement='stayed_within',tracker_stable='no',notes='identity switch but physical bird still visible')]
        ap=root/'annotations.csv';write_csv(ap,list(anno[0]),anno)
        scores=root/'scored'
        subprocess.run([sys.executable,str(TOOL),'score','--manifest',str(out/'manifest.csv'),'--annotations',str(ap),'--outdir',str(scores)],check=True,capture_output=True,text=True)
        m=json.loads((scores/'metrics.json').read_text())
        assert (m['tn'],m['fn'],m['tp'],m['fp'])==(1,1,0,0),m
        assert m['metrics']['precision'] is None and m['metrics']['recall']==0.0
        assert m['metrics']['f1_score']==0.0 and m['metrics']['accuracy']==0.5
        # All negative => F1/recall/precision undefined, accuracy 100% for inspected negatives.
        anno[1]['movement']='exceeded';write_csv(ap,list(anno[0]),anno)
        subprocess.run([sys.executable,str(TOOL),'score','--manifest',str(out/'manifest.csv'),'--annotations',str(ap),'--outdir',str(scores)],check=True,capture_output=True,text=True)
        m=json.loads((scores/'metrics.json').read_text())
        assert (m['tn'],m['fn'])==(2,0) and m['metrics']['f1_score'] is None and m['metrics']['recall'] is None
        # Invalid footage: do not count as a negative or positive.
        anno[1]['recording_continuous']='no';write_csv(ap,list(anno[0]),anno)
        subprocess.run([sys.executable,str(TOOL),'score','--manifest',str(out/'manifest.csv'),'--annotations',str(ap),'--outdir',str(scores)],check=True,capture_output=True,text=True)
        m=json.loads((scores/'metrics.json').read_text())
        assert m['scored']==1 and m['not_scored']==1 and m['reasons']['recording_not_continuous_or_uncertain']==1
        # Positive system outcome matched to the actual tracked ID at endpoint.
        beh[4]['n_inactive']='1';beh[4]['inactive_ids']='2';write_csv(bp,list(beh[0]),beh)
        positive_review=root/'positive_review'
        subprocess.run([sys.executable,str(TOOL),'prepare','--observations',str(op),'--source',str(video),
            '--behavior-csv',str(bp),'--outdir',str(positive_review),'--window','8','--interval','2','--samples','2'],
            check=True,capture_output=True,text=True)
        pos_manifest=list(csv.DictReader((positive_review/'manifest.csv').open(encoding='utf-8-sig')))
        assert sorted(x['pred_inactive'] for x in pos_manifest)==['0','1']
        # ID2 is system positive; manually not inactive -> FP, ID1 negative and moving -> TN.
        labeled=[dict(case_id=x['case_id'],identity_visible='yes',recording_continuous='yes',movement='exceeded',tracker_stable='yes',notes='') for x in pos_manifest]
        write_csv(ap,list(labeled[0]),labeled)
        subprocess.run([sys.executable,str(TOOL),'score','--manifest',str(positive_review/'manifest.csv'),
            '--annotations',str(ap),'--outdir',str(root/'pos_score')],check=True,capture_output=True,text=True)
        mm=json.loads((root/'pos_score'/'metrics.json').read_text())
        assert mm['fp']==1 and mm['tn']==1 and mm['metrics']['precision']==0.0
        # Changing that one manually confirmed case to inactive gives a TP.
        for x in labeled:
            mapped=next(y for y in pos_manifest if y['case_id']==x['case_id'])
            if mapped['pred_inactive']=='1':x['movement']='stayed_within'
        write_csv(ap,list(labeled[0]),labeled)
        subprocess.run([sys.executable,str(TOOL),'score','--manifest',str(positive_review/'manifest.csv'),
            '--annotations',str(ap),'--outdir',str(root/'pos_score')],check=True,capture_output=True,text=True)
        mm=json.loads((root/'pos_score'/'metrics.json').read_text())
        assert mm['tp']==1 and mm['tn']==1 and mm['metrics']['f1_score']==1.0
        # Non-valid status at t=4: reject preparation.
        beh[2]['status']='GAP';write_csv(bp,list(beh[0]),beh)
        x=subprocess.run([sys.executable,str(TOOL),'prepare','--observations',str(op),'--source',str(video),
            '--behavior-csv',str(bp),'--outdir',str(root/'invalid'),'--window','8','--interval','2','--samples','2'],
            capture_output=True,text=True)
        assert x.returncode!=0 and 'GAP/RECOVERING' in x.stderr,x.stderr
        print('PASS: prepare, images, TP/FP/FN/TN, metrics, undefined metrics, invalid footage rejection')


if __name__=='__main__':
    run_tests()
