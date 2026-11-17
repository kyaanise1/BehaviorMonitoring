"""Unit tests: Option A smoothing, eligibility, gaps, and threshold semantics."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from behavior_inactivity_broilernet_smoothed import InactivityModule,AUDIT_FIELDS


def run(model,position,missing=()):
    result=None
    for i in range(901):
        if i in missing:
            # Observe empty frame to keep global clock moving.
            result=model.observe(i*2,[],[],audit=True)
            continue
        x,y=position(i)
        result=model.observe(i*2,[15],[[x,y,80,80]],audit=True)
    return result,dict(zip(AUDIT_FIELDS,result.audit_rows[0]))


class Tests(unittest.TestCase):
    def test_same_static_activity_all_methods(self):
        for smoothing in ('raw','median3','median5'):
            model=InactivityModule(smoothing=smoothing)
            r,row=run(model,lambda i:(100,200))
            self.assertEqual(r.inactive,[15]);self.assertEqual(float(row['cumulative_activity_px']),0)
            self.assertEqual(int(row['eligible']),1)

    def test_raw_matches_original_formula(self):
        m=InactivityModule(smoothing='raw',activity_threshold_px=400)
        _,row=run(m,lambda i:(i*.5,0))
        self.assertAlmostEqual(float(row['cumulative_activity_px']),450,places=2)
        self.assertEqual(int(row['inactive']),0)

    def test_alternating_jitter_reduced_with_median3(self):
        raw,_=run(InactivityModule(smoothing='raw',activity_threshold_px=60),
                  lambda i:(100+(i%10==0),200))
        smooth,row=run(InactivityModule(smoothing='median3',activity_threshold_px=60),
                       lambda i:(100+(i%10==0),200))
        self.assertEqual(raw.inactive,[])
        self.assertEqual(smooth.inactive,[15])
        self.assertLess(float(row['cumulative_activity_px']),float(row['raw_activity_px']))

    def test_big_movement_remains_visible(self):
        mod=InactivityModule(smoothing='median3',activity_threshold_px=360)
        _,row=run(mod,lambda i:(0 if i<450 else 1000,0))
        self.assertGreater(float(row['cumulative_activity_px']),360)
        self.assertEqual(int(row['inactive']),0)

    def test_missing_long_gap_is_not_stillness(self):
        mod=InactivityModule(smoothing='median3',activity_threshold_px=360)
        _,row=run(mod,lambda i:(1,1),missing=set(range(400,440)))
        self.assertEqual(int(row['eligible']),0)
        self.assertEqual(int(row['max_gap_ok']),0)

    def test_threshold_diagnostics_only(self):
        mod=InactivityModule(smoothing='median3',activity_threshold_px=None)
        _,row=run(mod,lambda i:(0,0))
        self.assertEqual(row['inactive'],'')
        self.assertEqual(row['activity_ok'],'')

    def test_no_alert_before_30min(self):
        model=InactivityModule(smoothing='median3')
        r=None
        for i in range(900):
            r=model.observe(i*2,[15],[[0,0,80,80]],audit=False)
        self.assertEqual(r.inactive,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
