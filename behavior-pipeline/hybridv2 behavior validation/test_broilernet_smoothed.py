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

    # ---- block30: activity in body lengths on 30-s block means (added 17 Nov 2026) ----
    @staticmethod
    def _jitter(i, amp):
        # deterministic detector-like jitter of +-amp px around a fixed point
        return (100 + amp * ((i * 7919) % 11 - 5) / 5, 200 + amp * ((i * 104729) % 13 - 6) / 6)

    def test_block30_still_bird_with_jitter_is_inactive(self):
        mod = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        r, row = run(mod, lambda i: self._jitter(i, 6))
        self.assertEqual(r.inactive, [15])
        self.assertLess(float(row['block_activity_bl']), 1.0)
        # the old per-picture sum of the same jitter would be far above 450 px
        self.assertGreater(float(row['raw_activity_px']), 450)

    def test_block30_walking_bird_is_active(self):
        mod = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        r, row = run(mod, lambda i: (100 + 0.5 * i, 200))   # 450 px = 5.6 BL over 30 min
        self.assertEqual(r.inactive, [])
        self.assertGreater(float(row['cumulative_activity_bl']), 4.0)

    def test_block30_same_answer_at_any_sampling(self):
        # A still bird with detector jitter, sampled every 2, 4 or 6 s. The old px sum changes several-fold
        # with sampling (one jitter step per picture); block30 changes far less, and the decision is the same.
        def scores(smoothing, **kw):
            out, flags = [], []
            for step in (2, 4, 6):
                mod = InactivityModule(interval=step, smoothing=smoothing, min_fill=0.8, max_gap_s=60, **kw)
                res = None
                for n in range(1800 // step + 1):
                    t = n * step
                    res = mod.observe(t, [15], [[*self._jitter(t // 2, 6), 80, 80]], audit=True)
                row = dict(zip(AUDIT_FIELDS, res.audit_rows[0]))
                out.append(float(row['cumulative_activity_px'])); flags.append(res.inactive)
            return out, flags
        old, _ = scores('median3', activity_threshold_px=450)
        new, flags = scores('block30', activity_threshold_bl=4.0)
        self.assertGreater(max(old) / min(old), 2.0)
        self.assertLess(max(new) / min(new), max(old) / min(old) / 1.5)
        self.assertEqual(flags, [[15], [15], [15]])
        self.assertLess(max(new) / 80, 4.0)

    def test_block30_scales_with_body_length(self):
        # the same walk is fewer body lengths for a bigger bird
        small = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        big = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        walk = lambda i: (100 + 0.4 * i, 200)   # 360 px
        _, rs = run(small, walk)
        r_big = None
        for i in range(901):
            r_big = big.observe(i * 2, [15], [[*walk(i), 160, 160]], audit=True)
        rb = dict(zip(AUDIT_FIELDS, r_big.audit_rows[0]))
        self.assertGreater(float(rs['cumulative_activity_bl']), 4.0)
        self.assertLess(float(rb['cumulative_activity_bl']), 4.0)

    def test_block30_allows_short_detector_gaps(self):
        mod = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        missing = {k for g in range(50, 900, 60) for k in range(g, g + 4)}   # an 8-10 s gap every 2 minutes
        r, row = run(mod, lambda i: (100, 200), missing=missing)
        self.assertEqual(int(row['eligible']), 1)
        self.assertEqual(r.inactive, [15])

    def test_range_bl_is_reported(self):
        mod = InactivityModule(smoothing='block30', activity_threshold_bl=4.0, min_fill=0.8, max_gap_s=60)
        _, row = run(mod, lambda i: (100 + (i % 2) * 20, 200))
        self.assertAlmostEqual(float(row['range_bl']), 0.25, places=2)

    def test_no_alert_before_30min(self):
        model=InactivityModule(smoothing='median3')
        r=None
        for i in range(900):
            r=model.observe(i*2,[15],[[0,0,80,80]],audit=False)
        self.assertEqual(r.inactive,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
