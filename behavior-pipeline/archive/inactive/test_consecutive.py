"""Unit tests for Option B (run from behavior-pipeline root)."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from behavior_inactivity_consecutive import InactivityModule, AUDIT_FIELDS


def feed(mod, sec, x=100, y=100, bird_id=15, size=80):
    r=mod.observe(sec,[bird_id],[[x,y,size,size]],audit=True)
    return r,dict(zip(AUDIT_FIELDS,r.audit_rows[0]))


class ConsecutiveTests(unittest.TestCase):
    def test_markers_strict_5_15_30(self):
        m=InactivityModule()
        seen={}
        for i in range(902):
            r,d=feed(m,i*2)
            if i*2 in (300,302,900,902,1800,1802):
                seen[i*2]=r.markers_by_id[15]
        self.assertEqual(seen, {300:'NONE',302:'YELLOW',900:'YELLOW',902:'ORANGE',1800:'ORANGE',1802:'RED'})
        self.assertEqual(r.inactive,[15])
        self.assertEqual(float(d['stillness_duration_s']),1802)
    def test_large_position_change_resets_timer(self):
        m=InactivityModule()
        for i in range(180):
            feed(m,i*2)
        feed(m,360,180)
        r,d=feed(m,362,180)
        self.assertEqual(d['reset_reason'],'significant_movement')
        self.assertEqual(float(d['stillness_duration_s']),0)
        self.assertEqual(r.markers_by_id[15],'NONE')
    def test_slow_drift_cannot_accumulate_30min(self):
        m=InactivityModule()
        resets=0
        for i in range(902):
            r,d=feed(m,i*2,x=100+i*0.2)
            if d['reset_reason']=='significant_movement':resets+=1
        self.assertGreater(resets,0)
        self.assertNotIn(15,r.inactive)
    def test_gap_resets(self):
        m=InactivityModule()
        for i in range(61):feed(m,i*2)
        r,d=feed(m,138)
        self.assertEqual(d['reset_reason'],'observation_gap')
        self.assertEqual(float(d['stillness_duration_s']),0)
    def test_low_coverage_is_undetermined(self):
        m=InactivityModule()
        for i in range(200):
            if i%5==4:
                m.observe(i*2,[],[])
            else:
                r,d=feed(m,i*2)
        self.assertEqual(r.markers_by_id[15],'UNDETERMINED')
        self.assertLess(float(d['streak_fill_ratio']),.95)
    def test_body_length_specific_per_id(self):
        m=InactivityModule()
        for t in (0,2,4,6):
            r=m.observe(t,[1,2],[[100,100,50,50],[100,200,100,100]],audit=True)
        for t in (8,10,12):
            r=m.observe(t,[1,2],[[130,100,50,50],[130,200,100,100]],audit=True)
        rows={int(row[0]):dict(zip(AUDIT_FIELDS,row)) for row in r.audit_rows}
        self.assertEqual(rows[1]['reset_reason'],'')
        self.assertGreater(int(rows[1]['reset_count']),0)
        self.assertEqual(int(rows[2]['reset_count']),0)
        self.assertEqual(float(rows[1]['movement_tolerance_px']),25.0)
        self.assertEqual(float(rows[2]['movement_tolerance_px']),50.0)
    def test_probable_id_switch_resets_and_flags(self):
        m=InactivityModule()
        for i in range(100):feed(m,i*2)
        r,d=feed(m,200,400)
        self.assertEqual(d['reset_reason'],'possible_id_jump')
        self.assertEqual(r.markers_by_id[15],'UNDETERMINED')
        self.assertEqual(int(d['possible_id_jump']),1)
    def test_returns_no_id_when_absent(self):
        m=InactivityModule()
        feed(m,0)
        r=m.observe(2,[],[],audit=True)
        self.assertEqual(r.counts_by_marker['RED'],0)
        self.assertFalse(r.inactive)
        self.assertFalse(r.markers_by_id)
    def test_invalid_boxes(self):
        m=InactivityModule()
        with self.assertRaises(ValueError):m.observe(0,[1],[[1,2,0,3]])

if __name__=='__main__':unittest.main(verbosity=2)
