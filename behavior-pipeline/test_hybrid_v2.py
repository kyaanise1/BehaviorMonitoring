"""Synthetic unit tests; not a validation of performance on broiler videos."""
import unittest
from hybrid_tracker_v2 import RecoveryHybrid

class RecoveryTests(unittest.TestCase):
    def test_recovers_high_confidence_dormant_track(self):
        tr=RecoveryHybrid()
        a=tr.update([7],[[10,10,20,20]],[0.9],[[10,10,20,20]],[0.9])
        self.assertEqual(a[0],[1])
        b=tr.update([],[],[],[[12,11,20,20]],[0.95])
        self.assertEqual(b[0],[1])
        self.assertEqual(b[3],1)
    def test_keeps_byte_box_and_does_not_duplicate_it(self):
        tr=RecoveryHybrid()
        ids,boxes,scores,extra=tr.update([7],[[10,10,20,20]],[0.8],
                [[10,10,20,20],[10.2,9.9,20,20]],[0.9,0.95])
        self.assertEqual((ids,len(boxes),extra),([1],1,0))
        self.assertEqual(boxes,[[10.0,10.0,20.0,20.0]])
    def test_does_not_recover_unconfident_raw_box(self):
        tr=RecoveryHybrid()
        tr.update([7],[[10,10,20,20]],[.9],[[10,10,20,20]],[.9])
        self.assertEqual(tr.update([],[],[],[[11,10,20,20]],[.4])[3],0)
    def test_does_not_recover_teleported_box(self):
        tr=RecoveryHybrid()
        tr.update([7],[[10,10,20,20]],[.9],[[10,10,20,20]],[.9])
        self.assertEqual(tr.update([],[],[],[[500,10,20,20]],[.99])[3],0)
    def test_new_raw_only_box_does_not_create_track(self):
        tr=RecoveryHybrid()
        self.assertEqual(tr.update([],[],[],[[10,10,20,20]],[.99])[0],[])
    def test_one_to_one_recovery(self):
        tr=RecoveryHybrid()
        tr.update([7,8],[[10,10,20,20],[70,10,20,20]],[.9,.9],
                  [[10,10,20,20],[70,10,20,20]],[.9,.9])
        ids,boxes,scores,extra=tr.update([],[],[],[[12,10,20,20],[72,10,20,20]],[.9,.9])
        self.assertEqual(len(ids),2)
        self.assertEqual(len(set(ids)),2)
        self.assertEqual(extra,2)

if __name__=="__main__": unittest.main(verbosity=2)
