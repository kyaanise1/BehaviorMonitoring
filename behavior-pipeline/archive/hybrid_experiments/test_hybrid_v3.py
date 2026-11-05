"""Synthetic unit tests only; no ground-truth performance guarantees."""
import unittest
from hybrid_tracker_v3 import MotionRecoveryHybrid

class V3Tests(unittest.TestCase):
    def test_recovers_dormant_track(self):
        x=MotionRecoveryHybrid()
        x.update([7],[[10,10,20,20]],[.9],[[10,10,20,20]],[.9])
        ids,boxes,scores,recovered=x.update([],[],[],[[12,10,20,20]],[.95])
        self.assertEqual((ids,recovered),([1],1))

    def test_no_double_use_of_detections(self):
        x=MotionRecoveryHybrid(ambiguity_margin=0)
        x.update([7,8],[[10,10,20,20],[70,10,20,20]],[.9,.9],
                 [[10,10,20,20],[70,10,20,20]],[.9,.9])
        ids, _, _, n=x.update([],[],[],[[12,10,20,20],[72,10,20,20]],[.9,.9])
        self.assertEqual(n,2)
        self.assertEqual(len(ids),len(set(ids)))

    def test_confidence_gate(self):
        x=MotionRecoveryHybrid()
        x.update([7],[[10,10,20,20]],[.9],[[10,10,20,20]],[.9])
        self.assertEqual(x.update([],[],[],[[11,10,20,20]],[.5])[3],0)

    def test_no_new_raw_only_track(self):
        x=MotionRecoveryHybrid()
        self.assertEqual(x.update([],[],[],[[11,10,20,20]],[.9])[0],[])

    def test_duplicate_filter(self):
        x=MotionRecoveryHybrid()
        ids, boxes, _, n=x.update([7],[[10,10,20,20]],[.9],
                                  [[10,10,20,20],[10.3,10.1,20,20]],[.9,.9])
        self.assertEqual((len(ids),len(boxes),n),(1,1,0))

    def test_ambiguous_match_rejected(self):
        x=MotionRecoveryHybrid(ambiguity_margin=0.2)
        x.update([7,8],[[10,10,20,20],[20,10,20,20]],[.9,.9],
                 [[10,10,20,20],[20,10,20,20]],[.9,.9])
        # Remove both ByteTrack outputs, offer one equidistant raw box.
        ids,boxes,_,n=x.update([],[],[],[[15,10,20,20]],[.9])
        self.assertEqual(n,0)

    def test_motion_favors_predicted_position(self):
        # Isolated moving bird: observed position advances by 8 per frame.
        x=MotionRecoveryHybrid(recovery_distance=0.65,motion_weight=1.0)
        x.update([7],[[10,10,20,20]],[.9],[[10,10,20,20]],[.9])
        x.update([7],[[18,10,20,20]],[.9],[[18,10,20,20]],[.9])
        # Last position 18, expected 26. Test displacement 16 (0.8 body lengths)
        # but motion-predicted position error 8 (0.4 body lengths).
        ids,_,_,n=x.update([],[],[],[[34,10,20,20]],[.95])
        self.assertEqual((ids,n),([1],1))

if __name__=='__main__':
    unittest.main(verbosity=2)
