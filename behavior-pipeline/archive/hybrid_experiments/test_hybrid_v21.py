"""Synthetic tests for new association logic and V2 recovery compatibility.

Passing tests does not demonstrate accuracy on a real broiler tracking video.
"""
import unittest

from hybrid_tracker_v2 import RecoveryHybrid
from hybrid_tracker_v21 import ConflictAwareIDManager, RecoveryHybridV21


class V21Tests(unittest.TestCase):
    def test_baseline_mode_is_v2_exact_on_sequence(self):
        a = RecoveryHybrid()
        b = RecoveryHybridV21(identity_mode="baseline")
        events = [
            ([12, 13], [[10, 10, 20, 20], [60, 10, 20, 20]], [.9, .9],
             [[10, 10, 20, 20], [60, 10, 20, 20]], [.9, .9]),
            ([12], [[11, 10, 20, 20]], [.9],
             [[11, 10, 20, 20], [61, 10, 20, 20]], [.95, .9]),
            ([], [], [], [[12, 10, 20, 20]], [.91]),
            ([12], [[15, 10, 20, 20]], [.9],
             [[15, 10, 20, 20]], [.9]),
        ]
        for event in events:
            self.assertEqual(a.update(*event), b.update(*event))

    def test_global_assignment_avoids_greedy_conflict(self):
        # A greedy nearest-first assignment can assign T2->D1 first and force
        # T1->D2, even though the global T1->D1, T2->D2 costs less.
        tr = ConflictAwareIDManager(byte_bonus=0, same_id_gate=0.5)
        tr.update([101, 202], [[0, 10, 20, 20], [4, 10, 20, 20]], [.9, .9])
        ids, boxes, _ = tr.update([303, 404], [[2.5, 10, 20, 20], [6, 10, 20, 20]], [.9, .9])
        self.assertEqual(ids, [1, 2])
        self.assertEqual(len(boxes), 2)

    def test_plausible_bytetrack_id_beats_small_centroid_advantage(self):
        tr = ConflictAwareIDManager(byte_bonus=0.35)
        tr.update([101, 202], [[20, 10, 20, 20], [30, 10, 20, 20]], [.9, .9])
        ids, _, _ = tr.update([101, 202], [[28, 10, 20, 20], [22, 10, 20, 20]], [.9, .9])
        self.assertEqual(ids, [1, 2])

    def test_bytetrack_ids_not_blindly_trusted_after_large_jump(self):
        tr = ConflictAwareIDManager(byte_bonus=.35, same_id_gate=1.25)
        tr.update([101, 202], [[20, 10, 20, 20], [130, 10, 20, 20]], [.9, .9])
        ids, _, _ = tr.update([101, 202], [[130, 10, 20, 20], [20, 10, 20, 20]], [.9, .9])
        self.assertEqual(ids, [2, 1])

    def test_bytetrack_id_change_can_keep_spatial_identity(self):
        tr = ConflictAwareIDManager()
        self.assertEqual(tr.update([99], [[20, 10, 20, 20]], [.9])[0], [1])
        self.assertEqual(tr.update([188], [[21, 10, 20, 20]], [.9])[0], [1])

    def test_all_bytetrack_outputs_kept_one_to_one(self):
        tr = RecoveryHybridV21()
        ids, boxes, scores, recovered = tr.update(
            [1, 2], [[10, 10, 20, 20], [70, 10, 20, 20]], [.91, .93],
            [[10, 10, 20, 20], [70, 10, 20, 20]], [.91, .93])
        self.assertEqual(ids, [1, 2])
        self.assertEqual(len(boxes), 2)
        self.assertEqual(scores, [.91, .93])
        self.assertEqual(recovered, 0)

    def test_v2_recovery_kept_unchanged(self):
        tr = RecoveryHybridV21()
        tr.update([77], [[10, 10, 20, 20]], [.9],
                  [[10, 10, 20, 20]], [.9])
        ids, boxes, _, recovered = tr.update([], [], [],
                                              [[12, 11, 20, 20]], [.95])
        self.assertEqual((ids, recovered), ([1], 1))
        self.assertEqual(boxes, [[12.0, 11.0, 20.0, 20.0]])

    def test_raw_only_box_cannot_create_arbitrary_stable_id(self):
        tr = RecoveryHybridV21()
        self.assertEqual(tr.update([], [], [], [[10, 10, 20, 20]], [.99])[0], [])

    def test_missing_and_invalid_high_conf_boxes(self):
        tr = RecoveryHybridV21()
        tr.update([1], [[10, 10, 20, 20]], [.9], [[10, 10, 20, 20]], [.9])
        self.assertEqual(tr.update([], [], [], [[12, 10, 20, 20]], [.3])[3], 0)
        self.assertEqual(tr.update([], [], [], [[900, 10, 20, 20]], [.95])[3], 0)

    def test_ambiguous_input_duplicate_byte_id_is_rejected(self):
        tr = ConflictAwareIDManager()
        with self.assertRaises(ValueError):
            tr.update([10, 10], [[1, 1, 10, 10], [20, 20, 10, 10]], [.9, .9])

    def test_incompatible_box_size_does_not_claim_old_track(self):
        tr = ConflictAwareIDManager()
        tr.update([10], [[10, 10, 20, 20]], [.9])
        self.assertEqual(tr.update([10], [[11, 10, 60, 60]], [.9])[0], [2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
