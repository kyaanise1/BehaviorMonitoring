"""Unit tests for the separate hybrid_tracker.py module; original tracker.py is untouched."""
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).with_name("hybrid_tracker.py")
spec = importlib.util.spec_from_file_location("hybrid_tracker", MODULE_PATH)
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)


class HybridTests(unittest.TestCase):
    def test_new_and_byte_id_switch_still_preserves_nearest_stable_ids(self):
        tr = tracker.HybridIDManager()
        ids, boxes, cs = tr.update([101, 102], [[20, 40, 20, 20], [120, 40, 20, 20]], [0.9, 0.9])
        self.assertEqual(ids, [1, 2])
        ids, _, _ = tr.update([102, 101], [[22, 40, 20, 20], [118, 40, 20, 20]], [0.9, 0.9])
        self.assertEqual(ids, [1, 2])

    def test_byte_id_matches_moving_bird_and_persists_gap(self):
        tr = tracker.HybridIDManager(tight_frac=0.5, wide_frac=4, max_missed=2)
        self.assertEqual(tr.update([7], [[20, 40, 20, 20]], [0.9])[0], [1])
        self.assertEqual(tr.update([7], [[48, 40, 20, 20]], [0.9])[0], [1])
        self.assertEqual(tr.update([], [], [])[0], [])
        self.assertEqual(tr.update([99], [[51, 42, 20, 20]], [0.6])[0], [1])

    def test_not_same_id_if_far_teleport(self):
        tr = tracker.HybridIDManager()
        self.assertEqual(tr.update([7], [[20, 40, 20, 20]], [0.9])[0], [1])
        self.assertEqual(tr.update([7], [[500, 500, 20, 20]], [0.9])[0], [2])

    def test_stale_expiry(self):
        tr = tracker.HybridIDManager(max_missed=1)
        tr.update([11], [[20, 40, 20, 20]], [0.9])
        tr.update([], [], [])
        tr.update([], [], [])
        self.assertEqual(tr.update([99], [[20, 40, 20, 20]], [0.9])[0], [2])

    def test_preserves_count_boxes_scores(self):
        tr = tracker.HybridIDManager()
        boxes = [[20, 30, 15, 20], [50, 60, 18, 22]]
        ids, bb, scores = tr.update([7, 8], boxes, [0.2, 0.7])
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(bb, boxes)
        self.assertEqual(scores, [0.2, 0.7])

    def test_size_gating(self):
        tr = tracker.HybridIDManager(max_size_ratio=2)
        self.assertEqual(tr.update([1], [[20, 30, 15, 20]], [0.9])[0], [1])
        self.assertEqual(tr.update([1], [[20, 30, 150, 200]], [0.9])[0], [2])

    def test_argparse_make_runner_standalone_and_hybrid(self):
        class Values:
            def __init__(self, vals): self.vals = vals
            def int(self): return self
            def tolist(self): return self.vals
        class Boxes:
            def __init__(self, ids, boxes, scores):
                self.id = Values(ids); self.xywh = Values(boxes); self.conf = Values(scores)
        class FakeModel:
            def __init__(self): self.i = 0
            def track(self, frame, **kwargs):
                self.i += 1
                if self.i == 1: return [SimpleNamespace(boxes=Boxes([10], [[10, 10, 20, 20]], [0.9]))]
                return [SimpleNamespace(boxes=Boxes([99], [[12, 11, 20, 20]], [0.9]))]
        args = SimpleNamespace(tracker_type="hybrid", conf=0.1, imgsz=640, tracker="fake.yaml")
        fn = tracker.make_runner(args, FakeModel())
        self.assertEqual(fn(object())[0], [1])
        self.assertEqual(fn(object())[0], [1])
        args.tracker_type = "bytetrack"
        fn = tracker.make_runner(args, FakeModel())
        self.assertEqual(fn(object())[0], [10])


class RefindTests(unittest.TestCase):
    """The optional refind pass (refind_missed > 0)."""

    @staticmethod
    def _run(tr, frames):
        out = []
        for k, boxes in enumerate(frames):
            ids, _, _ = tr.update(list(range(100 * k, 100 * k + len(boxes))), boxes, [0.9] * len(boxes))
            out.append(ids)
        return out

    GONE_AND_BACK = ([[[100, 100, 40, 40], [300, 100, 40, 40]]] + [[[300, 100, 40, 40]]] * 6
                     + [[[110, 100, 40, 40], [300, 100, 40, 40]]])

    def test_off_by_default_is_the_earlier_tracker(self):
        ids = self._run(tracker.HybridIDManager(max_missed=4), self.GONE_AND_BACK)
        self.assertEqual(ids[0], [1, 2])
        self.assertEqual(ids[-1], [3, 2])

    def test_long_missing_bird_gets_its_id_back(self):
        ids = self._run(tracker.HybridIDManager(max_missed=4, refind_missed=20), self.GONE_AND_BACK)
        self.assertEqual(ids[-1], [1, 2])

    def test_never_takes_a_box_a_seen_bird_claims(self):
        # Bird 1 vanishes; bird 2 walks to where bird 1 was and keeps its own ID.
        frames = [[[100, 100, 40, 40], [200, 100, 40, 40]]] + [[[200 - 15 * k, 100, 40, 40]] for k in range(1, 7)]
        ids = self._run(tracker.HybridIDManager(max_missed=4, refind_missed=20), frames)
        self.assertTrue(all(row == [2] for row in ids[1:]))

    def test_id_is_released_after_refind_missed(self):
        frames = [[[100, 100, 40, 40]]] + [[]] * 21 + [[[100, 100, 40, 40]]]
        ids = self._run(tracker.HybridIDManager(max_missed=4, refind_missed=20), frames)
        self.assertEqual(ids[-1], [2])

    def test_refind_must_exceed_max_missed(self):
        with self.assertRaises(ValueError):
            tracker.HybridIDManager(max_missed=4, refind_missed=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
