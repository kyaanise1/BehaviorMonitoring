"""Unit tests for current behavior.py rule logic; synthetic, not field validation."""
import unittest
from collections import deque
from behavior import detect_huddles, update_inactivity


class BehaviorRuleTests(unittest.TestCase):
    def test_30_minute_stationary(self):
        history_early = {7: deque((float(t), 100.0, 100.0) for t in range(0, 1800, 2))}
        history_full = {7: deque((float(t), 100.0, 100.0) for t in range(0, 1801, 2))}
        self.assertNotIn(7, update_inactivity(history_early, 1798., 1800., 15., 450))
        self.assertIn(7, update_inactivity(history_full, 1800., 1800., 15., 450))

    def test_30_minute_moving(self):
        history = {7: deque((float(t), 100.0 + t / 4, 100.0) for t in range(0, 1801, 2))}
        self.assertNotIn(7, update_inactivity(history, 1800., 1800., 15., 450))

    def test_sparse_history_is_not_inactive(self):
        history = {7: deque([(0.0, 100.0, 100.0), (900.0, 100.0, 100.0),
                             (1800.0, 100.0, 100.0)])}
        self.assertNotIn(7, update_inactivity(history, 1800., 1800., 15., 450))

    def test_huddling_requires_three(self):
        birds = [(100, 100), (105, 102), (107, 103), (400, 400)]
        groups = detect_huddles(birds, body_len_px=20, min_birds=3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_only_two_not_huddling(self):
        self.assertEqual(detect_huddles([(100, 100), (108, 100), (400, 400)], 20, 3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
