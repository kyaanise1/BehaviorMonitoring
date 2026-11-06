import unittest
import numpy as np
from gap_guard import iter_guarded_samples, parse_exclusions


def frame(v):
    a = np.zeros((96, 96, 3), dtype=np.uint8)
    a[:, :, :] = v
    return a


class GapGuardTests(unittest.TestCase):
    def read(self, samples, **kw):
        return list(iter_guarded_samples(samples, interval=2, **kw))

    def test_normal_video(self):
        actual = self.read([(i*2,i+1,frame(i*15)) for i in range(5)])
        self.assertEqual([x.status for x in actual], ["VALID"]*5)

    def test_freeze_not_counted_as_observation(self):
        vals = [3, 10, 10, 10, 10, 20, 23]
        actual = self.read([(i*2,i+1,frame(v)) for i,v in enumerate(vals)])
        self.assertEqual([x.status for x in actual],
                         ["VALID","VALID","GAP","GAP","GAP","VALID","VALID"])
        self.assertTrue(all(x.frame is None for x in actual if x.status == "GAP"))

    def test_two_identical_samples_are_not_automatically_frozen(self):
        vals = [2, 3, 3, 4]
        actual = self.read([(i*2,i+1,frame(v)) for i,v in enumerate(vals)])
        self.assertEqual([x.status for x in actual], ["VALID"]*4)

    def test_timestamp_jump(self):
        actual = self.read([(0,1,frame(1)), (2,2,frame(2)), (25,3,frame(3))])
        self.assertEqual([x.status for x in actual], ["VALID","VALID","GAP","VALID"])
        self.assertEqual(actual[2].reason, "timestamp_jump")
        self.assertEqual(actual[2].time, 4)

    def test_manual_exclusion(self):
        actual = self.read([(i*2,i+1,frame(i*10)) for i in range(5)], exclusions=[(2,6)])
        self.assertEqual([x.status for x in actual], ["VALID","GAP","GAP","VALID","VALID"])

    def test_bad_frame(self):
        actual = self.read([(0,1,frame(1)),(2,2,None),(4,3,frame(2))])
        self.assertEqual([x.status for x in actual], ["VALID","GAP","VALID"])

    def test_parse_exclusions(self):
        self.assertEqual(parse_exclusions(["00:41:02-00:46:19"]), [(2462.0,2779.0)])
        with self.assertRaises(ValueError):
            parse_exclusions(["00:02:00-00:01:00"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
