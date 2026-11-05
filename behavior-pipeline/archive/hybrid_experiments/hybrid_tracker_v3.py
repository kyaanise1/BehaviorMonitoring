"""Experimental motion-aware recovery atop the existing ByteTrack+centroid hybrid.

Does not alter hybrid_tracker.py or hybrid_tracker_v2.py.
- Always preserve every ByteTrack output and the base HybridIDManager's IDs.
- Only recover unmatched high-confidence YOLO detections into DORMANT stable IDs.
- Associate with global one-to-one assignment, motion and box-size cues.
- Reject ambiguous associations rather than invent a confident identity.

This is a testable alternative to V2, not a proven accuracy improvement.
"""
import math
import statistics
from scipy.optimize import linear_sum_assignment
from hybrid_tracker_v2 import RecoveryHybrid, _iou


class MotionRecoveryHybrid(RecoveryHybrid):
    def __init__(self, *args, motion_weight=0.65, size_weight=0.20,
                 ambiguity_margin=0.12, max_velocity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= motion_weight <= 1:
            raise ValueError('motion_weight must be in [0,1]')
        if size_weight < 0 or ambiguity_margin < 0 or max_velocity <= 0:
            raise ValueError('invalid V3 matching parameters')
        self.motion_weight = motion_weight
        self.size_weight = size_weight
        self.ambiguity_margin = ambiguity_margin
        self.max_velocity = max_velocity
        self.frame_no = 0
        # tid -> (cx,cy,vx,vy,last_observed_frame)
        self.history = {}

    def _remember(self, tid, box, body):
        x, y = float(box[0]), float(box[1])
        prev = self.history.get(tid)
        if prev is None:
            vx, vy = 0.0, 0.0
        else:
            px, py, old_vx, old_vy, old_frame = prev
            gap = max(1, self.frame_no - old_frame)
            new_vx = (x-px)/gap
            new_vy = (y-py)/gap
            # Limit extreme velocities resulting from wrong initial matches.
            speed = math.hypot(new_vx, new_vy)
            cap = self.max_velocity * body
            if speed > cap:
                new_vx *= cap/speed
                new_vy *= cap/speed
            # For the first displacement, take measured velocity; smooth thereafter.
            if old_frame == 1:
                vx, vy = new_vx, new_vy
            else:
                vx = 0.6*new_vx + 0.4*old_vx
                vy = 0.6*new_vy + 0.4*old_vy
        self.history[tid] = (x, y, vx, vy, self.frame_no)

    def _cost(self, tid, track, box, body):
        x, y = float(box[0]), float(box[1])
        prev = self.history.get(tid)
        if prev is None:
            return None
        px, py, vx, vy, last = prev
        gap = max(1, self.frame_no-last)
        last_dist = math.hypot(px-x, py-y)
        pred_x = px+vx*gap
        pred_y = py+vy*gap
        motion_dist = math.hypot(pred_x-x, pred_y-y)
        # Recovery gate: a match is plausible either near last position or
        # near its motion-predicted location. A velocity estimate never proves ID.
        if min(motion_dist, last_dist) > self.recovery_distance*body:
            return None
        size_cost = (abs(math.log(track.w/box[2])) +
                     abs(math.log(track.h/box[3])))
        blended = (self.motion_weight*motion_dist +
                   (1.0-self.motion_weight)*last_dist)/body
        return blended + self.size_weight*size_cost

    def update(self, byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores):
        if len(raw_boxes) != len(raw_scores):
            raise ValueError('raw_boxes and raw_scores must have equal length')
        self.frame_no += 1
        stable_ids, boxes, scores = self.manager.update(byte_ids, byte_boxes, byte_scores)
        stable_ids, boxes, scores = list(stable_ids), list(boxes), list(scores)
        present = set(stable_ids)
        body = statistics.median(self.manager._body) if self.manager._body else 1.0
        body = max(body, 1e-6)

        # Motion estimates need to include the normal ByteTrack/centroid matches
        # before we consider raw-only recoveries, but dormant history stays intact.
        for tid, box in zip(stable_ids, boxes):
            self._remember(tid, box, body)

        raw_candidates = []
        for i, (box, score) in enumerate(zip(raw_boxes, raw_scores)):
            if float(score) < self.recovery_conf:
                continue
            b = list(map(float, box))
            if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                continue
            if any(_iou(b, previous) >= self.duplicate_iou or
                   math.hypot(b[0]-previous[0], b[1]-previous[1]) <= self.duplicate_center*body
                   for previous in boxes):
                continue
            raw_candidates.append((i, b, float(score)))

        dormant = [(tid, tr) for tid, tr in self.manager.tracks.items()
                   if tid not in present and 0 < tr.missed <= self.manager.max_missed]
        recovered = 0
        if dormant and raw_candidates:
            n, m = len(dormant), len(raw_candidates)
            # Additional per-track dummy columns allow any track to remain unmatched.
            # Treat invalid edges as enormous cost; do not force a bad association.
            LARGE = 1e6
            dummy_cost = self.recovery_distance + 1.0
            cost = [[LARGE] * (m+n) for _ in range(n)]
            for r, (tid, tr) in enumerate(dormant):
                cost[r][m+r] = dummy_cost
                for c, (i, b, score) in enumerate(raw_candidates):
                    if not self.manager._valid_size(tr,b):
                        continue
                    value = self._cost(tid,tr,b,body)
                    if value is not None:
                        cost[r][c] = value + 0.03*(1.0-score)

            rows, cols = linear_sum_assignment(cost)
            # A candidate is ambiguous if a near-equivalent alternative exists
            # for the same track OR the same detection. Check before updates.
            accepted = []
            for r,c in zip(rows,cols):
                if c>=m or cost[r][c]>=dummy_cost:
                    continue
                best = cost[r][c]
                alternatives = [cost[r][j] for j in range(m)
                                if j!=c and cost[r][j]<LARGE]
                alternatives += [cost[k][c] for k in range(n)
                                 if k!=r and cost[k][c]<LARGE]
                if alternatives and min(alternatives)-best < self.ambiguity_margin:
                    continue
                accepted.append((r,c))

            for r,c in accepted:
                tid, tr = dormant[r]
                _, b, score = raw_candidates[c]
                if tid in present:
                    continue
                if any(_iou(b,prev)>=self.duplicate_iou or
                       math.hypot(b[0]-prev[0],b[1]-prev[1])<=self.duplicate_center*body
                       for prev in boxes):
                    continue
                tr.cx,tr.cy,tr.w,tr.h = b
                tr.missed=0
                tr.hits+=1
                # Do NOT synthesize a ByteTrack ID for the raw-only detection.
                stable_ids.append(tid)
                boxes.append(b)
                scores.append(score)
                present.add(tid)
                self._remember(tid,b,body)
                recovered+=1

        assert len(stable_ids)==len(set(stable_ids))
        # Prune motion history for tracks the base manager has removed.
        for tid in list(self.history):
            if tid not in self.manager.tracks:
                del self.history[tid]
        self.frames+=1
        self.recovered_total+=recovered
        return stable_ids, boxes, scores, recovered
