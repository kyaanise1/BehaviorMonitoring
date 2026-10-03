"""Rule-based behavior logic. No YOLO or video code in here: numbers in, results out.

history format: {track_id: deque([(t, cx, cy), ...])}   (t in seconds, pixels)
"""
import math
import statistics


def prune_history(history, now, window_s, interval_s):
    """Drop points older than the window and forget tracks not seen for a full window."""
    keep_from = now - window_s - 2 * interval_s
    for tid in list(history):
        pts = history[tid]
        while pts and pts[0][0] < keep_from:
            pts.popleft()
        if not pts or now - pts[-1][0] > window_s:
            del history[tid]


def update_inactivity(history, now, window_s=1800, move_thresh_px=15, min_samples=1):
    """Return IDs whose centroid stayed inside a move_thresh_px box for window_s.

    A track qualifies only if it is visible in the current frame, its history
    spans the whole window, and it has at least min_samples points in that window
    (so sparse, flickering tracks are not flagged).
    """
    inactive = []
    for tid, pts in history.items():
        if not pts or pts[-1][0] != now:       # must be seen right now
            continue
        if now - pts[0][0] < window_s:         # not enough history yet
            continue
        recent = [p for p in pts if now - p[0] <= window_s]
        if len(recent) < min_samples:
            continue
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        if (max(xs) - min(xs)) < move_thresh_px and (max(ys) - min(ys)) < move_thresh_px:
            inactive.append(tid)
    return inactive


def detect_huddles(centroids, body_len_px, min_birds=3):
    """Group birds closer than one body length (connected components).

    centroids: [(cx, cy), ...]. Returns a list of clusters (lists of indices)
    that contain at least min_birds birds.
    """
    n = len(centroids)
    seen, huddles = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, cluster = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            cluster.append(i)
            for j in range(n):
                if j not in seen and math.dist(centroids[i], centroids[j]) < body_len_px:
                    seen.add(j)
                    stack.append(j)
        if len(cluster) >= min_birds:
            huddles.append(cluster)
    return huddles


def estimate_body_length(boxes_wh):
    """Median of the longer box side across birds, in pixels."""
    if not boxes_wh:
        return None
    return statistics.median(max(w, h) for w, h in boxes_wh)