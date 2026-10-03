"""video -> YOLO detect -> ByteTrack -> behavior rules -> CSV log (+ optional cloud).

Examples
    python run_pipeline.py --source clip.mkv --weights runs/broiler_yolo11n/weights/best.pt
    python run_pipeline.py --source rtsp://user:pass@ip/stream --live --cloud
    # short test of the inactivity rule (2-minute window instead of 30):
    python run_pipeline.py --source clip.mkv --window 120
"""
import argparse
import csv
from collections import defaultdict, deque

from ultralytics import YOLO

from behavior import detect_huddles, estimate_body_length, prune_history, update_inactivity
from common import sample_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, RTSP URL, or camera index")
    ap.add_argument("--weights", default="runs/broiler_yolo11n/weights/best.pt")
    ap.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between processed frames")
    ap.add_argument("--conf", type=float, default=0.1, help="keep low: ByteTrack does the 0.6 split")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--window", type=float, default=1800, help="inactivity window in seconds")
    ap.add_argument("--move-frac", type=float, default=0.5,
                    help="max centroid movement for 'inactive', as a fraction of body length")
    ap.add_argument("--min-fill", type=float, default=0.5,
                    help="fraction of expected samples a track needs inside the window")
    ap.add_argument("--min-huddle", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=300, help="seconds between repeat events")
    ap.add_argument("--log-csv", default="state_log.csv")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cloud", action="store_true", help="upload events to MongoDB")
    args = ap.parse_args()

    cloud = None
    if args.cloud:
        import cloud  # lazy: pymongo only needed when uploading

    # task="detect" is needed when loading an exported NCNN/ONNX folder or file
    model = YOLO(args.weights, task="detect")

    history = defaultdict(deque)           # track_id -> deque[(t, cx, cy)]
    body_lens = deque(maxlen=500)          # running body-length samples (pixels)
    last_logged = {}                       # event key -> last time logged
    min_samples = max(1, int(args.min_fill * args.window / args.interval))

    def should_log(key, t):
        if t - last_logged.get(key, float("-inf")) >= args.cooldown:
            last_logged[key] = t
            return True
        return False

    with open(args.log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "n_birds", "n_huddles", "n_inactive", "body_len_px"])

        for t, n, frame in sample_frames(args.source, args.interval, args.live):
            r = model.track(frame, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz,
                            persist=True, verbose=False)[0]

            ids, xywh = [], []
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                xywh = r.boxes.xywh.tolist()           # cx, cy, w, h in pixels

            for tid, (cx, cy, bw, bh) in zip(ids, xywh):
                history[tid].append((t, cx, cy))
            prune_history(history, t, args.window, args.interval)

            if xywh:
                body_lens.append(estimate_body_length([(b[2], b[3]) for b in xywh]))
            body_len = sorted(body_lens)[len(body_lens) // 2] if body_lens else None

            huddles, inactive = [], []
            if body_len:
                huddles = detect_huddles([(b[0], b[1]) for b in xywh], body_len, args.min_huddle)
                inactive = update_inactivity(history, t, args.window,
                                             args.move_frac * body_len, min_samples)

            w.writerow([f"{t:.1f}", len(ids), len(huddles), len(inactive),
                        f"{body_len:.1f}" if body_len else ""])
            f.flush()

            # events (rate-limited so one long episode is not logged every frame)
            if huddles and should_log("huddle", t):
                size = max(len(c) for c in huddles)
                print(f"[{t:8.1f}s] HUDDLING: {len(huddles)} cluster(s), largest = {size} birds")
                if cloud:
                    cloud.log_event("huddling", {"video_t": t, "clusters": len(huddles),
                                                 "largest": size, "n_birds": len(ids)})
            for tid in inactive:
                if should_log(("inactive", tid), t):
                    print(f"[{t:8.1f}s] INACTIVE: track {tid} for >= {args.window:.0f}s")
                    if cloud:
                        cloud.log_event("inactivity", {"video_t": t, "track_id": tid,
                                                       "window_s": args.window})

            if n % 30 == 0:
                print(f"[{t:8.1f}s] sample {n}: {len(ids)} birds, {len(history)} tracks")


if __name__ == "__main__":
    main()