"""Per-frame latency on the Raspberry Pi (or any machine).

    python benchmark_pi.py --weights best_ncnn_model --source clip.mkv --frames 100
    python benchmark_pi.py --weights best.onnx       --source clip.mkv
    python benchmark_pi.py --weights best.pt         --source clip.mkv

Reports preprocess / inference / postprocess (from Ultralytics) and tracking
(everything else in the model.track call), plus the total against your cycle time.
"""
import argparse
import statistics
import time

from ultralytics import YOLO

from common import sample_frames


def summarize(name, xs):
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    print(f"{name:<14} mean {statistics.mean(xs):8.1f} ms   p95 {p95:8.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--interval", type=float, default=2.0, help="sampling interval AND cycle budget")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    model = YOLO(args.weights, task="detect")
    pre, inf, post, trk, tot = [], [], [], [], []

    for t, n, frame in sample_frames(args.source, args.interval):
        t0 = time.perf_counter()
        r = model.track(frame, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz,
                        persist=True, verbose=False)[0]
        total = (time.perf_counter() - t0) * 1000
        if n > args.warmup:                      # skip first frames (model/graph warm-up)
            s = r.speed
            pre.append(s["preprocess"]); inf.append(s["inference"]); post.append(s["postprocess"])
            trk.append(max(0.0, total - s["preprocess"] - s["inference"] - s["postprocess"]))
            tot.append(total)
        if n >= args.frames + args.warmup:
            break

    print(f"\nFrames measured: {len(tot)}   model: {args.weights}")
    summarize("preprocess", pre)
    summarize("inference", inf)
    summarize("postprocess", post)
    summarize("tracking+misc", trk)
    summarize("TOTAL", tot)
    budget = args.interval * 1000
    print(f"\nCycle budget: {budget:.0f} ms -> {'OK' if max(tot) < budget else 'EXCEEDED on some frames'} "
          f"(worst frame {max(tot):.0f} ms)")


if __name__ == "__main__":
    main()