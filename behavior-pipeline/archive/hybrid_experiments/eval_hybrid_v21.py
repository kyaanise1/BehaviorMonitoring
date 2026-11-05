"""Side-by-side V2.1 hybrid offline evaluation; never overwrites V2 files.

Uses two YOLO models for the same reasons as V2, not optimized for Pi deployment.
Requires sibling files hybrid_tracker_v21.py, hybrid_tracker_v2.py,
hybrid_tracker.py, common.py, and eval_hybrid_tracking.py.
"""
import argparse


def predict(args):
    from ultralytics import YOLO
    from common import sample_frames
    from hybrid_tracker_v21 import RecoveryHybridV21

    tracked_model = YOLO(args.weights, task="detect")
    raw_model = YOLO(args.weights, task="detect")
    hybrid = RecoveryHybridV21(
        tight=args.tight, wide=args.wide, max_missed=args.max_missed,
        wide_conf=args.wide_conf, size_ratio=args.hybrid_size_ratio,
        recovery_conf=args.recovery_conf,
        recovery_distance=args.recovery_distance,
        duplicate_iou=args.duplicate_iou,
        identity_mode=args.identity_mode,
        byte_bonus=args.byte_bonus, same_id_gate=args.same_id_gate,
        size_weight=args.size_weight)

    with open(args.out, "w", encoding="utf-8") as f:
        for _, number, frame in sample_frames(args.source, args.interval):
            bt_result = tracked_model.track(
                frame, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz,
                persist=True, verbose=False)[0]
            if bt_result.boxes is None or bt_result.boxes.id is None:
                byte_ids, byte_boxes, byte_scores = [], [], []
            else:
                byte_ids = bt_result.boxes.id.int().tolist()
                byte_boxes = bt_result.boxes.xywh.tolist()
                byte_scores = bt_result.boxes.conf.tolist()

            raw_result = raw_model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            if raw_result.boxes is None:
                raw_boxes, raw_scores = [], []
            else:
                raw_boxes = raw_result.boxes.xywh.tolist()
                raw_scores = raw_result.boxes.conf.tolist()

            stable_ids, boxes, scores, recovered = hybrid.update(
                byte_ids, byte_boxes, byte_scores, raw_boxes, raw_scores)
            for tid, (cx, cy, w, h), score in zip(stable_ids, boxes, scores):
                f.write(f"{number},{tid},{cx-w/2:.2f},{cy-h/2:.2f},"
                        f"{w:.2f},{h:.2f},{score:.3f},-1,-1,-1\n")
            if args.verbose_frames and (recovered or number <= 9):
                print(f"Frame {number:>3}: ByteTrack={len(byte_ids):>2}, "
                      f"YOLO={len(raw_boxes):>2}, Recovered={recovered:>2}, "
                      f"HybridV2.1={len(stable_ids):>2}")

    print(f"Saved {args.out}; recovered {hybrid.recovered_total} "
          f"detections across {hybrid.frames} frames "
          f"(association={args.identity_mode}).")
    if args.identity_mode == "conflict":
        print("Close matches:", hybrid.manager.close_matches,
              "| same-ByteTrack-ID close matches:",
              hybrid.manager.byte_continuity_matches)


def score(args):
    from eval_hybrid_tracking import score as existing_score
    existing_score(args)


def main():
    parser = argparse.ArgumentParser(description="Hybrid V2.1 experiment")
    subs = parser.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("predict")
    p.add_argument("--source", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--tracker", default="configs/bytetrack_broiler.yaml")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="pred_hybrid_v21.txt")
    p.add_argument("--identity-mode", choices=["baseline", "conflict"],
                   default="conflict")
    p.add_argument("--tight", type=float, default=0.5)
    p.add_argument("--wide", type=float, default=4.0)
    p.add_argument("--max-missed", type=int, default=4)
    p.add_argument("--wide-conf", type=float, default=0.3)
    p.add_argument("--hybrid-size-ratio", type=float, default=2.0)
    p.add_argument("--recovery-conf", type=float, default=0.7)
    p.add_argument("--recovery-distance", type=float, default=1.5)
    p.add_argument("--duplicate-iou", type=float, default=0.5)
    p.add_argument("--byte-bonus", type=float, default=0.35,
                   help="Body-length cost reduction for plausible same-ByteTrack-ID matches")
    p.add_argument("--same-id-gate", type=float, default=1.25,
                   help="Max body-length displacement for first-pass same-ID candidate")
    p.add_argument("--size-weight", type=float, default=0.10,
                   help="Weight of size-consistency penalty in first-pass association")
    p.add_argument("--verbose-frames", action="store_true")
    p.set_defaults(fn=predict)

    s = subs.add_parser("score")
    s.add_argument("--gt", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--iou-thr", type=float, default=0.5)
    s.set_defaults(fn=score)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
