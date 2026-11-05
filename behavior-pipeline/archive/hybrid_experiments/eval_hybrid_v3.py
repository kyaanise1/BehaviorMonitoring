"""Evaluate conservative extra-detection recovery without changing old scripts.

WARNING: Two independent YOLO model instances run inference per frame, so this
version is for offline experimentation, NOT Raspberry Pi performance testing.
It avoids having a second model.predict call modify ByteTrack's predictor state.
"""
import argparse


def predict(args):
    from ultralytics import YOLO
    from common import sample_frames
    from hybrid_tracker_v3 import MotionRecoveryHybrid

    tracker_model = YOLO(args.weights, task="detect")
    detector_model = YOLO(args.weights, task="detect")
    hybrid = MotionRecoveryHybrid(tight=args.tight,wide=args.wide,
                            max_missed=args.max_missed,
                            wide_conf=args.wide_conf,
                            size_ratio=args.hybrid_size_ratio,
                            recovery_conf=args.recovery_conf,
                            recovery_distance=args.recovery_distance,
                            duplicate_iou=args.duplicate_iou,
                            motion_weight=args.motion_weight,
                            size_weight=args.size_weight,
                            ambiguity_margin=args.ambiguity_margin)
    with open(args.out,"w",encoding="utf-8") as f:
        for t,n,frame in sample_frames(args.source,args.interval):
            # ByteTrack and raw YOLO on separate YOLO model instances.
            bt = tracker_model.track(frame,tracker=args.tracker,conf=args.conf,
                                     imgsz=args.imgsz,persist=True,verbose=False)[0]
            if bt.boxes is None or bt.boxes.id is None:
                byte_ids,byte_boxes,byte_scores=[],[],[]
            else:
                byte_ids=bt.boxes.id.int().tolist()
                byte_boxes=bt.boxes.xywh.tolist()
                byte_scores=bt.boxes.conf.tolist()
            raw = detector_model.predict(frame,conf=args.conf,imgsz=args.imgsz,
                                          verbose=False)[0]
            if raw.boxes is None:
                raw_boxes,raw_scores=[],[]
            else:
                raw_boxes=raw.boxes.xywh.tolist()
                raw_scores=raw.boxes.conf.tolist()
            ids, boxes, scores, recovered = hybrid.update(
                byte_ids,byte_boxes,byte_scores,raw_boxes,raw_scores)
            for tid,(cx,cy,w,h),score in zip(ids,boxes,scores):
                f.write(f"{n},{tid},{cx-w/2:.2f},{cy-h/2:.2f},{w:.2f},{h:.2f},{score:.3f},-1,-1,-1\n")
            if args.verbose_frames and (recovered or n<=9):
                print(f"Frame {n:>3}: ByteTrack={len(byte_ids):>2}, "
                      f"YOLO={len(raw_boxes):>2}, Recovered={recovered:>2}, "
                      f"HybridV3={len(ids):>2}")
    print(f"Saved {args.out}; recovered {hybrid.recovered_total} "
          f"detections across {hybrid.frames} frames.")


def main():
    p=argparse.ArgumentParser(description="Experimental v3 hybrid evaluation")
    sub=p.add_subparsers(dest="cmd",required=True)
    run=sub.add_parser("predict")
    run.add_argument("--source",required=True)
    run.add_argument("--weights",required=True)
    run.add_argument("--tracker",default="configs/bytetrack_broiler.yaml")
    run.add_argument("--interval",type=float,default=2.0)
    run.add_argument("--conf",type=float,default=0.1)
    run.add_argument("--imgsz",type=int,default=640)
    run.add_argument("--out",default="pred_hybrid_v3.txt")
    run.add_argument("--tight",type=float,default=0.5)
    run.add_argument("--wide",type=float,default=4.0)
    run.add_argument("--max-missed",type=int,default=4)
    run.add_argument("--wide-conf",type=float,default=0.3)
    run.add_argument("--hybrid-size-ratio",type=float,default=2.0)
    run.add_argument("--recovery-conf",type=float,default=0.7)
    run.add_argument("--recovery-distance",type=float,default=1.5)
    run.add_argument("--duplicate-iou",type=float,default=0.5)
    run.add_argument("--motion-weight",type=float,default=0.65)
    run.add_argument("--size-weight",type=float,default=0.20)
    run.add_argument("--ambiguity-margin",type=float,default=0.12)
    run.add_argument("--verbose-frames",action="store_true")
    run.set_defaults(fn=predict)
    score=sub.add_parser("score")
    score.add_argument("--gt",required=True)
    score.add_argument("--pred",required=True)
    score.add_argument("--iou-thr",type=float,default=0.5)
    def do_score(args):
        from eval_hybrid_tracking import score as score_fn
        score_fn(args)
    score.set_defaults(fn=do_score)
    args=p.parse_args()
    args.fn(args)

if __name__=="__main__":
    main()
