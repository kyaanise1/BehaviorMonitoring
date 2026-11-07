#!/usr/bin/env python3
"""Semi-automated, human-grounded 30-minute inactivity review for Day 6.

Only stdlib + OpenCV required. Does not run, import, or modify Hybrid V2.

Prepare: collect spatially spread or random *anchored* physical-bird review cases
from the existing per-track observations, save visual anchor/checkpoints, and
create a self-contained local HTML labeling page that uses the original video.
Score: compare separately annotated ground truth with recorded system outcomes.

Read README.md for essential caveats about genuine manual ground truth.
"""
import argparse
import csv
import html
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

FIELDS = ["case_id", "session", "anchor_track_id", "start_s", "end_s",
          "anchor_x", "anchor_y", "movement_limit_px", "pred_inactive",
          "pred_basis", "tracked_at_endpoint", "source_video", "anchor_image"]
ANNOTATION_FIELDS = ["case_id", "identity_visible", "recording_continuous",
                     "movement", "tracker_stable", "notes"]
TRUE_YES = {"yes"}
ALLOWED = {
    "identity_visible": {"", "yes", "no", "uncertain"},
    "recording_continuous": {"", "yes", "no", "uncertain"},
    "movement": {"", "stayed_within", "exceeded", "uncertain"},
    "tracker_stable": {"", "yes", "no", "uncertain"},
}


def read_csv(path, required):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def fmt_time(seconds):
    n = int(float(seconds))
    return f"{n//3600:02d}:{(n//60)%60:02d}:{n%60:02d}"


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_candidates(anchors, count, seed, selection):
    anchors = list(anchors)
    if count <= 0:
        raise ValueError("--samples must be positive")
    count = min(count, len(anchors))
    rng = random.Random(seed)
    if selection == "random":
        selected = rng.sample(anchors, count)
    else:
        # Spread start-frame review positions across the enclosure, instead of
        # selecting only tracks that already seem stationary to the algorithm.
        remaining = anchors[:]
        selected = [remaining.pop(rng.randrange(len(remaining)))]
        while len(selected) < count:
            selected_xy = [(float(v["cx"]), float(v["cy"])) for v in selected]
            idx = max(range(len(remaining)), key=lambda i: min(
                (float(remaining[i]["cx"]) - x) ** 2 +
                (float(remaining[i]["cy"]) - y) ** 2 for x, y in selected_xy))
            selected.append(remaining.pop(idx))
    return selected


def video_frame_at(cap, seconds, cv2):
    # common.sample_frames() uses decoded frame_index / fps (not MP4 PTS).
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = max(0, int(round(float(seconds) * fps)))
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
        raise RuntimeError(f"Video seeking failed at {fmt_time(seconds)}")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read original video at {fmt_time(seconds)} "
                           f"(frame {frame_idx}); check source/video timeline")
    return frame


def make_image(frame, path, cv2, *, marker=None, max_width=1100):
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width)
    if scale != 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    else:
        frame = frame.copy()
    if marker is not None:
        x, y = (int(round(float(z) * scale)) for z in marker)
        cv2.circle(frame, (x, y), 22, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 83]):
        raise RuntimeError(f"Could not save {path}")


def infer_prediction(target_id, session, end_t, track_by_key, behavior_row, tolerance=3.0):
    # Negative *flock-level* prediction establishes negative for every target.
    # Nonzero flock result requires an individual association at the endpoint.
    available = track_by_key.get((session, target_id), [])
    endpoint = next((r for r in available if abs(float(r["t_s"]) - end_t) <= tolerance), None)
    tracked = endpoint is not None
    if behavior_row is not None:
        if behavior_row.get("status") != "VALID":
            return "", "invalid_video_window", tracked
        try:
            n = int(behavior_row["n_inactive"])
        except (TypeError, ValueError):
            return "", "missing_behavior_prediction", tracked
        if n == 0:
            return "0", "zero_flock_alerts", tracked
        if tracked:
            ids = set(filter(None, behavior_row.get("inactive_ids", "").split("|")))
            return ("1" if str(target_id) in ids else "0"), "track_at_endpoint", True
        return "", "cannot_match_positive_flock_prediction", False
    if tracked:
        return endpoint["inactive"], "track_at_endpoint", True
    return "", "no_endpoint_prediction", False


def prepare(args):
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from e
    video = Path(args.source).resolve()
    outdir = Path(args.outdir).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if outdir == video.parent:
        raise ValueError("Use a separate output folder")

    obs = read_csv(args.observations, ["t_s", "session", "status", "track_id", "cx", "cy", "inactive"])
    if not obs:
        raise ValueError("Observations CSV has no tracking rows")
    by_key = defaultdict(list)
    for r in obs:
        by_key[(int(r["session"]), int(r["track_id"]))].append(r)
    for arr in by_key.values():
        arr.sort(key=lambda r: float(r["t_s"]))

    behaviors = read_csv(args.behavior_csv, ["t_s", "status", "n_inactive", "inactive_ids"]) if args.behavior_csv else []
    behaviors.sort(key=lambda r: float(r["t_s"]))
    target_start = float(args.start)
    observed_starts = sorted({float(r["t_s"]) for r in obs if float(r["t_s"]) >= target_start})
    if not observed_starts:
        raise ValueError(f"No observations at or after start {args.start}")
    start_t = observed_starts[0]
    if start_t - target_start > args.interval * 1.5:
        raise ValueError("No observation close to --start; choose a valid sample timestamp")
    start_rows = [r for r in obs if abs(float(r["t_s"]) - start_t) < 0.001 and r["status"] == "VALID"]
    if not start_rows:
        raise ValueError("Start sample isn't VALID; choose an uninterrupted window")
    sessions = {int(r["session"]) for r in start_rows}
    if len(sessions) != 1:
        raise ValueError("Start sample contains multiple tracking sessions")
    session = next(iter(sessions))
    end_min = start_t + args.window
    if behaviors:
        valid_ends = [float(r["t_s"]) for r in behaviors if r["status"] == "VALID" and float(r["t_s"]) >= end_min]
    else:
        valid_ends = [float(r["t_s"]) for r in obs if int(r["session"]) == session and r["status"] == "VALID" and float(r["t_s"]) >= end_min]
    if not valid_ends:
        raise ValueError("Not enough footage after --start to form a 30-minute window")
    end_t = min(valid_ends)
    if end_t - end_min > args.interval * 1.6:
        raise ValueError("First endpoint is too far after 30 minutes; check timeline")
    if behaviors:
        breaks = [r for r in behaviors if start_t <= float(r["t_s"]) <= end_t and r["status"] != "VALID"]
        if breaks:
            times = ", ".join(fmt_time(r["t_s"]) for r in breaks[:5])
            raise ValueError(f"Window contains GAP/RECOVERING at {times}; use a different --start")
    if any(int(r["session"]) != session and start_t < float(r["t_s"]) < end_t for r in obs):
        raise ValueError("The 30-minute window crosses a tracking reset")
    checkpoint_seconds = [start_t + args.window * i / 6 for i in range(7)]
    checkpoint_seconds[-1] = end_t
    behavior_at_end = None
    if behaviors:
        behavior_at_end = min(behaviors, key=lambda r: abs(float(r["t_s"]) - end_t))
        if abs(float(behavior_at_end["t_s"]) - end_t) > args.interval * 1.6:
            raise ValueError("Behavior CSV does not contain an endpoint close to the observations")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
        raise RuntimeError(f"Cannot read video: {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        approx_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        if approx_duration + 2.0 < end_t:
            raise ValueError(f"Video metadata duration {approx_duration:.1f}s is shorter than "
                             f"evaluation endpoint {end_t:.1f}s; verify video and CSV match")
        outdir.mkdir(parents=True, exist_ok=True)
        checkpoints = []
        frames = []
        for i, second in enumerate(checkpoint_seconds):
            frame = video_frame_at(cap, second, cv2)
            dest = outdir / "images" / f"checkpoint_{i:02d}.jpg"
            make_image(frame, dest, cv2)
            checkpoints.append({"time": fmt_time(second), "time_s": round(second, 3), "path": f"images/{dest.name}"})
            frames.append(frame if i == 0 else None)
        anchor_frame = frames[0]
        height, width = anchor_frame.shape[:2]
        anchors = [r for r in start_rows if 0 <= float(r["cx"]) < width and 0 <= float(r["cy"]) < height]
        if not anchors:
            raise ValueError("No start positions inside source video; CSV may not match this video")
        chosen = choose_candidates(anchors, args.samples, args.seed, args.selection)
        selected = []
        manifest = []
        for i, r in enumerate(chosen, 1):
            tid = int(r["track_id"])
            px, py = float(r["cx"]), float(r["cy"])
            case_id = f"D6-{i:03d}"
            img_name = f"images/{case_id}_anchor.jpg"
            make_image(anchor_frame, outdir / img_name, cv2, marker=(px,py))
            pred, basis, tracked = infer_prediction(tid, session, end_t, by_key, behavior_at_end, args.interval * 1.6)
            record = {
                "case_id": case_id,
                "session": str(session), "anchor_track_id": str(tid),
                "start_s": f"{start_t:.3f}", "end_s": f"{end_t:.3f}",
                "anchor_x": f"{px:.2f}", "anchor_y": f"{py:.2f}",
                "movement_limit_px": r.get("movement_limit_px", ""),
                "pred_inactive": pred, "pred_basis": basis,
                "tracked_at_endpoint": "1" if tracked else "0",
                "source_video": str(video), "anchor_image": img_name,
            }
            manifest.append(record)
            selected.append({
                "case_id":case_id, "anchor": img_name,
                "start_s":round(start_t,3), "end_s":round(end_t,3),
                "start": fmt_time(start_t), "end": fmt_time(end_t),
                "limit": record["movement_limit_px"] or "unknown",
                "track_id": tid,
                "points": [[round(float(row["t_s"]), 3), round(float(row["cx"]), 1), round(float(row["cy"]), 1)]
                           for row in by_key.get((session, tid), [])
                           if start_t <= float(row["t_s"]) <= end_t],
            })
    finally:
        cap.release()
    write_csv(outdir / "manifest.csv", FIELDS, manifest)
    relative_video = os.path.relpath(video, outdir).replace(os.sep, "/")
    video_url = "/".join(quote(section) for section in relative_video.split("/"))
    page_data = {"cases": selected, "checkpoints": checkpoints,
                 "video": video_url, "start": fmt_time(start_t), "end": fmt_time(end_t),
                 "window": args.window, "n_anchors": len(start_rows)}
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    page = template.replace("__INJECT_DATA_JSON__", json.dumps(page_data, ensure_ascii=False).replace("</", "<\\/"))
    (outdir / "review.html").write_text(page, encoding="utf-8")
    print(f"Prepared {len(manifest)} review cases from {len(anchors)} visible anchor tracks")
    print(f"Fixed review interval: {fmt_time(start_t)} to {fmt_time(end_t)} (>= {args.window:.0f}s)")
    print(f"Saved: {outdir / 'review.html'}")
    print(f"Saved: {outdir / 'manifest.csv'}")
    if not behaviors:
        print("WARNING: No --behavior-csv: gaps cannot be verified and absent IDs may lack predictions.")
    if not any(r["pred_inactive"] == "1" for r in manifest):
        print("WARNING: No system-positive cases in sampled set; positive precision may be undefined.")
    print("Review original CONTINUOUS video before labeling; still images alone cannot prove 30-min inactivity.")


def metric(num, den):
    return None if den == 0 else num / den


def score(args):
    manifest = read_csv(args.manifest, FIELDS)
    annotations = read_csv(args.annotations, ANNOTATION_FIELDS)
    seen = {}
    valid_ids = {r["case_id"] for r in manifest}
    for row in annotations:
        key = row["case_id"].strip()
        if key not in valid_ids:
            raise ValueError(f"Annotation case_id not in manifest: {key}")
        if key in seen:
            raise ValueError(f"Duplicate annotation case_id: {key}")
        for col, valid in ALLOWED.items():
            if row[col].strip().lower() not in valid:
                raise ValueError(f"Invalid '{col}' value '{row[col]}' for {key}")
        seen[key] = row
    tp=fp=fn=tn=0
    scored=[]
    reason_counts=defaultdict(int)
    for m in manifest:
        case = m["case_id"]
        a = seen.get(case)
        reason = None
        gt = pred = ""
        if a is None or all(not a.get(x, "").strip() for x in ["identity_visible","recording_continuous","movement"]):
            reason = "not_reviewed"
        else:
            identity = a["identity_visible"].strip().lower()
            continuous = a["recording_continuous"].strip().lower()
            movement = a["movement"].strip().lower()
            tracker_stable = a["tracker_stable"].strip().lower()
            if identity != "yes":
                reason = "physical_bird_not_reliably_followed"
            elif continuous != "yes":
                reason = "recording_not_continuous_or_uncertain"
            elif movement not in {"stayed_within", "exceeded"}:
                reason = "manual_movement_uncertain"
            elif m["pred_inactive"] not in ("0", "1"):
                reason = "no_matchable_system_prediction"
            elif m["pred_basis"] == "track_at_endpoint" and tracker_stable != "yes":
                reason = "track_identity_not_verified_for_positive_flock_prediction"
            else:
                gt = "1" if movement == "stayed_within" else "0"
                pred = m["pred_inactive"]
                if gt == "1" and pred == "1": result = "TP"; tp += 1
                elif gt == "0" and pred == "1": result = "FP"; fp += 1
                elif gt == "1" and pred == "0": result = "FN"; fn += 1
                else: result = "TN"; tn += 1
                reason = result
        reason_counts[reason] += 1
        scored.append({"case_id":case, "anchor_track_id":m["anchor_track_id"],
                       "start_s":m["start_s"], "end_s":m["end_s"],
                       "manual_gt":gt,"system_pred":pred, "result":reason,
                       "identity_visible": a["identity_visible"] if a else "",
                       "recording_continuous":a["recording_continuous"] if a else "",
                       "movement":a["movement"] if a else "",
                       "tracker_stable":a["tracker_stable"] if a else "",
                       "notes":a["notes"] if a else ""})
    n=tp+fp+fn+tn
    metrics = {
        "accuracy":metric(tp+tn,n), "precision":metric(tp,tp+fp),
        "recall":metric(tp,tp+fn), "f1_score":metric(2*tp,2*tp+fp+fn),
        "specificity":metric(tn,tn+fp),
    }
    output = Path(args.outdir) if args.outdir else Path(args.manifest).parent / "scores"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cases_scored.csv", ["case_id","anchor_track_id","start_s","end_s", "manual_gt","system_pred","result", "identity_visible","recording_continuous","movement","tracker_stable","notes"], scored)
    write_csv(output / "confusion_matrix.csv", ["actual", "pred_inactive", "pred_not_inactive"], [
        {"actual":"Inactive", "pred_inactive":tp, "pred_not_inactive":fn},
        {"actual":"Not inactive", "pred_inactive":fp, "pred_not_inactive":tn},
    ])
    report = {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"scored":n,
              "not_scored":len(manifest)-n,"reasons":dict(reason_counts),"metrics":metrics,
              "ground_truth_positive":tp+fn,"system_positive":tp+fp,
              "definition":"Per-physical-chicken, fixed 30-minute window at the manifest endpoint"}
    (output / "metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    def pct(value): return "N/A (undefined)" if value is None else f"{value*100:.2f}%"
    print(f"Scored {n}/{len(manifest)} cases | TP={tp} FP={fp} FN={fn} TN={tn}")
    for name, value in metrics.items(): print(f"{name.title():<12}: {pct(value)}")
    if tp+fn == 0:
        print("NOTE: No manual-positive cases. Recall and positive-class F1 cannot establish sensitivity.")
    if tp+fp == 0:
        print("NOTE: No system-positive cases. Positive precision is undefined.")
    if n: print("NOTE: These are reviewed anchored cases, not independent all-flock accuracy estimates.")
    print("Unscored reasons:", {k:v for k,v in reason_counts.items() if k not in {"TP","FP","FN","TN"}})
    print("Files:", output / "metrics.json", output / "confusion_matrix.csv", output / "cases_scored.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="Extract anchor/checkpoint images and create blind labeling interface")
    p.add_argument("--observations", required=True, help="Existing inactivity_track_observations.csv")
    p.add_argument("--source", required=True, help="Original Day 6 video matching observations")
    p.add_argument("--behavior-csv", help="Day 6 gap-aware behavior CSV (strongly recommended)")
    p.add_argument("--outdir", default="day6_inactivity_review")
    p.add_argument("--samples", type=int, default=12, help="Number of birds randomly selected at start of window")
    p.add_argument("--selection", choices=["spread","random"],default="spread")
    p.add_argument("--seed", type=int,default=42)
    p.add_argument("--start", type=float,default=0.0, help="Start of 30-minute continuous window in seconds")
    p.add_argument("--window", type=float,default=1800.0)
    p.add_argument("--interval", type=float,default=2.0)
    p.set_defaults(func=prepare)
    s = sub.add_parser("score", help="Score exported manual annotations against model predictions")
    s.add_argument("--manifest", required=True)
    s.add_argument("--annotations", required=True)
    s.add_argument("--outdir")
    s.set_defaults(func=score)
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except (OSError,ValueError,RuntimeError,csv.Error) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
