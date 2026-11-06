"""Extract frames using exactly the same sampling routine as V2 evaluation.

Run from the behavior-pipeline project directory (requires common.py).
"""
import argparse
import csv
from pathlib import Path

import cv2
from common import sample_frames


def main():
    ap = argparse.ArgumentParser(description="Prepare an independent V2 validation sequence")
    ap.add_argument("--source", required=True, help="New video file, not the tuning clip")
    ap.add_argument("--outdir", default="validation_frames")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval must be positive")
    folder = Path(args.outdir)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.jpg")) or any(folder.glob("*.png")):
        ap.error(f"{folder} contains images. Use a new empty folder to avoid mixing frames.")

    count = 0
    with (folder / "timestamps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video_time_s", "filename"])
        for t, n, image in sample_frames(args.source, args.interval):
            if image is None:
                raise RuntimeError(f"Could not decode sampled frame {n}")
            name = f"frame_{n:06d}.jpg"
            if not cv2.imwrite(str(folder / name), image):
                raise RuntimeError(f"Could not save {folder / name}")
            writer.writerow([n, f"{t:.6f}", name])
            count += 1
    if not count:
        raise RuntimeError("No frames extracted; check the video path")
    print(f"Saved {count} frames in {folder}")
    print(f"Frame 1: {folder / 'frame_000001.jpg'}")
    print(f"Frame timing: {folder / 'timestamps.csv'}")
    print("Annotate these exact images with persistent bird IDs and export MOT ground truth.")


if __name__ == "__main__":
    main()
