"""Retired: CVAT frame numbers must not be rescaled.

This script used to write finald12_annotations_15fps.xml with every frame number multiplied by 15 / 11.01. The
CVAT frame numbers are already SOURCE frames of the 15 fps Day 12 recording (CVAT keeps the original frame index;
step=30 means one annotated picture every 30 source frames = every 2 s). Rescaling them by 15 / 11.01 and then
dividing by 15 gives exactly the same times as dividing the original numbers by 11.01, so the "converted" file
was still timed at 11 fps and its boxes did not sit on the birds (mean best IoU against detections 0.316, against
0.757 for the original numbers / 15). It has been removed.

What to do instead:
  * keep finald12_annotations.xml as it is and use time = frame / 15 (validate_day12_inactivity_updated.py now
    defaults to --fps 15);
  * run the pipeline on the ORIGINAL Day_12_2026-09-08_0600_to_0700.mkv, not Day12_repaired.mp4 (see FPS_NOTE.txt);
  * check any annotation/run pair with check_cvat_fps.py.
"""
import sys

if __name__ == "__main__":
    sys.exit(__doc__)
