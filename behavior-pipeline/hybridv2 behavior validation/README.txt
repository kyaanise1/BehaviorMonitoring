IMPROVED OPTION A — Broiler-Net-inspired 30-minute cumulative activity
=================================================================

STATUS
------
- Complete Python files, designed for the EXISTING Hybrid V2 project.
- Checked: Python syntax, seven unit tests, Day 12 offline replay using your
  uploaded observations.csv + system_log.csv.
- NOT checked end-to-end: real video inference requires your own Hybrid V2
  hybrid_tracker_v2.py, behavior_huddling_v2.py, YOLO weights, configs, and video.
- The published Broiler-Net paper does not establish 360 or 450 px as your
  system's validated threshold. This is an experimental engineering adaptation.

WHAT IS INCLUDED
----------------
  behavior_inactivity_broilernet_smoothed.py
    Full, replacement-capable Option A inactivity module with raw / median3 /
    median5 modes. Keeps all original inactivity audit fields and adds detail.

  run_behavior_system_broilernet_smoothed.py
    Full video runner based on your supplied run_behavior_system_broilernet.py.
    Tracking, recovery, huddling, feeder zones, and their settings stay intact.
    Only inactivity imports, smoothing choice, and audit metadata change.

  replay_broilernet_smoothed_from_csv.py
    Full replay utility using already-saved centroid CSVs. No YOLO needed.

  test_broilernet_smoothed.py
    Reproducible seven unit tests.

  day12_replay_median3/
    Sample replay output from the CSV files uploaded to this chat.

INSTALL: DO NOT OVERWRITE YOUR WORKING FILES
--------------------------------------------
Place the FOUR .py files alongside the existing Broiler-Net scripts inside:

  behavior-pipeline\hybridv2 behavior validation\

The runner uses your existing folder structure:
  behavior-pipeline\behavior.py
  behavior-pipeline\common.py
  behavior-pipeline\hybrid_tracker_v2.py
  behavior-pipeline\behavior_huddling_v2.py
  behavior-pipeline\configs\...

If your files are organized differently, adjust PIPELINE_ROOT in the scripts.
Keep original run_behavior_system_broilernet.py and
behavior_inactivity_broilernet.py unchanged for comparison.

POWERSHELL — REPLAY YOUR SAVED DAY 12 CSVs (START HERE)
------------------------------------------------------
From the behavior-pipeline directory (change input/output paths to yours):

python ".\hybridv2 behavior validation\replay_broilernet_smoothed_from_csv.py" `
  --observations ".\results\inactivity_day12\observations.csv" `
  --log ".\results\inactivity_day12\system_log.csv" `
  --out-dir ".\results\day12_median3" `
  --interval 2 `
  --window 1800 `
  --smoothing median3 `
  --activity-threshold-px 360 `
  --min-fill 0.95 `
  --max-gap-s 6

The generated file smoothed_inactivity_observations.csv includes:
  cumulative_activity_px  = selected mode (median3 in this command)
  raw_activity_px         = original unfiltered Option A
  median3_activity_px     = 3-observation median-filtered path
  median5_activity_px     = 5-observation median-filtered path
  eligible                = time, coverage and gap requirements passed
  possible_id_jump        = raw jump > 1.5 estimated body lengths; AUDIT ONLY

The generated eligible_track_summary.csv reports ranges for each eligible ID.
The generated candidate_threshold_sweep.csv reports descriptive threshold
sensitivity, NOT ground-truth validation.

You can also use:
  --smoothing raw
  --smoothing median5
  --diagnostics-only
  --compare-thresholds 360,400,450,500,650,700

POWERSHELL — FULL VIDEO RUN
---------------------------
From behavior-pipeline (adjust actual video and weights paths):

python ".\hybridv2 behavior validation\run_behavior_system_broilernet_smoothed.py" `
  --source "..\video\Day12_repaired.mp4" `
  --weights "..\runs\broiler_yolo11n\weights\best.pt" `
  --tracker ".\configs\bytetrack_broiler.yaml" `
  --zones ".\configs\feeding_zones.json" `
  --interval 2 `
  --window 1800 `
  --smoothing median3 `
  --activity-threshold-px 360 `
  --min-fill 0.95 `
  --max-gap-s 6 `
  --inactivity-observations-csv ".\results\day12_median3_run\observations.csv" `
  --log-csv ".\results\day12_median3_run\system_log.csv"

TEST
----
python ".\hybridv2 behavior validation\test_broilernet_smoothed.py"

DECISION RULE
-------------
1. Build a 30-minute rolling history for each CURRENT Hybrid V2 track ID.
2. Use median3 on x/y centroids as a candidate jitter reduction step.
3. Sum Euclidean distances between successive filtered centroids.
4. Require >=855 observations in the window at a 2s interval (95% fill),
   a sample near the beginning of the window, a 30-min-old history, and
   no gap >6s between observations.
5. Mark INACTIVE when eligible and filtered sum < provisional threshold.
   This is Option A, NOT Option B; movement does NOT reset a 30-min timer.

IMPORTANT SCIENTIFIC LIMITATIONS
--------------------------------
- A different filtering method needs a DIFFERENT empirically calibrated
  activity threshold. Do not reuse 360 or report 450 as validated.
- Use separate calibration and testing recordings, independent physical-bird
  behavior labels, and the same sampling pipeline during calibration/testing.
- Your CVAT annotations include interpolation. Numeric activity from CVAT
  is NOT an exact target for YOLO centroid movement.
- Median filtering can erase short genuine movement; it cannot resolve ID
  switches or prove a chicken stayed visible. Review the tracking overlay.
- Suspected jumps are logged, not automatically clipped or silently deleted.
- Track IDs that fail time/coverage/gap eligibility are UNEVALUABLE; don't
  count them as true negative chickens when computing classification metrics.
- Does not change YOLO, ByteTrack, tracking recovery, or huddling.

DAY 12 REPLAY (UPLOADED FILES)
-----------------------------
With the rounded positions in your submitted CSVs:
  At ~1801.738 s, ID 15 raw activity:       ~631.812 px
  At ~1801.738 s, ID 15 median3 activity:   ~408.402 px
  At ~1801.738 s, ID 16 median3 activity:   ~725.812 px
  ID 15 median3 activity, eligible range:  400.126–449.797 px
  ID 15 threshold 360: 0 / 150 eligible samples flagged
  ID 15 threshold 450: 150 / 150 eligible samples flagged
  Other three eligible IDs at threshold 450: no flags

These are descriptive SAME-DAY observations, not an independent performance
validation or a recommendation to publish the 450 px threshold.
