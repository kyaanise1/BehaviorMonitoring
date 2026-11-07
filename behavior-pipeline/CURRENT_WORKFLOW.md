# Current behavior pipeline

**Status:** Hybrid V2, feeder-aware, strict no-chaining, with local and temporal recovery.

Current runner (do not rename or relocate without updating commands):
`hybridv2 behavior validation/run_pipeline_hybrid_v2_feeder_aware_linkage_test.py`

Key dependencies (remain at their original paths):
- `common.py`, `hybrid_tracker_v2.py`, `hybrid_tracker.py`
- `behavior.py`, `evaluate_huddling_clusters.py`
- `hybridv2 behavior validation/behavior_linkage_test.py`
- `configs/` (tracker and current feeder/drinker zones)

Primary reproducibility files:
- `day5_annotations/` and `day5_validation_frames/`
- `day5_validation_timestamps.csv`
- `results/day5/reference_strict/` (copied reference outputs)
- Original day5_strict_final_*.csv files left in place to preserve old CLI commands

Day 5 reproduced reference at IoU >= 0.50 (60 frames, 72 manual clusters):
TP=54, FP=6, FN=18, precision=.9000, recall=.7500, F1=.8182.
This is a DEVELOPMENT result, not an independent holdout.

Other source scripts and settings were intentionally left at the old paths. See
`archive/code/`, `archive/visual_reviews/`, and `archive/results/` for history.

Do not confuse earlier methods (connected/average chaining, bbox gap recovery)
with the current strict method. Confirm the exact runner parameters when
reproducing metrics.