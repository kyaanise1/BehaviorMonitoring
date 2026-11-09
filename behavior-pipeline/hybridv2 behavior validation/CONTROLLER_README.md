# Hybrid V2 shared controller — original inactivity, no gap-aware processing

Replace only these three files in `behavior-pipeline/hybridv2 behavior validation/`:

- `run_behavior_system.py` — shared video/YOLO/Hybrid V2 tracking, both behavior outputs
- `behavior_inactivity_v2.py` — 30-minute inactivity rule from the uploaded `run_pipeline_hybrid_v2(3).py`, using the same `behavior.py` functions as that runner
- `behavior_huddling_v2.py` — unchanged feeder-aware huddling module (average-linkage, strict no-chaining, local/temporal recovery)

Other original scripts and `run_behavior_v2.py` do not need to be deleted or changed. Dependency modules remain where they are: `common.py`, `hybrid_tracker_v2.py`, `behavior.py` in parent `behavior-pipeline`; `behavior_linkage_test.py` in validation folder. The **new controller does not import** `gap_guard.py`.

## Run Day 6 repaired video

From `behavior-pipeline` in the `.venv` PowerShell terminal:

```powershell
python ".\hybridv2 behavior validation\run_behavior_system.py" --source "D:\Frijiel Magpulong\Documents\USTP\Thesis\Data Gathering\BehaviorMonitoring\video\Day6_repaired.mp4" --weights "../runs/broiler_yolo11n/weights/best.pt" --tracker "configs/bytetrack_broiler.yaml" --interval 2 --conf 0.1 --window 1800 --min-huddle 3 --zones "configs/feeding_zones.json" --huddle-method average --cluster-zone-overlap 0.30 --local-recovery --temporal-recovery --log-csv "day6_system_nogap_log.csv" --clusters-csv "day6_system_nogap_clusters.csv" --inactivity-observations-csv "day6_system_nogap_inactivity_observations.csv"
```

This processes **all available sampled frames**; the optional `--timestamps-csv` restricts *only cluster exports*, not the underlying tracking or 30-minute history. The following outputs are produced:

- `day6_system_nogap_log.csv`: `t_s`, bird count, huddling count, inactivity count, track IDs, feeder exclusion count. No GAP/RECOVERING status.
- `day6_system_nogap_clusters.csv`: feeder-aware predicted huddling clusters, with optional temporal `origin` column.
- `day6_system_nogap_inactivity_observations.csv`: movement/history audit per track, compatible with `finish_inactivity_summary.py`; `session=0` exists solely for compatibility and **does not certify continuous footage**.

To create the inactivity summary (diagnostics only, NOT accuracy):

```powershell
python ".\hybridv2 behavior validation\finish_inactivity_summary.py" --observations-csv "day6_system_nogap_inactivity_observations.csv" --summary-csv "day6_system_nogap_inactivity_summary.csv" --interval 2
```

Validate huddling using independent Day 6 annotations (three flags are required by the existing evaluator, but all point to the same new predictions):

```powershell
python evaluate_huddling_clusters.py --annotations "day6_annotations" --baseline "day6_system_nogap_clusters.csv" --feeder-aware "day6_system_nogap_clusters.csv" --strict "day6_system_nogap_clusters.csv" --output "day6_system_nogap_evaluation.csv" --per-frame "day6_system_nogap_per_frame.csv"
```

## What changed versus the previous controller

- Removed the `gap_guard` import and all `GAP` / `RECOVERING` state management, video freeze detection, time-jump checking, automatic reset, warmup, `--no-gap-guard`, `--freeze-*`, `--max-time-step`, `--warmup-frames`, `--min-visible`, and `--exclude` CLI parameters.
- Inactivity is calculated exactly via `behavior.py`'s `prune_history`, `estimate_body_length`, and `update_inactivity`, as called in the uploaded `run_pipeline_hybrid_v2(3).py`, with defaults **1800 seconds, movement threshold 0.5 x estimated body length, sample fill 0.5**.
- Huddling rules and controller's Hybrid V2 tracking configuration are unchanged; both still share one tracking stream. Existing two YOLO calls per frame (track + raw recovery) remain.
- No artificial histories are created during missing frames, but the script will NOT detect timeline faults. A repaired MP4 may have missing/duplicated footage even if it plays normally. **Use manually verified continuous video for inactivity validity**; never interpret this runner's output as proof of continuity.
- The shared controller retains one unified behavior CSV, but its first eight columns match the original inactivity runner's output rather than the previous gap-aware CSV. `eval_behavior_gapaware.py` expects statuses and is no longer applicable; use `eval_behavior.py` when independently annotated interval ground truth exists.

**Testing**: dependency-mocked comparisons check huddling parity against the previous controller running with gap guard off, plus inactivity parity with the uploaded original runner, using a shortened synthetic 12-second window. This is not a real-video F1 test. A full Day 6 run and independent inactivity labels are still needed for actual performance validation.
