# Hybrid V2 — gap-aware behavior-monitoring experiment

This is a **separate experimental runner**. It does not replace `run_pipeline_hybrid_v2.py`, `hybrid_tracker_v2.py`, or the previously reported tracking results.

## Install

Copy these four `.py` files into:

`behavior-pipeline/hybridv2 behavior validation/`

- `gap_guard.py`
- `run_pipeline_hybrid_v2_gapaware.py`
- `test_gap_guard.py`
- `eval_behavior_gapaware.py`

Keep all your existing project files in their original locations.

In **PowerShell**, while in `behavior-pipeline`:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python ".\hybridv2 behavior validation\test_gap_guard.py"
```

## Run on repaired Day 6 video

```powershell
python ".\hybridv2 behavior validation\run_pipeline_hybrid_v2_gapaware.py" --source "../video/Day6_repaired.mp4" --weights "../runs/broiler_yolo11n/weights/best.pt" --tracker "configs/bytetrack_broiler.yaml" --interval 2 --window 1800 --min-huddle 3 --log-csv "behavior_day6_gapaware.csv"
```

For a manually confirmed *invalid interval on the repaired video's playback time*, add (example only):

```powershell
--exclude "00:41:02-00:46:19"
```

**Do not assume these timecodes apply to the repaired MP4.** Inspect its playback timeline first; the original MKV's camera-clock jump might not map directly to MP4 playback time. The automatic freeze detector does not require manual `--exclude`.

## What happens at a gap?

- Three near-identical sequential frames by default trigger a freeze; repeated frames are `GAP`.
- A timestamp jump over 6 s is `GAP` (but see timestamp caution below).
- Unreadable frames and frames with zero tracked birds are `GAP`.
- After a gap, the runner **recreates ByteTrack and Hybrid V2 and clears behavior histories**.
- The first three resumed samples are `RECOVERING` and have no behavior label.
- Only subsequent `VALID` samples are eligible for huddling/inactivity labels.
- CSV columns `status` and `reason` explain each sample. Invalid/recovering behavior counts are BLANK, not zero.

## Check results

```powershell
Import-Csv "behavior_day6_gapaware.csv" | Group-Object status | Select-Object Name,Count
Import-Csv "behavior_day6_gapaware.csv" | Where-Object { $_.status -eq "GAP" } | Select-Object -First 15 t_s,status,reason | Format-Table
Import-Csv "behavior_day6_gapaware.csv" | Where-Object { $_.status -eq "VALID" -and [int]$_.n_inactive -gt 0 } | Select-Object -First 15 t_s,inactive_ids | Format-Table
```

### Behavior ground truth

When you have independently annotated true behavior intervals in `behavior_gt_day6.csv`:

```powershell
python ".\hybridv2 behavior validation\eval_behavior_gapaware.py" --gt "behavior_gt_day6.csv" --pred "behavior_day6_gapaware.csv" --step 5
```

**Do not use the old `eval_behavior.py`** on the new CSV: it does not skip invalid frames and would treat GAP/RECOVERING rows incorrectly. The new evaluator scores only VALID time points. It still measures any-huddle / any-inactivity at each time, not per-chicken outcomes.

## Caveats important for your thesis

1. Visual freeze detection is a **heuristic**. A very still scene might be flagged; compression artifacts or an updating camera overlay could hide a freeze. Review reported intervals against footage and tune `--freeze-diff` / `--freeze-frames` only on development footage. The guard excludes top/bottom 10% for the frozen-scene comparison to reduce sensitivity to camera timestamps.
2. `common.sample_frames` computes video times using **frame index divided by OpenCV FPS**, not true camera timestamps. A jump in the source MKV camera-clock overlay may not become a timestamp jump in the processed stream. For known discontinuities, verify the repaired-video timeline and use `--exclude` or create segments with preserved real-time metadata.
3. Detecting a freeze **does not restore missing movements**. Do not claim that a chicken remained inactive across a recording gap. A detected gap resets the entire 30-minute observation history.
4. `--min-fill 0.5` is inherited from the original behavior rules, but *does not prove individual identity continuity across 30 min*. Check long tracks against video manually.
5. Video recording failures should be reported as a data-acquisition limitation, with the amount of valid and excluded footage, not hidden.
6. This test still uses two YOLO passes per valid frame; not optimized for Raspberry Pi deployment.
