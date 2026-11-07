# Second-rally check: 36.0–40.0 s

This is a separate rally in `data/raw/clip.mp4`. The camera and court lines match
the 20–23 s development segment, so it uses `outputs/court_points.json`.
The model settings are unchanged. Tracking runs at 10 samples per second.

Generated files:

- `preview.jpg`: raw-video contact sheet used to choose this rally.
- `tracking_v3.json`: original v3 tracker, 40 timestamps.
- `tracking_v4_pose.json`: accepted ankle midpoint, otherwise v3.
- `tracking_v4_hybrid.json`: 65% accepted pose, 35% v3, otherwise v3.
- `fusion_diagnostics.json`: pose acceptance and rejection reasons.
- `evaluation/RESULTS.md`: completed comparison against the manual labels.

## Manual labels needed

Open a PowerShell terminal in the project folder and run:

```powershell
.\.venv\Scripts\python.exe annotate_players.py --start 36 --end 40 --step 0.3 --output outputs\holdout_36_40\manual_evaluation.json
```

This opens 14 raw-video frames from 36.0 through 39.9 s. For each frame, press
`1` for Near and `2` for Far, then click the midpoint of both grounded shoes,
or the grounded shoe when only one is touching. Use `X` if the ground point is
unclear. Press `N` for the next frame and `Q`
to save and quit. Do not inspect tracker markers while placing labels.

Once the labels are complete, compare the three tracks with:

```powershell
.\.venv\Scripts\python.exe evaluate_trackers.py --reference outputs\holdout_36_40\manual_evaluation.json --output-dir outputs\holdout_36_40\evaluation --tracker "v3=outputs\holdout_36_40\tracking_v3.json" --tracker "Pose=outputs\holdout_36_40\tracking_v4_pose.json" --tracker "Hybrid=outputs\holdout_36_40\tracking_v4_hybrid.json"
```

This is a new rally, but it is from the same match and camera. It checks a
different movement sequence; it does not test other venues or cameras.
