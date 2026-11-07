# Second-rally evaluation (36.0–40.0 s)

The same v3 and pose-fusion settings used for 20–23 s were run on 40 samples
from a later rally. The camera view and court calibration are the same.
This is a different rally in the same match, not a different camera or venue.

Manual references: 14 timestamps, 22 visible-foot contacts, 2 subjective
airborne ground estimates, and 4 explicit skips. Skips are excluded. The
airborne estimates are shown separately and are not used as grounded accuracy.

| Track | Visible-foot mean error | Within 0.50 m | Coverage |
|---|---:|---:|---:|
| v3 | 0.412 m | 14/22 | 22/22 |
| Pose replacement | 0.361 m | 17/22 | 22/22 |
| Hybrid (65% pose) | **0.242 m** | **20/22** | **22/22** |

The hybrid's Near mean is 0.198 m versus 0.445 m for v3. Its Far mean is
0.287 m versus 0.378 m for v3. Among the 18 visible references where a pose
was accepted, the hybrid improved 13 and worsened 5. On exactly those 18
references, v3 mean error is 0.424 m and hybrid mean error is 0.218 m.
The largest hybrid increase in error is 0.347 m at 37.5 s (Far).

Pose acceptance across all 40 sampled times: Near 40/40; Far 22/40. Far
rejections: 8 no pose, 7 weak pose match, and 3 no accepted v3 ground update.
All rejected poses fall back to v3.

The two subjective airborne estimates show the main safety issue. At 37.2 s
and 39.9 s (Near), the script accepted pose midpoints even though the manual
ground state was marked airborne. Hybrid error against these subjective
estimates rose from 0.319 to 0.573 m and from 0.137 to 0.654 m, respectively.
`ground_updated=True` is therefore not enough to prove ground contact.

Interpretation: The hybrid gain repeats on a second rally, but the test still
uses one match and one camera. The labels also have visual uncertainty.
Keep the current settings frozen for this comparison. Before turning this into
a stateful v4 tracker, add a way to detect or abstain on airborne feet and
test on a different match or camera.

Details: `evaluation_summary.json`, `tracker_errors.csv`, and
`worst_cases.jpg` in this folder. In the image, cyan is the manual reference,
magenta is v3, yellow is the pose point, and green is the hybrid point.
Airborne reference markers are projected subjective estimates. The raw labels are in
`../manual_evaluation.json`.
