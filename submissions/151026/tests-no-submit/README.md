# Task 3 — calibration tests, not planned submissions

These files preserve tests on existing 24-epoch checkpoints. Task 3 did not
train models. The three new submissions are in the [parent folder](../README.md).

| Existing model | Local macro F1 | Local mAP | Selected score power |
|---|---:|---:|---:|
| Baseline | 90.65% | 27.24% | 0 |
| Stronger box localization | 91.45% | 27.89% | 0 |

Both models performed best without class-based detection-score adjustment.
Keep these ZIPs and metrics as records; they are not part of the three planned
uploads. Scores are from the local holdout, not the official leaderboard.
