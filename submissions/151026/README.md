# Submissions — 15 October 2026

Upload the three ZIPs in this folder separately to Codabench to compare official
scores. They are tasks 0, 1, and 2 of Slurm job array 324156. All use pretrained
TerraMind + MoE-LoRA, seed 0, and completed 36 training epochs. The best checkpoint
was selected using the labeled training holdout. Each ZIP uses a single model,
without ensembling or test-time augmentation.

## Why we ran these experiments

The previous 24-epoch models classified images well but had weaker detection.
We tested longer training, then emphasized bounding-box accuracy, then added
variation in image appearance. These runs use the same local holdout to compare
the effect of each additional change.

| Task / ZIP | Change and purpose | Selected epoch | Local macro F1 | Local mAP | Local Delta m |
|---|---|---:|---:|---:|---:|
| 0: [longer_baseline_calibrated.zip](longer_baseline_calibrated.zip) | Train for 36 rather than 24 epochs to test whether more training helps; box loss weight 1. | 36 | 91.07% | 31.75% | +9.55 |
| 1: [longer_boxloss_calibrated.zip](longer_boxloss_calibrated.zip) | Double the RoI box-regression loss weight to emphasize accurate object boundaries, keeping the longer schedule. | 36 | 91.71% | 32.26% | +10.88 |
| 2: [longer_boxloss_photometric_calibrated.zip](longer_boxloss_photometric_calibrated.zip) | Add brightness/contrast augmentation to stronger box loss to improve robustness to image appearance. | 34 | 92.43% | 32.59% | +11.89 |

All three ran through epoch 36; task 2's best checkpoint was at epoch 34.
Near the end, task 0 still gained a little mAP, task 1 flattened, and task 2
peaked at 34. The learning-rate schedule decayed to zero at epoch 36. These
results do not establish whether a longer schedule would improve further.

## Calibration and task 3

After training, tasks 0–2 tested adjusting detection confidence using class
probabilities, with score powers 0, 0.5, and 1. Power 0 leaves detection scores
unchanged. Settings were selected using labeled training holdout scores, without
using official validation labels. Tasks 0 and 1 selected power 0; task 2 selected
power 0.5, slightly improving mAP from approximately 32.58% to 32.59%.

Task 3 performed the same calibration tests on two existing 24-epoch models.
It did not train any models. Both selected power 0, so the score adjustment did
not help them. Their ZIPs and metrics are preserved in
[tests-no-submit/](tests-no-submit/README.md) as experiment records, separate from
the three planned submissions.

## What to submit

Upload the three ZIPs in this folder one at a time. Task 2 is the strongest
local candidate, but official rankings may differ. Scores above are local
holdout results; official leaderboard scores are pending.

Each ZIP contains `classification.csv` for all 1,426 course validation images
and `detection.json` with COCO detections. Adjacent `.metrics.json` files record
local scores, selected epochs, calibration settings, and checkpoint provenance.
Original job paths inside those records are retained; only this submission
folder was renamed.

The later TerraMind 48-epoch candidates and the running ViT comparison are
preserved under [`tests-no-submit/`](tests-no-submit/README.md). They have not
been uploaded; see that README for settings and local scores.
