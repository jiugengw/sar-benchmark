# Matched YOLO26 medium / large comparison

The first large speckle run (330592) started from 100 baseline epochs and added
40 speckle epochs at learning rate 0.002. The medium speckle run (330447) started
from 100 baseline + 40 plain epochs, then added 40 speckle epochs at 0.001.
Those runs do not isolate architecture size. The large run also reused its old
classifier; its F1 is not a comparison of refitted classifier performance.

The replacement run mirrors the best local medium candidate (`speckle-cand`):

Completed job: **330632**, `sar-y26l-matched`, 18 October 2026, exit code 0,
elapsed **3:07:30**. [Candidate and results](../nonsubmissions/181026/yolo26l-matched-best-330632/),
[completed graphs](../reports/yolo26-matched-size/comparison.png),
[final score graph](../reports/yolo26-matched-size/final-results.png).
The initial submission 330628 was cancelled during its first epoch to correct
the output-name comparison check; its partial checkpoint is not reused.

| Stage | Medium reference | Large replacement |
|---|---|---|
| Baseline | 100 epochs, YOLO26m-P2 | 100 epochs, YOLO26l-P2 |
| Plain continuation | 40 epochs, lr 0.002 | 40 epochs, lr 0.002 |
| Speckle continuation | 40 epochs, lr 0.001, probability 0.5, looks 4 | Same |
| Classifier | Frozen detector; spatial pooling; 1024 hidden units; 240 epochs | Same |
| Box correction | Twin-grouped cross-fit; apply only when every check gains | Same |

All stages use seed 0, batch 16 for detection, the same augmentations, and the
same 8,453 / 939 split. Baseline training arguments and split files were audited
as identical except architecture and output paths. Both baseline best checkpoints
are at epoch 100. The existing medium results are reused. This is a single-seed
recipe comparison; it does not isolate random
initialization effects or prove a general architecture ranking.

Initialization audit, 18 October: the baseline script loaded COCO weights into
an intermediate model, but then discarded them at the training handoff. These
baseline checkpoints must be treated as trained from random initialization.
The continuation stages load their trained `.pt` checkpoints correctly. This
matched-size job continues to compare like schedules and baseline procedures.
Corrected pretrained COCO/DOTA runs are separate experiments; see
[YOLO26_PRETRAINING_AUDIT.md](YOLO26_PRETRAINING_AUDIT.md).

Run `python models/yolo/matched_large.py --tag CHECK_NAME --check-only` to audit
the sources. Submit `sbatch sbatch/yolo26l_matched_best.sbatch` to train large.
Each stage's generated arguments are checked against medium before proceeding.
Intermediate evaluation is deferred until the classifier is refitted.

Primary comparison: exported one2many detection mAP on full and clean holdouts,
before calibration. Also report refitted classifier F1, Delta m, and per-class AP.
Calibrated scores include fit images; inspect the cross-fit evidence separately.
The job saves `results/<tag>/architecture_comparison.json` and a validation ZIP
under `nonsubmissions/<date>/<tag>/`. No official upload is performed.

## Completed results

| Local score | Medium | Large |
|---|---:|---:|
| Uncalibrated full mAP (%) | 45.6923 | 45.1587 |
| Uncalibrated clean mAP (%) | 42.1203 | 42.2190 |
| Refitted full macro-F1 (%) | 95.4847 | 95.1337 |
| Refitted clean macro-F1 (%) | 94.4712 | 93.8595 |
| Calibrated full mAP (%) | 46.3636 | 46.0427 |
| Calibrated clean mAP (%) | 43.2264 | 43.4033 |
| Calibrated full Delta m | +39.8825 | +39.0804 |
| Calibrated clean Delta m | +33.3364 | +33.3471 |

Medium leads on the full holdout; clean detection and calibrated Delta m are close.
This single matched run does not show a clear large-model improvement. Calibrated
scores use fit images; both detectors passed the same disjoint box cross-fit checks.
The large detector checkpoint is preserved through Git LFS at
`checkpoints/yolo26l-matched-best-330632/model.pt`; the classifier and validation ZIP
are in its candidate directory. Large has no official score.

Dome-DETR job 330602 failed before training because the inherited 10-cell attention
windows do not divide the 64-cell feature map at 512 input. Its failed run files
were removed at the user's request. Its retry is deferred until after this comparison.
