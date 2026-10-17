# YOLO26m-P2 fine-tuning — October 17

The user has six hours from 18:16 Singapore time and two official submissions left.
The target is official Delta m >= 30. Preserve both remaining attempts until a
candidate shows a detection gain through the exact submission pipeline.

## Existing official evidence

| Run | Image macro-F1 | Detection mAP | Delta m | Provenance |
|---|---:|---:|---:|---|
| E5 YOLO26m-P2 | 89.7473% | 39.2409% | +23.1761 | Saved official scoring JSON |
| Phase 3 YOLO26m joint | 92.5600% | 38.7776% | +23.7968 | Saved official scoring JSON |
| RT-DETR submission | 91.45% | 38.59% | +22.85 | User-reported rounded official scores; raw score file unavailable |

Delta m = 50 * (F1 / 0.9331 + mAP / 0.2613 - 2), for metrics on a 0-1 scale.
One percentage point of mAP adds approximately 1.91 Delta-m points; one F1 point
adds approximately 0.54. At F1 95%, mAP 41.335% reaches +30. At E5's existing
official mAP 39.2409%, even F1 100% only reaches approximately +28.67.

## Active experiments

* **329477: DenoDet V2, 48 epochs.** The user explicitly chose to keep it running.
  Export a saved checkpoint before the deadline if its full training has not finished.
* **329644: YOLO26m detector continuation.** Start from E5's own course-trained
  100-epoch checkpoint; train detection alone for up to two 40-epoch blocks.
  Fit the image head afterward with frozen detector features. This avoids image
  classification gradients changing the detector during continuation.
* **329663: YOLO26m stride-2 P1 experiment.** Queued because Slurm allows only two
  concurrent jobs. Retain E5's backbone, neck, and existing P2-P5 heads; add one
  finer detection level. Bridge and wind-turbine boxes have median shortest sides
  of 8.8 and 7.6 pixels on the local holdout.

The completed 329643 branch screen is invalid as a comparison: backend setup
overrode the model flag and evaluated the same branch twice. The corrected exact
export evaluations below explicitly set the NMS switch and assert the effective
branch. Queued replacement screen 329655 was cancelled because the main pipeline
already performs that comparison.

## First intermediate candidate — preserve official submission slots

The E5 spatial image head is a development result, not a recommendation to spend
one of the remaining official attempts. It retains E5's detection and improves
image classification on the unchanged 939-image holdout:

| Chosen branch | Image macro-F1 | COCO mAP | Delta m |
|---|---:|---:|---:|
| One-to-many, NMS IoU 0.7 | 94.1325% | 40.6720% | +28.2670 |
| One-to-one | 94.1325% | 39.1722% | +25.3970 |

The image head concatenates each P2-P5 feature map's mean, maximum, and 2x2 spatial
average pooling, followed by LayerNorm, dropout, a 512-unit hidden layer, GELU,
dropout, and a nine-class output. Training uses original/horizontal/vertical/both
flip views of training images only. Validation and submission use one original
image at 512x512, one backbone forward, one selected detector branch, and one head.
The detector has 21,071,016 parameters and the classifier has 4,347,401, for
25,418,417 total, conservatively below the 40M full-training limit.

Package: `nonsubmissions/yolo26m_p2_finetune_329644/e5_spatial_head/`.
Both exact holdout prediction sets are under `results/yolo26m_p2_finetune_329644/`.
No official score is available for this intermediate candidate.

## Checks

* Syntax checks for the modified Python entry points.
* Legacy linear-head and new spatial-head checkpoint save/load output equivalence.
* Full 939-image validation through the actual submission exporter, including
  original-pixel coordinate conversion, classification ID coverage, and branch assertions.
* Exact-export macro-F1 equals the classifier's best validation macro-F1.
* Synthetic-image warmup avoids processing an actual validation image twice.

## Generalization checks without cross-validation

Freeze these recipes instead of adding further hyperparameter searches on the
same holdout. Compare the complete validation mAP/box-loss curves and per-class
AP, and require gains to persist across nearby epochs. An isolated peak does not
justify an official attempt. Keep the training/holdout split unchanged; a new
split of images E5 already trained on is not a fresh independent test.

`yolo_baseline/paired_holdout_audit.py` resamples images in pairs and stratifies
by image class, so E5 and a candidate always see the same bootstrap samples. It
reuses COCO's exact per-image matching and verifies its accumulation against
COCO before reporting intervals. This quantifies holdout sampling variability;
it does not correct the earlier checkpoint/architecture selection on this
holdout, or predict the difference on the official split.

For the intermediate spatial head, the 200-replicate paired F1 gain interval is
approximately **+1.67 to +4.37 percentage points**. Detection is unchanged within
floating-point variation. That supports a stable classification improvement on
this holdout, but provides no evidence of the detection improvement required
for the target. Preserve the two official attempts.

Input stays at 512x512. No ensembles, test-time augmentation, external SAR
pretrained checkpoints, official validation labels, or official validation
training are used. E5's original generic pretraining and own course training are
retained. Local gains do not establish the official target.

## Completed 40-epoch fine-tuning candidate

| Detection branch | Image macro-F1 | Exact COCO mAP | Local Delta m |
|---|---:|---:|---:|
| One-to-many (selected) | 94.5821% | 44.7547% | +36.3202 |
| One-to-one | 94.5821% | 43.4731% | +33.8678 |

Package: `submissions/yolo26m_p2_finetune_40ep_329644/yolo26m_p2_finetune_40ep_329644.zip`.
The ZIP passed archive integrity and schema checks with all 1,426 unique image labels and 103,065 valid detections.
Official scores supplied by the user: **macro-F1 93.3310%, mAP 44.1919%, Delta m 34.5697**, accuracy 93.6886%, AP50 70.8679%. The target of 30 is exceeded by 4.5697 points. Raw values are saved under the candidate scoring_result directory.

At 20:22 SGT the second YOLO 40-epoch block is running (11 epochs complete), with approximately one hour including export remaining. DenoDet failed at 20:08 while saving epoch 20 due to filesystem quota; epoch 19 is preserved. P1 failed at startup immediately after that slot opened. These two jobs are no longer running or queued. The file-count quota was relieved by moving regenerable Python bytecode caches to temporary storage.

Against the original E5 on the same 939-image holdout, paired bootstrap estimates
a mAP gain of 4.08 percentage points (95% interval +3.28 to +5.01) and an F1 gain
of 3.50 points (interval +2.10 to +5.01). These conditional intervals do not correct
prior model selection or establish the official score.

Published artifact directories use the experiment name `yolo26m_p2_finetune_329644`.
Live jobs retain their original internal paths so ongoing training and monitoring continue.
