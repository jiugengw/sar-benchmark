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

## TerraMind 48-epoch runs — not submitted

These two runs came from Slurm array 324733. Both used pretrained TerraMind +
MoE-LoRA, seed 0, a 48-epoch schedule, box-loss weight 2, photometric
augmentation, and the same labeled-training 90/10 split. They differ only in
anchor profile. Checkpoints were selected by local Delta m, then calibrated
using the same holdout. Neither ZIP has been uploaded.

| ZIP | Anchors | Selected epoch | Local macro F1 | Local mAP | Local Delta m |
|---|---|---:|---:|---:|---:|
| [Default anchors](terramind_48ep_boxloss_photometric_seed0_calibrated.zip) | Default | 46 | 92.99% | 35.26% | +17.29 |
| [Large anchors](terramind_48ep_boxloss_photometric_large_anchors_seed0_calibrated.zip) | Large | 46 | 93.45% | 35.54% | +18.08 |

The matching `.metrics.json` files record the checkpoint path, calibration,
and local scores. The 48-epoch validation score was slightly lower than the
selected epoch-46 score for both runs.

### Holdout predictions and weak-class review

The saved per-image predictions are now included here:

- [Large-anchor predictions](terramind_48ep_boxloss_photometric_large_anchors_seed0_predictions_val.json)
- [Default-anchor predictions](terramind_48ep_boxloss_photometric_seed0_predictions_val.json)
- [Shared holdout annotations](terramind_48ep_instances_val_holdout.json)

For the selected large-anchor checkpoint, the weak AP is a mix of missed small
objects and poor box overlap. On the 939-image holdout, bridge has 359 boxes,
including 278 below 16×16 pixels; playground has 252 boxes, including 159
below that size. In a diagnostic greedy match at confidence 0.3 and IoU 0.5,
302 bridge boxes and 204 playground boxes were unmatched. Of the small boxes,
263 bridge and 128 playground boxes were unmatched. At confidence 0.05, the
diagnostic also marked 622 bridge predictions and 215 playground predictions
as having IoU from 0.1 to below 0.5, indicating localization errors among
predictions that overlap an object. These counts are for error review, not COCO
AP; confidence thresholds change the match counts.

The image-level classifier also swaps the classes: in the selected holdout
confusion matrix, 8 of 123 true bridge images were predicted as playground, and
14 of 109 true playground images were predicted as bridge. This is separate
from the detector's box and category errors.

Example overlays use orange for missed ground-truth boxes, green for matched
ground truth, purple for poorly localized predictions, red for false
positives, and blue for matched predictions:

| Bridge examples | Playground examples |
|---|---|
| [Image 3015](weak-class-review/bridge_3015.jpg) · [Image 5344](weak-class-review/bridge_5344.jpg) | [Image 5995](weak-class-review/playground_5995.jpg) · [Image 4361](weak-class-review/playground_4361.jpg) |

The audit supports testing smaller anchors for small-object recall. Since many
predictions also have loose overlap, a separate higher box-loss experiment
could test localization; the current model already uses box-loss weight 2.
Change one factor per run and compare on the same holdout.

## ViT 48-epoch comparison — running, not submitted

Slurm array **325636** compares pretrained ViT + MoE-LoRA with box-loss
weights 1 and 2. Both tasks use seed 0, 48 epochs, photometric augmentation,
large anchors, and the same data split and schedule. Logs are
`runs/vit48-325636_0.out` and `runs/vit48-325636_1.out`; results and ZIPs will be
written under the matching `vit48-325636` directories when the tasks finish.

The original frozen ImageNet-pretrained ViT backbone was recovered from the
existing 24-epoch ViT checkpoint because the cluster's offline cache lacked
the Hub weights. `extract_frozen_vit_backbone.py` reproduces
`checkpoints/vit_pretrained_base_seed0.pt`; strict loading confirmed all 150
backbone tensors match. The two running models each have 107,186,713 total
parameters and 21,145,369 trainable parameters. Keep these runs as experiments;
no submission has been selected or uploaded.
