# Task 3 — calibration tests, not planned submissions

## Default-anchor uncertainty comparison — completed 16 October 2026

Job **326398** completed 48 epochs of pretrained TerraMind + MoE-LoRA with
default anchors, uncertainty task weighting, box-loss weight 2, photometric
augmentation, batch size 16 and seed 0. It uses the same local holdout as the
small-anchor uncertainty run. The selected checkpoint is epoch 48; calibration
selected class-score power 0.5.

| Configuration | Local macro F1 | Local mAP | Local Delta m |
|---|---:|---:|---:|
| Default anchors, uncertainty, calibrated | 92.64% | 35.71% | +17.97 |
| Small anchors, uncertainty, calibrated | 92.51% | 36.26% | +18.96 |
| Default anchors, fixed weights, calibrated | 92.99% | 35.26% | +17.29 |

The default-anchor uncertainty model scored 35.65% mAP and +17.86 Delta m
before calibration. Small anchors improve calibrated mAP by 0.56 percentage
points and Delta m by 1.00 relative to default anchors with uncertainty, while
macro F1 falls by 0.13 points. These are single-seed local comparisons, not
official scores. The small-anchor fixed-weight ablation has not been run.

Ready-to-upload ZIP:
[terramind_48ep_default_anchors_uncertainty_calibrated.zip](tm-default-uncertainty-326398/terramind_48ep_default_anchors_uncertainty_calibrated.zip).
Adjacent `.metrics.json` and `.inference.json` files preserve calibrated scores
and inference provenance. Checkpoints and local holdout artifacts are archived
under `checkpoints/tm-default-uncertainty-326398/` and
`results/tm-default-uncertainty-326398/`. The ZIP has not been uploaded to
Codabench; job duration was 6h 29m, including calibration and packaging.

## Earlier experiment records

The later small-anchor ResNet34 and TerraMind jobs have also completed. Their
ZIPs and score/inference metadata are in `small48-resnet34-326077/` and
`small48-terramind-326154/`. See the
[overnight results](../../../docs/SMALL_ANCHOR_CNN_VIT_20261015.md#completed-overnight-results--16-october-2026)
for the comparison and follow-up jobs. TerraMind reached local mAP 36.26% and
Delta m +18.96; ResNet34 reached 32.19% and +11.93 after calibration. Neither
ZIP has been uploaded. Earlier references below to these runs as planned
describe their status before the overnight completion.

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

### Anchor-shape coverage precheck

Before training another detector, we compared anchor shapes with the labeled
training boxes. Each box was resized to a 512-pixel image and compared with
candidate anchor shapes at the same center. The table reports the percentage
of boxes whose best shape match has IoU at least 0.7. The `narrow-only` profile
adds the tall-thin aspect ratio (5); `wide` adds both extreme ratios (0.2 and
5); `+6 px` adds a 6-pixel size to the default size set; and `wide + 6 px`
combines the wide ratios with that added size.

| Class | Training boxes | Default | Large | Wide | Narrow-only | Default + 6 px | Wide + 6 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aircraft | 2,532 | 99.5% | 91.0% | 99.6% | 99.5% | 99.5% | 99.6% |
| Airport | 919 | 52.2% | 45.5% | 75.2% | 61.7% | 52.2% | 75.2% |
| Bridge | 3,984 | 88.0% | 87.9% | 90.0% | 88.5% | 94.8% | 96.8% |
| Car | 4,089 | 99.6% | 99.6% | 99.7% | 99.6% | 99.6% | 99.7% |
| Harbor | 2,650 | 85.2% | 86.5% | 87.0% | 86.0% | 85.2% | 87.0% |
| Oil tank | 5,580 | 92.6% | 92.6% | 92.9% | 92.8% | 97.9% | 98.3% |
| Playground | 2,610 | 97.0% | 97.0% | 97.0% | 97.0% | 97.0% | 97.1% |
| Ship | 1,544 | 84.0% | 80.4% | 93.0% | 89.3% | 84.0% | 93.0% |
| Wind turbine | 8,625 | 74.7% | 74.7% | 74.7% | 74.7% | 92.4% | 92.4% |

This rough shape-only check suggests wide aspect ratios fit airport and ship
boxes better, while adding a 6-pixel size helps bridge, oil-tank, and
wind-turbine box shapes. The combination improves the theoretical match for
both groups. Playground box shapes already match well, so this check does not
explain their weak detection. The large profile does not improve shape coverage
consistently.

These are not detector recall or AP measurements. The calculation assumes a
perfectly centered anchor and compares sizes and ratios without accounting for
feature-map stride, anchor locations, or training assignments. In particular,
it treats candidate shapes as a pool and does not simulate putting the 6-pixel
size only on the finest feature level. It is a prioritization aid; only a
trained-model comparison can establish whether the change helps.

The trained 24-epoch anchor experiments provide a separate check: on the
official scores, default anchors reached mAP 28.04% and Delta m +2.89; large
anchors reached 27.71% and +1.98; wide anchors reached 28.16% and +2.38. On the
local holdout, the corresponding mAP/Delta m were 27.22%/+0.66, 27.42%/+1.30,
and 27.57%/+1.12. Thus, the theoretical shape-fit gains for wide anchors did
not translate into a clear overall-score gain, and the large-anchor result
was mixed across local and official evaluation.

### Next planned anchor experiment — single 90/10 holdout, not run

Run one 48-epoch TerraMind + MoE-LoRA model using the existing 90/10 split,
with the default anchor sizes and ratios plus ratios 0.25 and 4 and a 6-pixel
size on the finest (stride-4) feature level. Keep box-loss weight 2,
photometric augmentation, and all other training settings the same. Compare
its holdout metrics with the existing 48-epoch default-anchor run, which used
the same split. This is one training run; do not run 5-fold evaluation for
this experiment.

The current shared RPN head assumes the same number of anchors at every
feature level. Putting the 6-pixel size only on stride 4 therefore needs
level-specific RPN prediction outputs (or an equivalent implementation); do
not add 6 pixels to every level just to preserve equal anchor counts. Keep the
experiment within the model parameter limits and record the actual anchor
configuration in the resulting checkpoint and evaluation metadata. The
`small_extreme` profile now implements this configuration with separate RPN
prediction outputs. The TerraMind experiment remains planned, not a trained
result. New CNN and ViT jobs using these anchors are documented in
[the experiment record](../../../docs/SMALL_ANCHOR_CNN_VIT_20261015.md).

## ViT 48-epoch comparison — completed, not submitted

Slurm array **325636** compares pretrained ViT + MoE-LoRA with box-loss
weights 1 and 2. Both tasks use seed 0, 48 epochs, photometric augmentation,
large anchors, and the same data split and schedule. Logs are
`runs/vit48-325636_0.out` and `runs/vit48-325636_1.out`; results and original ZIPs
are under the matching `vit48-325636` directories. Both jobs completed. Copies of
the ZIPs, local score metadata and inference records are also stored here:

| ZIP | Box loss | Selected epoch | Local macro F1 | Local mAP | Local Delta m |
|---|---:|---:|---:|---:|---:|
| [ViT weight 1](vit_48ep_boxloss1_photometric_large_anchors_seed0_calibrated.zip) | 1 | 48 | 93.44% | 32.74% | +12.72 |
| [ViT weight 2](vit_48ep_boxloss2_photometric_large_anchors_seed0_calibrated.zip) | 2 | 44 | 92.11% | 33.44% | +13.34 |

Both selected calibration power 0 (no class-based score adjustment). These are
single-seed local results, not official scores. Weight 2 improves mAP by 0.70
points and overall Delta m by 0.62, while F1 falls 1.33 points. This is a
provisional task tradeoff, not a repeated-seed improvement claim. TerraMind's
matching large-anchor selected checkpoint reaches 93.45% F1 / 35.54% mAP /
+18.08 Delta m, ahead overall on this holdout. Longer ViT training improved
detection compared with historical 24-epoch runs, but anchors, augmentation and
schedule also differ, so this comparison does not isolate epoch count.

The completed paired box review is in [the findings](../../../reports/vit-terramind-48/FINDINGS.md),
with a [filtered overlay gallery](../../../reports/vit-terramind-48/gallery/index.html)
and [contact sheet](../../../reports/vit-terramind-48/comparison-contact-sheet.jpg).
Identical annotations were verified across all three models (939 images,
3,281 objects). At confidence 0.3 and matching IoU 0.5, TerraMind matches
2,436 objects versus ViT loss 2's 2,350; at IoU 0.75 the counts are 1,469
versus 1,324. Both missed objects and box accuracy contribute. These are
diagnostic counts, not a COCO AP decomposition or proof of the underlying cause.

The original frozen ImageNet-pretrained ViT backbone was recovered from the
existing 24-epoch ViT checkpoint because the cluster's offline cache lacked
the Hub weights. `extract_frozen_vit_backbone.py` reproduces
`checkpoints/vit_pretrained_base_seed0.pt`; strict loading confirmed all 150
backbone tensors match. The two completed models each have 107,186,713 total
parameters and 21,145,369 trainable parameters. Keep these runs as experiments;
no submission has been selected or uploaded.
