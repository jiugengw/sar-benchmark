# CNN and ViT small-anchor experiments

## Completed overnight results — 16 October 2026

Jobs **326077** (ResNet34) and **326154** (TerraMind) completed training,
calibration and submission packaging successfully. Both ran all 48 epochs.
The table uses the selected checkpoint and selected calibration on the local
939-image holdout; these are not official leaderboard scores.

| Model | Selected epoch | Macro F1 | mAP | Delta m | Calibration power | Job duration |
|---|---:|---:|---:|---:|---:|---|
| ResNet34, small anchors, fixed task weights | 42 | 93.94% | 32.19% | +11.93 | 0.5 | 3h 41m |
| TerraMind MoE-LoRA, small anchors, uncertainty weighting | 44 | 92.51% | 36.26% | +18.96 | 0 | 6h 43m |

ResNet34 before calibration scored 32.15% mAP and +11.86 Delta m. TerraMind's
selected calibration leaves detection scores unchanged. Compared with the
previous 48-epoch large-anchor TerraMind candidate (93.45% F1, 35.54% mAP,
+18.08 Delta m), the new TerraMind model improves mAP by 0.72 percentage points
and Delta m by 0.88 while F1 falls by 0.93 points. Anchors and task weighting
both changed, so this does not isolate either change's effect.

Artifacts are archived under `results/small48-resnet34-326077/`,
`checkpoints/small48-resnet34-326077/`, `results/small48-terramind-326154/`
and `checkpoints/small48-terramind-326154/`. Results include holdout predictions,
annotations, split, training history and compute records. Model weights use Git
LFS. Calibration metrics, inference records and ZIPs are under the corresponding
`submissions/151026/tests-no-submit/small48-*/` directories. These ZIPs have not
been uploaded to Codabench.

Additional jobs submitted for comparison:

| Job | Script | Configuration |
|---|---|---|
| 326397 | `sbatch/terramind_64_baseline.sbatch` | 64 epochs, default anchors, box loss 1, fixed task weights, no photometric augmentation |
| 326398 | `sbatch/terramind_48_default_uncertainty.sbatch` | 48 epochs, default anchors, box loss 2, uncertainty weighting, photometric augmentation |
| 326894 | `sbatch/terramind_48_baseline.sbatch` | 48 epochs, same settings as the 64-epoch baseline |

All use pretrained TerraMind + MoE-LoRA, seed 0, batch size 16, LR 1e-4,
validation every two epochs and best-checkpoint selection by local Delta m.
Each script calibrates and packages predictions after training. The baseline
pair differs in total schedule length: cosine decay reaches zero at its own
final epoch, so epoch 48 of the 64-epoch run is not equivalent to a standalone
48-epoch run. These three jobs were unfinished when this record was written;
their completed artifacts will be archived separately.

## Updated decision: TerraMind learned weighting replaces the queued ViT run

At the user's request, queued ViT job **326086 was cancelled before training**.
Replacement **326154** runs TerraMind + MoE-LoRA, `small_extreme` anchors,
48 epochs, box-loss weight 2 and photometric augmentation with the same seed-0
90/10 holdout. CNN job **326077** is unchanged. The two original large-anchor ViT
jobs **325636_0/1** are unchanged. The replacement depends on `afterany:325636_0`.

The replacement uses `--loss-weighting uncertainty`. Two training-only scalars
`s_cls`, `s_det`, initialized to zero, optimize:

```text
L_det = L_objectness + L_rpn_box + L_roi_class + 2 * L_roi_box
L = exp(-s_cls) * L_image_class + exp(-s_det) * L_det + (s_cls + s_det) / 2
```

This is a Kendall-style two-task uncertainty objective, adapted to a composite
detection loss; it is not an exact likelihood derivation for the mixed detector
losses. Source: [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).
The scalars train with AdamW at head LR 1e-4 and the same warmup/cosine schedule,
with zero weight decay. The linear log-variance term penalizes trivially zeroing
the positive weights. No validation labels or Δm gradients update these scalars.
They do not guarantee the optimum validation metric or a particular F1/mAP tradeoff.

Counts: **106,517,624 prediction parameters**, **21,169,786 trainable parameters**
including the two scalars. They are excluded from prediction and saved separately
as `loss_weighting.pt` for the selected checkpoint and `final_loss_weighting.pt`
for the last epoch. `training_history.json` records mean component losses,
composite detection loss, objective, log variances and effective weights every
epoch. Validation history records the corresponding effective weights. These
records are archived with metrics/checkpoints and the latest values are also
included in compute heartbeats.

Log: `runs/small48-326154.out`; run directory:
`runs/small48-terramind-326154/terramind_pretrained_moelora_uncertainty/`.
Corresponding results/checkpoints and test-only ZIPs use the same job identifier.
Checks: 23 passed, including objective gradients, finite stationary scales,
loss-weight checkpoint reconstruction and existing CNN/anchor/calibration checks.
Cached pretrained TerraMind loading and actual parameter counts were checked.

The fixed-weight TerraMind small-anchor baseline remains a planned experiment;
it has not been submitted in this session. Comparing this replacement directly
with the old default-anchor model changes both anchors and task weighting, so it
cannot attribute a gain to task weighting alone.

## Earlier submission record (ViT job subsequently cancelled)

These are new runs, not measured improvements. Both use seed 0 and the existing
8,453/939 training/holdout split, 48 epochs, batch size 16, AdamW, head LR 1e-4,
weight decay 0.05, one warmup epoch followed by cosine decay, photometric
augmentation, 512×512 grayscale stretched inputs and the existing normalization.
RoI box-loss weight is 2; evaluate every two epochs and select highest local Δm.

Submitted on 15 October 2026:

- CNN: **326077**, initially pending `QOSMaxJobsPerUserLimit` while the two old
  ViT runs remain active.
- ViT: **326086**, dependency `afterany:325636_0`, initially pending `Dependency`.

Logs: `runs/small48-326077.out` and `runs/small48-326086.out`. Outputs use
`small48-resnet34-326077` and `small48-vit-326086` under runs/results/checkpoints;
ZIPs use those names under `submissions/151026/tests-no-submit/`.

| Backbone | Pretraining | Adaptation | Backbone LR | Total / trainable parameters |
|---|---|---|---:|---:|
| ViT-B/16 | ImageNet-21k, recovered frozen backbone | MoE-LoRA | 1e-4 | 107,211,128 / 21,169,784 |
| ResNet-34 | torchvision ImageNet-1k V1 | Full CNN fine-tuning | 2e-5 | 38,460,280 / 38,460,280 |

The CNN uses native stride-4/8/16/32 feature maps and torchvision FPN. Its final
stride-32 map also feeds the image classifier. The pretrained RGB first
convolution is summed into one grayscale input channel, and its original
ImageNet classifier is discarded. Both heads/FPN are initialized for our task.
No pretrained SAR detector or external SAR training data are used.

These experiments differ in architecture, pretraining data and adaptation/LR;
they compare complete training setups and do not isolate architecture alone.

Both use `small_extreme` anchors:

- Stride 4: 6, 8, 11 pixels.
- Stride 8: 16, 23 pixels.
- Stride 16: 32, 45 pixels.
- Stride 32: 64, 91 pixels.
- Stride 64: 128, 181 pixels.
- All levels: height/width ratios 0.25, 0.5, 1, 2, 4.

The shared RPN convolution feeds separate prediction outputs for 15 anchors per
location on stride 4 and 10 on the other levels. Six pixels is not added to
coarser levels. The experimental large profile (181/256 on stride 64) is not used.
Actual anchor configuration is preserved in metrics and compute records.

Official CNN source: https://download.pytorch.org/models/resnet34-b627a593.pth

SHA256: `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`.

Launch with `sbatch sbatch/small_anchor_48.sbatch resnet34` or
`sbatch sbatch/small_anchor_48.sbatch vit`. Each requests one RTX PRO 6000 Blackwell Max.
Use scheduler dependencies when needed to stay within three concurrent GPUs.
Runtime/peak-memory/provenance records are saved automatically. Calibration and
submission generation run after training; ZIPs go into `tests-no-submit` and are
not uploaded to Codabench.

CPU checks: 21 tests passed across new anchor/CNN training and checkpoint
reconstruction, existing detection configurations, partial fine-tuning and
calibration. Official ResNet weights and grayscale adaptation are checked
separately before submission. Full-resolution GPU execution is checked by the
actual training job; duration and memory are not yet measured for this CNN.
