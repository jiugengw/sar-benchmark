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
