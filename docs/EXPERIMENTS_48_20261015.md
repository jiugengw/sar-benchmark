# 48-epoch experiments — 15 October 2026

Submitted array **324733** using `sbatch sbatch/improve_48.sbatch`. Shell syntax checks
passed and 14 relevant detection/calibration tests passed before submission.

The official Delta m target is now **above 15**. The best previous submission
(36-epoch box loss + photometric, checkpoint 34) scored macro F1 92.72%,
mAP 31.92%, and Delta m +10.76 on the official evaluation.

Run `sbatch sbatch/improve_48.sbatch` from the repository root to launch two tasks:

| Task | Setup | Question |
|---|---|---|
| 0 | Stronger box loss + photometric, default anchors | Does a longer training/LR schedule improve the strongest existing setup? |
| 1 | Same setup + large anchors | Do large anchors improve this setup under the same longer schedule? |

Both start new training from pretrained TerraMind with MoE-LoRA; they do not
resume the 36-epoch checkpoints. AdamW starts with base LR 1e-4, uses one epoch
of linear warmup, then cosine decay to zero at epoch 48 instead of 36. The
existing scheduler derives its length from `--epochs`, so no scheduler code
change is needed. Comparison with the 36-epoch run tests both longer training
and the extended decay schedule together.

Keep seed 0, the same labeled-training 90/10 holdout, batch size 16, weight
decay 0.05, uniform class sampling, stretched 512-pixel images, RoI box loss
weight 2, brightness/contrast jitter 0.15, and frozen original backbone blocks.
Validate every two epochs and retain the checkpoint with highest local Delta m.
Then evaluate score calibration powers 0, 0.5, and 1 locally and package each
winner into a separate submission ZIP. No official validation labels are used
for training, checkpoint selection, or calibration. Official improvement above
15 is an objective, not a guaranteed outcome.

Each task requests one Blackwell GPU and an eight-hour limit, with up to two
tasks concurrent. Prior 36-epoch runs took about 4h 48m each; estimate about
6–7 hours per task plus queue time. Outputs use the unique array job ID:
`runs/improve-48-<job>/`, `results/improve-48-<job>/`,
`checkpoints/improve-48-<job>/`, and `nonsubmissions/improve-48-<job>/`.
Log files are `runs/improve-48-<job>_<task>.out`.

The ZIPs are generated automatically but must be uploaded separately for
official scores. The old-model-only calibration task is not repeated.
