# DenoDet V2 epoch-32 snapshot

Submission: [denodet-v2-329477.zip](denodet-v2-329477.zip), containing predictions
for all 1,426 course validation images. Official scores are pending.

The best checkpoint at the end of the shortened recovery run was epoch 32:
local holdout macro-F1 92.6610%, detection mAP 33.7505%, Delta m 14.2341.
On the 720-image clean holdout, macro-F1 is 90.8584%, mAP 30.0697%, and
Delta m 6.2251. These are local measurements, not official scores.

The original 48-epoch training job `329477` failed while saving epoch 20 because
disk quota was exceeded. Recovery job `329870` stopped intentionally at epoch 32
while retaining the original 48-epoch learning-rate schedule. Job `330220` was
subsequently submitted to resume from epoch 32 through epoch 48. This directory
preserves the earlier ZIP and metrics while that continuation runs.

The model uses ImageNet-only initialization, a shared image head, no ensemble,
and no test-time augmentation. Checkpoint paths and full settings are in
`metrics.json`; model weights remain under local `runs/`. Matching archived
holdout artifacts are in
[`results/denodet-v2-329477-epoch32/`](../../../results/denodet-v2-329477-epoch32/).
