# ViT + LoRA + SAR CNN, 96 epochs (48 + 48 continuation), seed 42

Upload **`submission_val.zip`** to the **Codabench validation phase**. The ZIP contains
`classification.csv` and `detection.json` at its root, covering all **1,426 official
validation images**, with **21,239 detections**. Classification labels are 0–8;
detection categories are 1–9; boxes use original-image `[x,y,width,height]` coordinates.

**Official Codabench scores are pending.** This candidate has been exported and
published to GitHub; it has not been submitted to Codabench by this export workflow.
The local scores below use 1,408 original images held out from labeled training data,
with deterministic preprocessing and no random augmentation. They are not official
leaderboard scores, and local mAP 33.03% is not a Codabench Score/Delta-m of 33.03.

| Model | Best epoch | Macro-F1 (%) | mAP (%) | AP50 (%) | AP75 (%) | AP small (%) |
|---|---:|---:|---:|---:|---:|---:|
| Parent ViT + CNN, 48 epochs |48|94.83|29.15|59.83|24.96|25.90|
| **ViT + CNN, 48+48 continuation** |**96**|**94.92**|**33.03**|**63.57**|**30.22**|**28.60**|

Continuation improves local mAP by 3.89 percentage points, AP75 by 5.26 points and
AP small by 2.71 points over its parent checkpoint. Both use the same split and
expanded training data. This is one seed 42 continuation result; it is not a
three-seed validation of 96-epoch training.

The architecture is ImageNet-pretrained ViT-B/16 + rank-64 LoRA (alpha 128), a
384-channel feature pyramid, a 110,274-parameter CNN branch fused at strides 4/8,
and a 1300-wide RoI head. Total parameters: 126,038,011; trainable: 39,996,667.
The best combined Macro-F1+mAP checkpoint is epoch 96.

Training uses 7,984 original training images plus 3,992 saved augmentation variants,
with online flips, intensity changes, speckle noise and background cutout.
The parent stage trains 48 epochs at peak LR 4e-4 with 2 warmup epochs and cosine decay.
The continuation restores model/Adam/RNG from epoch 48 and runs epochs 49–96 at peak
LR 1e-4 with 1 warmup epoch and a new 48-epoch cosine stage. Effective batch is 256
(micro128 × accumulation2). This differs from a fresh 96-epoch cosine schedule.

Inference uses a single checkpoint and a single deterministic 512×512 view per image.
No TTA, ensemble, tiling or label calibration is applied. The frozen experiment 07
inference modules load the full checkpoint strictly. Export job 10217 completed
successfully on one Slurm GPU in 65 seconds, including process/import overhead.

| File | Purpose |
|---|---|
| `submission_val.zip` | The file to upload to Codabench. |
| `submission_val.metadata.json` | Checkpoint hash/epoch, prediction counts, parameters and inference timing. |
| `local_validation_metrics.json` | Full original local-holdout metrics for epoch 96. |
| `local_comparison.csv` | Local best-checkpoint comparison across our training experiments. |
| `training_config.yaml` | Continuation configuration; epochs 96, schedule start 48. |
| `parent_training_config.yaml` | The preceding fresh 48-epoch training stage. |
| `export_manifest.json` | Provenance, source/data/file hashes and submission validation checks. |

Configuration paths are relative to the original `cv_project` workspace.
The source experiment is `cv_project/experiments/07_vit_lora_cnn_continue96/`;
its source snapshot and checkpoint remain in that workspace. This directory packages
competition predictions and results, independently of this repository's training entrypoint.
