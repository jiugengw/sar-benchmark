# ViT + LoRA + SAR CNN, 48 epochs, seed 42

The packaged validation ZIP, [`submission_val.zip`](../../nonsubmissions/vit_lora_sar_cnn_48ep_seed42/submission_val.zip), was never submitted. It contains
`classification.csv` and `detection.json` at the ZIP root, with predictions for
all **1,426 official validation images**. The archive has 29,290 detections.
Classification labels are 0–8; detection categories are 1–9; boxes use original
image coordinates `[x, y, width, height]`.

Official Codabench scores are **pending**. The scores below come from a separate
local holdout of 1,408 original images drawn from the labeled training split.
All three local runs use this same holdout, without random augmentation.

| Model / training | Macro-F1 (%) | mAP (%) | AP50 (%) | AP small (%) |
|---|---:|---:|---:|---:|
| ViT + LoRA, 24 epochs | 94.42 | 20.29 | 46.58 | 18.35 |
| ViT + LoRA, continued to 48 epochs | 94.52 | 26.55 | 55.45 | 23.32 |
| **ViT + LoRA + SAR CNN, 48 epochs** | **94.83** | **29.15** | **59.83** | **25.90** |

The CNN run uses ImageNet-pretrained ViT-B/16, rank-64 LoRA (alpha 128),
a 384-channel feature pyramid, and a lightweight CNN that fuses local features
at strides 4 and 8. Total parameters: 126,038,011; trainable: 39,996,667.
The best combined Macro-F1 + mAP checkpoint is epoch 48.

Training uses 7,984 original training images plus 3,992 saved augmentation
variants, with online SAR augmentation, effective batch 256, and seed 42.
The CNN model was trained for a fresh 48-epoch schedule. The continuation run
uses a second learning-rate stage after epoch 24, so this comparison does not
isolate the architecture change alone.

Inference uses this single CNN checkpoint, one deterministic 512×512 view per
image, without TTA, ensembles, or tiling. Export job 10134 completed successfully
in 18 seconds on one Slurm GPU. Predictions retain the model's detection labels
and confidence scores without label calibration against hidden validation data.

Files:

- `submission_val.zip`: moved to `nonsubmissions/vit_lora_sar_cnn_48ep_seed42/`; never submitted.
- `submission_val.metadata.json`: inference counts, parameter counts, and timing.
- `local_validation_metrics.json`: full local holdout metrics at epoch 48.
- `local_comparison.csv`: local scores for the three training runs.
- `training_config.yaml`: training settings, with paths relative to the original
  `cv_project` workspace.
- `export_manifest.json`: checkpoint and file hashes, source fingerprints,
  validation split identity, and validation checks.

The source experiment is `cv_project/experiments/02_vit_lora_cnn48/`; its model
implementation and checkpoint remain there. This directory packages predictions
and results rather than a replacement for this repository's training entrypoint.
