# Model submission ZIPs

Each ZIP is a separate model's predictions for all 1,426 course validation images. Upload one ZIP at a time to
Codabench. These are predictions on `data/course-data/val/`, separate from the labeled training holdout used for
local metrics.

Official scores, where available, are preserved in each submission's `scoring_result/` folder. The five detection experiments below are ready to upload and have no official scores yet.

| ZIP filename | Backbone | Adaptation | Seed | Checkpoint / notes |
|---|---|---|---|---|
| [submission_1.zip](submission_1/submission_1.zip) | ViT, pretrained | LoRA | 0 | Original submission, kept under the filename used when uploading. Its model weights are not available in this checkout, so keep this ZIP. |
| [terramind_pretrained_moelora_seed0.zip](terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.zip) | TerraMind, pretrained | MoE-LoRA | 0 | Final epoch 24 checkpoint from `runs/terramind-moelora-seed0/`. |
| [terramind_pretrained_lora_seed0.zip](terramind_pretrained_lora_seed0/terramind_pretrained_lora_seed0.zip) | TerraMind, pretrained | LoRA | 0 | Final epoch 24 checkpoint from `checkpoints/terramind_pretrained_lora/seed_0/`. Ready to upload. |
| [vit_pretrained_moelora_seed0.zip](vit_pretrained_moelora_seed0/vit_pretrained_moelora_seed0.zip) | ViT, pretrained | MoE-LoRA | 0 | Final epoch 24 checkpoint from `checkpoints/vit_pretrained_moelora/seed_0/`. Ready to upload. |
| [terramind_pretrained_moelora_unfreeze2_seed0_epochs12.zip](terramind_pretrained_moelora_unfreeze2_seed0_epochs12/terramind_pretrained_moelora_unfreeze2_seed0_epochs12.zip) | TerraMind, pretrained | MoE-LoRA + last 2 blocks unfrozen | 0 | Final epoch 12 screening checkpoint from `checkpoints/screening-12/partial-last2/terramind_pretrained_moelora_unfreeze2/seed_0/`. Ready to upload. |

`make_submission.py` generates each ZIP from a trained checkpoint. Each submission uses one model, processes
each image once, and uses no ensembling or test-time augmentation.

Every ZIP contains:

- `classification.csv`: header `image_id,label`, with one class label (0–8) per image ID.
- `detection.json`: COCO detections with category IDs 1–9, confidence scores, and `[x, y, width, height]` boxes in
  the original image's pixel coordinates.

Local scores are saved separately in `results/<backbone>_pretrained_<adapt>/metrics.json`. The leaderboard supplies
the official validation scores after upload; the ZIP itself contains predictions rather than scores.

Each model has its own folder containing its ZIP and a `scoring_result/` folder. Upload or move the
downloaded scoring result folder into the corresponding `scoring_result/` location, keeping its original files.
New submissions have a `.gitkeep` placeholder in that folder until scoring results are available.
Local training metrics stay in `results/`.

```text
submissions/
├── submission_1/
│   ├── submission_1.zip
│   └── scoring_result/
├── terramind_pretrained_moelora_seed0/
│   ├── terramind_pretrained_moelora_seed0.zip
│   └── scoring_result/
├── terramind_pretrained_lora_seed0/
│   ├── terramind_pretrained_lora_seed0.zip
│   └── scoring_result/
├── vit_pretrained_moelora_seed0/
│   ├── vit_pretrained_moelora_seed0.zip
│   └── scoring_result/
└── terramind_pretrained_moelora_unfreeze2_seed0_epochs12/
    ├── terramind_pretrained_moelora_unfreeze2_seed0_epochs12.zip
    └── scoring_result/
```

To generate a submission from a completed run:

```bash
python make_submission.py \
  --run runs/terramind-moelora-seed0/terramind_pretrained_moelora \
  --data data/course-data --split val \
  --out submissions/terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.zip
```

## Detection experiments

We tested five separate changes to TerraMind + MoE-LoRA to improve detection while retaining classification performance. The baseline's local bridge AP (2.14%) and playground AP (1.64%) were weak, and AP75 (22.02%) was much lower than AP50 (57.79%). These motivated tests of candidate box shapes, object proportions, class exposure, and box accuracy.

Every run used the same labeled-training 90/10 split, seed 0, 24 epochs, batch size 16, adapter/head learning rates of 1e-4, and weight decay 0.05. Original backbone blocks stayed frozen. Each row changes one factor from the baseline. Held-out labels and the official validation split were not used for training. Each ZIP uses a single model with no ensemble or test-time augmentation.

The table shows final-epoch **local holdout** scores, not official leaderboard scores.

| Experiment / submission ZIP | Change and reason | Macro-F1 | mAP | AP75 | Delta m |
|---|---|---:|---:|---:|---:|
| Existing baseline | Reference configuration | 90.65% | 27.22% | 22.02% | +0.66 |
| [Wide anchors](terramind_pretrained_moelora_wide_anchors_seed0/terramind_pretrained_moelora_wide_anchors_seed0.zip) | Added aspect ratios 0.2 and 5 to 0.5, 1, 2, keeping sizes fixed, to cover elongated objects. | 90.25% | 27.57% | 21.96% | +1.12 |
| [Large anchors](terramind_pretrained_moelora_large_anchors_seed0/terramind_pretrained_moelora_large_anchors_seed0.zip) | Changed the coarsest sizes from 128/181 to 181/256 pixels, keeping ratios fixed, to cover large objects. | 91.11% | 27.42% | 22.25% | +1.30 |
| [Preserve geometry](terramind_pretrained_moelora_preserve_geometry_seed0/terramind_pretrained_moelora_preserve_geometry_seed0.zip) | Resized proportionally and padded to 512×512 to avoid distorting rectangular images and boxes. | 90.40% | 27.12% | 21.95% | +0.33 |
| [Class balance](terramind_pretrained_moelora_class_balance_seed0/terramind_pretrained_moelora_class_balance_seed0.zip) | Sampled images with probability proportional to inverse square root of their class image count, to expose rare classes more often. | 90.37% | 27.40% | 22.46% | +0.85 |
| [Box localization](terramind_pretrained_moelora_box_localization_seed0/terramind_pretrained_moelora_box_localization_seed0.zip) | Doubled only the RoI box-regression loss weight, to encourage more accurate boundaries. | 91.45% | 27.89% | 24.02% | +2.36 |

**Stronger box localization is the best completed candidate:** it improves local Delta m from +0.66 to +2.36 and AP75 from 22.02% to 24.02%, with higher macro-F1 and mAP. Wider/larger anchors and balancing offer smaller gains; preserving geometry alone falls below the baseline. These results support submitting the localization model first, but do not establish an official improvement or achievement of the +11 target. The best recorded official baseline is macro-F1 91.87%, mAP 28.04%, Delta m +2.89, evaluated on different images.

All five runs completed successfully. Seven planned two-factor combinations were not run because the remaining allocation could not fit another capped run. No experiment ZIP was uploaded automatically.

For each ZIP, the matching `results/<folder>/` contains metrics, holdout predictions, and annotations; `checkpoints/<folder>/` contains weights and configuration. For example, the localization [results](../results/terramind_pretrained_moelora_box_localization_seed0/metrics.json) and [checkpoint configuration](../checkpoints/terramind_pretrained_moelora_box_localization_seed0/metrics.json) accompany its ZIP. The same folder name connects all three artifact types.

Shared [data diagnostics](../results/detection_experiments/data_diagnostics.json) and [completion records](../results/detection_experiments/status.tsv) preserve the experiment provenance. Saved configurations retain the original job paths as historical metadata; use the current checkpoint paths when generating new predictions. Model weights use Git LFS.
