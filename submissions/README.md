# Model submission ZIPs

Each ZIP is a separate model's predictions for all 1,426 course validation images. Upload one ZIP at a time to
Codabench. These are predictions on `data/course-data/val/`, separate from the labeled training holdout used for
local metrics.

The first two ZIPs below have been submitted to Codabench. The remaining ZIPs are generated from completed
training runs and ready to upload; they do not yet have official scoring results.

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
