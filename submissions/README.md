# Submitted models

Each ZIP is a separate model's predictions for all 1,426 course validation images. Upload one ZIP at a time to
Codabench. These are predictions on `data/course-data/val/`, separate from the labeled training holdout used for
local metrics.

The two ZIPs below have been submitted to Codabench. Unsubmitted experiment ZIPs are removed from this folder;
they can be regenerated from their saved checkpoints when needed.

| ZIP filename | Backbone | Adaptation | Seed | Checkpoint / notes |
|---|---|---|---|---|
| [submission_1.zip](submission_1/submission_1.zip) | ViT, pretrained | LoRA | 0 | Original submission, kept under the filename used when uploading. Its model weights are not available in this checkout, so keep this ZIP. |
| [terramind_pretrained_moelora_seed0.zip](terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.zip) | TerraMind, pretrained | MoE-LoRA | 0 | Final epoch 24 checkpoint from `runs/terramind-moelora-seed0/`. |

`make_submission.py` generates each ZIP from a trained checkpoint. Each submission uses one model, processes
each image once, and uses no ensembling or test-time augmentation.

Every ZIP contains:

- `classification.csv`: header `image_id,label`, with one class label (0–8) per image ID.
- `detection.json`: COCO detections with category IDs 1–9, confidence scores, and `[x, y, width, height]` boxes in
  the original image's pixel coordinates.

Local scores are saved separately in `results/<backbone>_pretrained_<adapt>/metrics.json`. The leaderboard supplies
the official validation scores after upload; the ZIP itself contains predictions rather than scores.

Each submitted model has its own folder containing its ZIP and a `scoring_result/` folder. Upload or move the
downloaded scoring result folder into the corresponding `scoring_result/` location, keeping its original files.
Local training metrics stay in `results/`.

```text
submissions/
├── submission_1/
│   ├── submission_1.zip
│   └── scoring_result/
└── terramind_pretrained_moelora_seed0/
    ├── terramind_pretrained_moelora_seed0.zip
    └── scoring_result/
```

To generate a submission from a completed run:

```bash
python make_submission.py \
  --run runs/terramind-moelora-seed0/terramind_pretrained_moelora \
  --data data/course-data --split val \
  --out submissions/terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.zip
```
