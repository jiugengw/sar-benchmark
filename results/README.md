# Saved experiment results

Results are organized by model configuration, for example:

- `terramind_pretrained_moelora/`
- `vit_pretrained_moelora/`
- `terramind_pretrained_lora/`

`train.py` automatically archives new results under `<backbone>_<init>_<adapt>/seed_<seed>/` when training finishes.
For example, a seed-1 ViT + LoRA run saves to `results/vit_pretrained_lora/seed_1/`. The `--results` option chooses a
different parent directory. Reusing a configuration, seed, and results parent replaces that saved result; choose
a distinct parent for separate hyperparameter experiments. Saved examples may have their JSON files directly in
the configuration folder; point inspection tools at whichever folder contains the three JSON files.

Each directory preserves these files for Git and reproducible detection inspection:

| File | Purpose |
|---|---|
| `metrics.json` | Training configuration and local holdout scores. |
| `predictions_val.json` | Final model's detection predictions on that labeled holdout. |
| `instances_val_holdout.json` | Correct COCO image metadata and annotations for the same holdout. |

These predictions are for the local training holdout; the submission ZIPs contain predictions for the separate
course validation split. Detailed logs remain saved locally under `runs/`. Model weights and their configuration
are preserved separately under [checkpoints/](../checkpoints/README.md), using Git LFS for the weights.

The `vit_pretrained_lora/` directory contains only historical metrics. Detection overlays also require the
prediction and annotation files listed above.

To regenerate the HTML report from the preserved results and extracted dataset:

```bash
python inspect_detections.py \
  --runs results/terramind_pretrained_moelora results/vit_pretrained_moelora \
  --data data/course-data --out reports/detection-review
```

Add other result directories to `--runs` to compare more models. All compared runs must use the same labeled
holdout. The HTML and annotated images are regenerated artifacts; the JSON files here, the training images, and
`inspect_detections.py` are their inputs.

Open `reports/detection-review/index.html` locally, or follow the [browser and SSH viewing instructions](../docs/WORKFLOW.md#view-the-report).
The generated HTML and images are ignored by Git and can be recreated from these inputs.

The files here are visible to Git and are included in repository history after a commit.

## ViT + LoRA + SAR CNN, 48 epochs

[`vit_lora_sar_cnn_48ep_seed42/`](vit_lora_sar_cnn_48ep_seed42/) contains the
Codabench validation submission ZIP for all 1,426 course validation images,
inference provenance, and local holdout comparisons. Its local Macro-F1 / mAP
are 94.83% / 29.15%; official Codabench scores are pending. See that directory's
README for the distinction between local and official validation results.

## Detection experiment comparison

The five `terramind_pretrained_moelora_<experiment>_seed0/` folders preserve final metrics and holdout detection artifacts. See the [submission comparison](../nonsubmissions/README.md#detection-experiments) for each change's rationale, scores, and corresponding ZIP. Shared data diagnostics and completion records live in `detection_experiments/`.
