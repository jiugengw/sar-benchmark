# Saved model weights

Checkpoints are organized as `<backbone>_<init>_<adapt>/seed_<seed>/`. Each folder contains:

| File | Purpose |
|---|---|
| `model.pt` | Learned model weights, saved as a PyTorch state dictionary. |
| `metrics.json` | Model configuration, training settings, and local evaluation scores. |

`train.py` copies both files here after training finishes. The `--checkpoints` option chooses a different parent
folder. Use separate parents for experiments with different learning rates or epoch counts but the same model
configuration and seed; reusing the same checkpoint path replaces its files.

The model code plus `model.pt` and the configuration are needed to make new predictions. The state dictionary
does not include optimizer or scheduler state for resuming training exactly.

`model.pt` files under this directory use Git LFS, configured in `.gitattributes`. To retrieve committed weights
after cloning:

```bash
git lfs install
git lfs pull --include="checkpoints/**/model.pt"
```

To make a submission using saved weights:

```bash
python make_submission.py \
  --run checkpoints/terramind_pretrained_moelora/seed_0 \
  --data data/course-data --split val \
  --out submissions/terramind_pretrained_moelora_seed0.zip
```

Choose an unused output filename; submission generation refuses to overwrite an existing ZIP. Saved result JSON
files live under `results/`, and generated reports live under `reports/detection-review/`.

The original ViT + LoRA model weights are unavailable in this checkout. Its existing submission ZIP and metrics
are preserved separately; they cannot reconstruct the model weights.
