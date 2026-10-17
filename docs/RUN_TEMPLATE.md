# What every run must save

Back to the [README](../README.md). Use this checklist for every new training run and every packaged ZIP, so each one
can be graphed, compared and reproduced. `python tools/plot_training_curves.py --check` fails while any ZIP is
missing items 7 or 8; a gap explained by `missing` in its `run.json` only warns.

## Training run: in `results/<run>/seed_0/` (committed)

| # | File | What it holds | Why |
|---|---|---|---|
| 1 | `config.json` or `args.yaml` | Every setting: model, epochs, batch, learning rates, augmentation, seed, init weights | Reproduce the run |
| 2 | `split.json` | The train/holdout image ids | Same holdout for every comparison |
| 3 | **Per-epoch history**: `training.csv` (Ultralytics `results.csv`) or `metrics.json` `val_history` + `training_history.json` | For **every epoch**: training loss, **validation loss**, validation mAP, mAP50, macro-F1, Δm | Curves: is it still learning or overfitting? |
| 4 | `predictions_val.json`, `instances_val_holdout.json`, `image_classification_predictions.json` | Holdout boxes, ground truth and per-image classes | Re-grade (full and clean) and the holdout viewer |
| 5 | `metrics.json` | Final scores on the full holdout **and** the clean subset (`clean` / `clean_holdout`), parameter counts | Compare runs fairly |
| 6 | `compute.json` | GPU, hours, peak memory | Course report |

Validate every epoch (not every 4), and log validation loss as well as metrics. Without validation loss the curves
can only show overfitting indirectly. Copy the per-epoch history out of `runs/`, which is not committed, into the
results folder before the job ends.

## Packaged ZIP: beside `<name>.zip` in `nonsubmissions/<DDMMYY>/<name>/` (packaging date; move to `submissions/<name>/` once uploaded)

| # | File | What it holds |
|---|---|---|
| 7 | `<name>.run.json` | Where the training history is (`histories`), or the parent package's `run.json` (`parent`) for post-processed or classifier-only packages; `missing` explains a gap |
| 8 | `<name>.curves.png` | Drawn by `python tools/plot_training_curves.py` |
| 9 | `<name>.inference.json`, `holdout.metrics.json` | Inference settings and timing; local scores of exactly this package |
| 10 | `scoring_result/scores.json` (submissions only) | Codabench's official scores, downloaded after upload |

Example `run.json` for a new detector run, and for a package built on top of it:

```json
{"zip": "yolo26m_p2_new.zip",
 "histories": [{"type": "ultralytics", "path": "results/yolo26m_p2_new/seed_0/training.csv", "label": "detector"}]}
```

```json
{"zip": "yolo26m_p2_new_boxfix.zip", "parent": "nonsubmissions/181026/yolo26m_p2_new/yolo26m_p2_new.run.json",
 "histories": [], "note": "box-edge correction only; no extra training"}
```

History types: `mmengine` (merged per-epoch JSON, with logged training-loss means), `ultralytics` (results.csv), `terramind` (train.py `metrics.json`), `classifier_csv` (image head
`classifier_training.csv`), `quick_head` (`head_comparison.json`). Add a loader in `tools/plot_training_curves.py`
for any new format.

## Before calling a run finished

```bash
python tools/plot_training_curves.py
```

```bash
python tools/plot_training_curves.py --check
```

The first command draws every ZIP's curves and refreshes [TRAINING_CURVES.md](TRAINING_CURVES.md); then run
`python tools/build_curve_compare.py` so the run appears in the comparison page. The second must
report no problems for the new run. Then add a short entry for the run to [HISTORY.md](HISTORY.md): why, what,
result, lesson.
