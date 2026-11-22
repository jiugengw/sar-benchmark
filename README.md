# CS701 SAR benchmark

Team 06's project for the CS701 SAR multi-task challenge (SMU): classify each 512 px SAR image into one of nine
scenes and detect every object in it. Companion repository: [sar-dataset-audit](https://github.com/jiugengw/sar-dataset-audit),
a leakage audit of the public SAR benchmarks the challenge data come from.

## Results

| Test set | macro-F1 | mAP | Δm |
|---|---:|---:|---:|
| Phase 1 (1,426 images) | 95.40 | 59.96 | +65.85 |
| Phase 2 (2,065 images) | 96.11 | 61.21 | +71.57 |
| Copy-free re-split of all labelled images, pipeline and reference both retrained | 94.61 | 44.53 | +68.41 |

Δm = 50 × [(F1 / F1<sub>ref</sub> − 1) + (mAP / mAP<sub>ref</sub> − 1)] against the course reference (a frozen
TerraMind ViT with LoRA and a Faster R-CNN head). One network, one forward pass, no ensembles, no test-time augmentation.

## Method

1. **Detector:** YOLO26m with an extra P2 level and a stride-1 first convolution, so 8 px objects are seen at full
   resolution. Every weight is trained, starting from Ultralytics' DOTA weights.
2. **Staged training:** 100 epochs with full augmentation, then three 40-epoch stages (lighter augmentation, simulated
   speckle, rare-class resampling), each restarting the learning rate.
3. **Exact refinement and weight soup:** six 20-epoch runs without augmentation (TAL β = 7–12), averaged into one model.
4. **Box post-processing:** rounding to whole pixels, ±8% hedge copies and low-score boxes, all ranked below the
   original boxes.
5. **Scene classification:** two light heads on the frozen detector, averaged.

Full recipe: [docs/BEST_MODEL_RECIPE.md](docs/BEST_MODEL_RECIPE.md). Weights: release
[`model-flipsoup-twohead`](https://github.com/jiugengw/sar-benchmark/releases/tag/model-flipsoup-twohead)
([final_model/README.md](final_model/README.md)).

## Reproduce

```bash
gh release download model-flipsoup-twohead --pattern '*.pt' --dir final_model
python models/yolo/make_submission.py --final --out final.zip   # GPU; rebuilds the Phase 1 submission ZIP
python scripts/score_released.py final.zip val                    # scores it on the released labels
```

Setup, data and the commands for each training stage: [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md).

## Documents

- [docs/HISTORY.md](docs/HISTORY.md): every experiment, with configs, job IDs and scores.
- [docs/RELEASED_LABELS.md](docs/RELEASED_LABELS.md): the final model on the released val and test labels, per class.
- [docs/LEAKFREE_SPLIT.md](docs/LEAKFREE_SPLIT.md): 47.7% of the official test repeats training images; the copy-free
  re-split ([results/leakfree/split.csv](results/leakfree/split.csv)) and the retrain on it.
- [docs/NOT_ALLOWED_TRICKS.md](docs/NOT_ALLOWED_TRICKS.md): label-transfer diagnostics; not part of the method.
- [submissions/README.md](submissions/README.md): every uploaded ZIP and its official score.

## Layout

```
models/yolo/     detector training, heads, post-processing, make_submission.py
models/terramind course reference model (ViT / TerraMind + LoRA + Faster R-CNN)
sarbench/        data, metrics and evaluation shared by the models
sbatch/          Slurm jobs (sbatch/staging/: the jobs behind the final model and the re-split)
scripts/ tools/  scoring, reports and cluster helpers
docs/            recipe, history and analyses
results/ submissions/ nonsubmissions/   saved predictions, scores and ZIPs
```
