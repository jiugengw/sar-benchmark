# CS701 SAR benchmark

Our work on the CS701 SAR multi-task benchmark: classify each SAR image (9 classes) and detect its objects. Our best
official result is **Δm +36.80**, from `yolo26m_p2_candidate_330230.zip`: a YOLO26m-P2 detector with a separate image
classifier and a per-class box-edge correction.

## What to read

- **[docs/HISTORY.md](docs/HISTORY.md): the full record.** Every attempt, part by part: why we tried it, what we
  ran (configs, job IDs, code), local and official scores, and the lesson.
- **[docs/DELTA35_BRAINSTORM_20261017.md](docs/DELTA35_BRAINSTORM_20261017.md): our plans.** Where the remaining
  mAP is lost, ranked ideas to go further, and a one-change-at-a-time experiment plan.
- **[submissions/README.md](submissions/README.md): official scores.** Every ZIP uploaded to Codabench, with its Δm.
  Packaged ZIPs that were never uploaded are in [nonsubmissions/](nonsubmissions/).
- **[reports/training-comparison/index.html](reports/training-comparison/index.html):** compare runs side by side.
- **[reports/holdout-viewer/index.html](reports/holdout-viewer/index.html): compare runs image by image.** All 939
  holdout images with ground truth and up to three runs' boxes, filters (class, size, clean/twin, missed, false
  positives, A vs B), a live score threshold and zoom. Build its data first with `python tools/build_holdout_viewer.py`
  (needs `data/course-data/`).
- **[reports/curve-compare/index.html](reports/curve-compare/index.html): compare training curves.** Tick any runs
  (or best 5 / all) to overlay validation mAP, F1, Δm and train vs validation loss; rebuild its data with
  `python tools/build_curve_compare.py`.
- **[docs/TRAINING_CURVES.md](docs/TRAINING_CURVES.md): training curves** for every packaged ZIP (train vs validation
  loss, validation mAP / F1 / Δm per epoch), with how to spot overfitting.
- **[docs/RUN_TEMPLATE.md](docs/RUN_TEMPLATE.md): what every new run must save** so it can be graphed and compared.
- **[docs/REPORT_CHECKLIST.md](docs/REPORT_CHECKLIST.md):** what the course report must state, and how to export
  GPU hours.

## Task and rules

- **Input:** one 8-bit grayscale SAR image. **Output:** its class, plus a box, class and score for each object.
  Every image has one class and one or more boxes, all of that class.
- **Classes:** aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- **Data:** [course dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  **Leaderboard and full terms:** [Codabench](https://www.codabench.org/competitions/18271).
- **Rules:**
  - One model: no ensembles.
  - At most 130 M parameters in total (frozen ones count) and at most 40 M trainable, heads included.
  - Each image goes through the model **once**, at ≤ 512 × 512 px: no test-time augmentation.
  - The report states parameters (total and trainable) and compute.
- **Metric:** macro-F1 for classification, COCO mAP for detection, ranked by
  `Δm = 50 × [(F1 / 0.9331 − 1) + (mAP / 0.2613 − 1)]` against the TA's ViT + LoRA reference
  (macro-F1 93.31%, mAP 26.13%). One mAP point is worth about 1.91 Δm and one F1 point about 0.54 Δm.

## Best model: test and rebuild it

`yolo26m_p2_candidate_330230.zip`, official Δm +36.80, 29.7 M parameters. It was built in four steps
([details](docs/HISTORY.md)):

1. E5: YOLO26m with a stride-4 (P2) detection level, 100 epochs.
2. Continued for 40 detector-only epochs (job 329644), one-to-many prediction branch.
3. A new, larger image classifier (the small network that names the image's class), trained while the detector
   stayed frozen, so the boxes did not change. Its hidden layer has 1024 neurons instead of 512 (a layer width, not
   an image size: images stay 512 × 512 px, one pass). It raised holdout macro-F1 from 94.58% to 96.06% (job 330230, `models/yolo/quick_classifier.py`).
4. A per-class box-edge correction, cross-fitted on twin-free holdout images.

Files: the ZIP, `image_classifier.pt` and `box_bias_crossfit.json` are in
[submissions/yolo26m_p2_candidate_330230/](submissions/yolo26m_p2_candidate_330230/). Its holdout predictions are in
`results/yolo26m_p2_candidate_330230/seed_0/`. The detector weights are **not in git**; they are on the cluster at
`runs/deadline-yolo26m-329644/continue_1/weights/best.pt`.

**Score it on the local holdout (CPU, seconds).** This grades the saved predictions on all 939 holdout images and on
the 720 "clean" ones that have no near-duplicate twin in training:

```bash
python -m sarbench.twins score results/yolo26m_p2_candidate_330230/seed_0
```

It should print mAP 0.4558, macro-F1 0.9606, Δm +38.70 (full) and mAP 0.4271, macro-F1 0.9510, Δm +32.68 (clean).
Official Δm usually falls between the two.

**Rebuild the submission ZIP (GPU, cluster).** First predict the 1,426 official validation images with the frozen
detector and the image classifier. Then apply the cross-fitted box correction. The second command writes
`nonsubmissions/best_rebuild/best_rebuild.zip`; compare it with the submitted ZIP.

```bash
python models/yolo/make_submission.py --weights runs/deadline-yolo26m-329644/continue_1/weights/best.pt --classifier-checkpoint submissions/yolo26m_p2_candidate_330230/image_classifier.pt --detection-branch one2many --out runs/best_rebuild_base.zip
```

```bash
python models/yolo/package_box_bias.py --tag best_rebuild --base-zip runs/best_rebuild_base.zip
```

## Code structure

Three model families, each with its own training code. Run everything from the repository root; Slurm scripts are in
`sbatch/`.

| Family | Train | Slurm | Evaluate / package | Docs |
|---|---|---|---|---|
| **YOLO / RT-DETR detectors** with an image-classification head (best: YOLO26m-P2, Δm +36.80) | `models/yolo/train_yolo26s_p2.py` (YOLO26 n–l with a P2 level), `train_100_epoch.py` (YOLO11), `train_rtdetr_p2.py`, `train_image_head.py` (image head on frozen features), `train_joint.py` / `train_rtdetr_joint.py` (Phase 3), `deadline_recovery.py` (detector continuation + spatial head), `train_yolo26m_p1.py` (stride-2 level) | `sbatch/yolo26_all_p2_100.sbatch`, `sbatch/sar_phase2_architectures.sbatch`, `sbatch/phase3_*.sbatch`, `sbatch/deadline_yolo26m*.sbatch`, `sbatch/quick_classifier.sbatch` | `models/yolo/run_experiment.py` (holdout scoring), `models/terramind/make_submission.py`, `quick_box_bias.py` + `package_box_bias.py`, `paired_holdout_audit.py` | [models/yolo/README.md](models/yolo/README.md) |
| **ViT / TerraMind / ResNet34 + Faster R-CNN** (TA reference and our first experiments) | `models/terramind/train.py` (code in `sarbench/`), `models/terramind/run_all.sh`, `models/terramind/cross_validate.py` | `sbatch/terramind_*.sbatch`, `sbatch/small_anchor_48.sbatch`, `sbatch/targeted_small_*.sbatch`, `sbatch/vit_compare_48.sbatch` | `models/terramind/make_submission.py`, `models/terramind/calibrate_submission.py` | [docs/DETAILS.md](docs/DETAILS.md), [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| **DenoDet V2** (SAR-specific FFTResNet-50 + GFL) | `models/denodet/train.py` | `sbatch/denodet_v2.sbatch`, `sbatch/denodet_v2_resume.sbatch` | `models/denodet/export.py`, `export_checkpoint.py` | [models/denodet/README.md](models/denodet/README.md) |

How the best submission (`yolo26m_p2_candidate_330230.zip`, Δm +36.80) was built:

1. **E5 detector:** YOLO26m with a P2 level, 100 epochs from COCO weights: `models/yolo/train_yolo26s_p2.py` via `sbatch/yolo26_all_p2_100.sbatch`.
2. **Continuation:** 40 more detector-only epochs from E5, then a spatial image head on the frozen features: `models/yolo/deadline_recovery.py` via `sbatch/deadline_yolo26m.sbatch` (official Δm +34.57 on its own).
3. **Larger image classifier:** the detector stays frozen (boxes unchanged) while a new classifier with a 1024-neuron hidden layer (wider than the old 512; input images stay 512 × 512) learns the image class from its features: `models/yolo/quick_classifier.py` via `sbatch/quick_classifier.sbatch`.
4. **Box-edge correction:** per class and image width, fitted on clean holdout images: `models/yolo/quick_box_bias.py`, packaged with `models/yolo/package_box_bias.py`.

Repository layout:

```
sarbench/             model, data, metrics and twin-aware evaluation used by models/terramind/train.py and the YOLO scorer
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base, ResNet34
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
├── metrics.py        accuracy, macro-F1, COCO box AP
├── twins.py          near-duplicate twins and the clean holdout subset (python -m sarbench.twins)
└── optim.py          learning-rate groups for adapters, heads, and unfrozen backbone weights
models/               the three model families above: terramind/ (uses the root requirements.txt), yolo/ and denodet/ (own requirements.txt)
tools/                detection HTML report, Slurm GPU-hour accounting, one-off backbone diagnostics
scripts/              interactive setup and run helpers (Phase 3, DenoDet)
tests/                CPU tests; dataset-dependent tests need the course data
submissions/          ZIPs uploaded to Codabench, with official scoring results
nonsubmissions/       ZIPs packaged but never uploaded
results/              saved metrics, holdout predictions, and correct annotations
checkpoints/          saved model.pt weights (Git LFS) and model configuration
reports/              comparison reports, galleries and audits
archive/              old code kept for reference; not used
```
