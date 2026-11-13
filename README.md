# CS701 SAR benchmark

This repository has two sections:

- **Course project (below):** our CS701 SAR model: training, submissions and the record of every experiment
  (`models/`, `sbatch/`, `submissions/`, `results/`, `docs/`, `tools/`, ...).
- **[Dataset audit](dataset_audit/README.md):** a separate study of leakage in public SAR detection benchmarks
  (SARDet-100K, RSAR, M4-SAR). It has its own scripts, results and data in `dataset_audit/`.

Our work on the CS701 SAR multi-task benchmark: classify each SAR image (9 classes) and detect its objects. Our final
model scores **Δm +65.85** (macro-F1 95.40, mAP 59.96), from [`submissions/final/final.zip`](submissions/final/):
one YOLO26m-P2 detector with a stride-1 first layer: the weight average of six all-data memorise runs that also see
the images of six classes flipped (as the test set repeats them), with rounded and hedged boxes; the detector's own
low-score boxes, the boxes its NMS removed and wider hedge copies ranked last; and a two-head image classifier on its
frozen features. Our best score, +65.92, added 19 refit epochs (+0.12 before the extra boxes), within run-to-run
noise, so the final model (Phase 1 and Phase 2) leaves them out.

**Teammates: start with [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md)** (setup on the cluster, the exact commands to
rebuild the best model, the rules) and [docs/BEST_MODEL_RECIPE.md](docs/BEST_MODEL_RECIPE.md) (what each stage does).

## What to read

- **[How we got here](docs/HISTORY.md#summary): the path from Δm −1.47 to +65.92** (final model +65.85), one table row per step, and
  [the last week in detail](docs/HISTORY.md#rounding-hedging-soup-stride-1-and-repeated-scenes--2026-october-2026)
  (rounding, hedging, soup, stride 1, the classifier, per-class probes and the flipped copies in the test set).
- **[docs/HISTORY.md](docs/HISTORY.md): the full record.** Every attempt, part by part: why we tried it, what we
  ran (configs, job IDs, code), local and official scores, and the lesson.
- **[docs/DELTA45_PLAN.md](docs/DELTA45_PLAN.md): our plan to reach Δm +45** (draft), followed by the +35-phase analysis. The candidate models are explained [below](#candidate-models-to-run-next). Where the remaining
  mAP is lost, ranked ideas to go further, and a one-change-at-a-time experiment plan.
- **[submissions/README.md](submissions/README.md): official scores.** Every ZIP uploaded to Codabench, with its Δm.
  Packaged ZIPs that were never uploaded are in [nonsubmissions/](nonsubmissions/).
- **[docs/NOT_ALLOWED_TRICKS.md](docs/NOT_ALLOWED_TRICKS.md): tricks we thought of that aren't allowed** (label
  transfer from training images), with their probe scores. Not part of our method.
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

## Candidate models to run next

The rules allow pretrained weights from **optical** images but **not from SAR**. "Optical" means ordinary camera
pictures made from visible light, like the photos a phone takes: colour, sharp edges, no speckle. Our SAR images
are radar echoes instead: microwave reflections, grayscale and grainy. A model pretrained on photos has already
learned general shapes and edges, then we fine-tune it on SAR. That helps, but less than pretraining on the same
kind of image would.

The usual optical pretraining sets:

- **ImageNet:** about 1.3 million everyday photos, each labelled with one of 1,000 things (a dog, a car, a bridge).
  Used to pretrain a *backbone*, the part of a model that turns pixels into features.
- **COCO:** about 118,000 everyday photos with boxes drawn around 80 kinds of object (people, cars, boats,
  aeroplanes). Used to pretrain a whole *detector*.
- **Objects365:** about 2 million photos with boxes for 365 kinds of object; a bigger COCO-like set.
- **AI-TOD and VisDrone:** aerial and drone photos full of tiny objects (AI-TOD's average object is about 13 px).
  Still optical, but seen from above like our SAR images.

**What every model was pretrained on.** "Backbone" is the part that turns pixels into features; "detector" means
the whole model including its box-predicting head.

Models we have already trained:

| Model | Pretrained on | Where in our code |
|---|---|---|
| **YOLO11 / YOLO26** (n–l), incl. the current best YOLO26m-P2 | Whole detector on **COCO** (official Ultralytics weights) | `pretrained/yolo11*.pt`, `pretrained/yolo26*.pt` |
| **RT-DETR-L** (E7) | Whole detector on **COCO** (official Ultralytics weights) | `pretrained/rtdetr-l.pt` |
| **ViT-B/16** (TA reference) | Backbone on **ImageNet-21k** (about 14 million photos, 21,000 labels) | `sarbench/backbones.py` |
| **TerraMind-1.0-base** (TA reference) | Backbone on Earth-observation data from satellites, **including Sentinel-1 radar (SAR)** and Sentinel-2 optical | `sarbench/backbones.py` |
| **ResNet34** | Backbone on **ImageNet-1K** (torchvision) | `sarbench/backbones.py` |
| **DenoDet V2** (ResNet-50) | Backbone on **ImageNet-1K** only; no SAR checkpoint | `models/denodet/` |

TerraMind has seen radar images, but the course rules name it as an allowed generic pretrained model. The TA also
confirmed that public detector checkpoints pretrained on optical aerial photos (AI-TOD, DOTA) count as generic.

Candidates to run next (details in the [Δm +45 plan](docs/DELTA45_PLAN.md)), all within 40M trainable parameters
and one pass at 512 px:

| Model | What it is | Why it might help us | Pretrained on |
|---|---|---|---|
| **YOLO26m-P2 from DOTA** (`sbatch/yolo26m_p2_dota.sbatch`) | Our current architecture, started from Ultralytics' `yolo26m-obb.pt` instead of COCO; all other settings match the 100-epoch COCO run 328339 | DOTA has bridges, harbors, sports fields (ground track field, courts), storage tanks, ships, planes and vehicles seen from above; bridge and playground are our weakest classes (≈ 6 and 4 AP) | Whole detector on **DOTA v1** (optical aerial photos, rotated boxes), official Ultralytics weights |
| **Dome-DETR** (ACM MM 2025, `sbatch/dome_detr.sbatch`) | A DETR (transformer detector) built for tiny objects: it first finds where objects cluster, then spends its attention and box queries there | Designed for objects about our size (AI-TOD averages ≈ 13 px; our median is ≈ 16 px); reports +3.3 AP on AI-TOD-V2 over earlier methods | Backbone (HGNetV2) on **ImageNet**. Released detector weights trained on **AI-TOD-V2 and VisDrone**: aerial and drone photos, optical, so allowed; academic use only. No COCO version mentioned in the paper |
| **DEIM / D-FINE** (M size, ≈ 19M parameters) | DETRs that predict each box edge as a refined probability distribution instead of one number; DEIM is a training recipe that converges faster | Our biggest loss is boxes that are a pixel or two off on tiny objects, and finer edge prediction targets exactly that | Backbone (HGNetV2) on **ImageNet**. Whole detector on **COCO**, or on **Objects365 then COCO**, which the authors recommend as the starting point for fine-tuning |
| **ConvNeXt-T + GFL** | A modern CNN backbone (≈ 28.6M) with the GFL detection head, which scores boxes by their predicted quality | A strong, different architecture family; ConvNeXt-T beat ResNet-50 on the large SAR benchmark SARDet-100K | Backbone on **ImageNet-1K** only. The GFL head starts untrained (no COCO checkpoint for this pairing was found) |

Not allowed: anything pretrained on SAR, such as SARDet-100K/MSFA, SARATR-X or SAR-JEPA weights. Lower priority:
ViT-based detectors such as RF-DETR, whose 14–16 px patches are too coarse for 6–15 px objects.

## Best model: test and rebuild it

`yolo26m_p2_candidate_330230.zip`: official **Δm +36.80** (macro-F1 94.69%, mAP 44.98%), 29.7 M parameters.

**How it runs on one image** (one pass at 512 × 512 px, one model):

```
SAR image ──► detector (YOLO26m-P2) ──► boxes ──► box nudge ──► final boxes
                    │
                    └── its internal features ──► image classifier ──► image class
```

**How it was built**, in three steps. Each step kept everything before it unchanged.

| Step | Part | What we did | Holdout result after this step |
|---|---|---|---|
| 1 | **Detector** | Trained YOLO26m with an extra fine-detail (P2) layer for small objects, starting from COCO weights: 100 epochs (run "E5"), then 40 more with a fresh, lower learning rate (job 329644). 140 epochs in total. | mAP 40.7% → 44.8% |
| 2 | **Image classifier** | Trained a small network that reads the detector's features and outputs the image's class. The detector was frozen, so its boxes did not change. The network's hidden layer is 1024 neurons wide (job 330230). | macro-F1 96.1% |
| 3 | **Box nudge** | Shifted every box edge by a tiny fixed amount learned per class and image size (at most 1 px), because the detector's boxes were slightly but consistently off. Learned on the 720 twin-free holdout images. | mAP 45.6% |

Notes:

- "1024 neurons" is the width of one layer inside the classifier, not an image size.
- YOLO26 has two prediction heads. We use the "one-to-many" head, which proposes several boxes per object and then
  drops duplicates; it scores about 1.3 mAP points higher than the "one-to-one" head.
- Before step 2's classifier, a smaller 512-neuron one on the same detector scored official Δm +34.57. Steps 2 and
  3 together added +2.2.
- The box nudge was tested before use: learning it on one half of the clean images and measuring on the other half
  gained +0.6 to +0.9 mAP points every time.
- The 140 detector epochs were two runs with different settings. A single 140-epoch run has not been tried.

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

1. **Detector, 140 epochs in two stages:** 100 epochs from COCO weights with `models/yolo/train_yolo26s_p2.py` via
   `sbatch/yolo26_all_p2_100.sbatch` (E5), then 40 more with `models/yolo/deadline_recovery.py` via
   `sbatch/deadline_yolo26m.sbatch`.
2. **Image classifier (1024 neurons) on the frozen detector:** `models/yolo/quick_classifier.py` via
   `sbatch/quick_classifier.sbatch`.
3. **Box nudge:** fitted with `models/yolo/quick_box_bias.py`, packaged with `models/yolo/package_box_bias.py`.

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
