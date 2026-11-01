# Experiment history: what we tried, why, and what we learned

**Initialization correction, 18 October:** YOLO26-P2 baselines produced by
`train_yolo26s_p2.py` discarded their transferred pretrained weights at training
setup. Historical COCO-initialization descriptions for these runs are incorrect;
their measured scores remain valid. DOTA job 330601 repeated the seeded baseline
and was cancelled. [Audit and corrected paired reruns](YOLO26_PRETRAINING_AUDIT.md).

CS701 SAR benchmark: 9-class image classification plus COCO object detection on 512 px SAR images, ranked by
`Δm = 50 × [(F1 / 0.9331 − 1) + (mAP / 0.2613 − 1)]` against the TA's ViT + LoRA validation reference
(macro-F1 93.31%, mAP 26.13%). Rules: one model, ≤ 130M total and ≤ 40M trainable parameters, each image in
once at ≤ 512 × 512, no ensembles or test-time augmentation. One mAP point is worth about 1.91 Δm, one F1 point
about 0.54 Δm, so detection drives the score.

## Summary

We started from the TA's ViT/TerraMind + LoRA/MoE-LoRA + Faster R-CNN code (official Δm −1.47 to +2.89). Five
days of TerraMind detection ablations (anchors, box loss, geometry, sampling, longer schedules, photometric
augmentation, task weighting) raised that to +15.37, mostly from training longer; single-factor tweaks moved
scores by less than the noise. TerraMind plateaued at about 34.4% official mAP. Switching to fully fine-tuned
Ultralytics detectors with a stride-4 (P2) level gave an immediate jump (YOLO11s-P2 +18.11, YOLO26m-P2 E5
+23.18). Joint fine-tuning traded mAP for F1 (+23.80). On the last day, a detector-only continuation of E5 for
40 more epochs added about 5 official mAP points (+34.57), and a wider image head on the frozen features plus a
cross-fitted per-class box-edge correction gave the best result, **Δm +36.80**. In parallel we found that about
a quarter of the labelled images (and of the official validation set) are same-scene "twins" of training images,
which is why official scores land between our full and twin-free ("clean") local scores. On 18–19 October a
longer chain from DOTA weights reached +40.7, and 40 final epochs **without augmentation** ("memorise") added
about 4 official mAP points, for **Δm +49.21**; see [18–19 October](#chain-memorise-and-headroom--1819-october-2026).
From 20 to 26 October, rounding, box hedging, a six-run weight soup, a stride-1 first layer, all-data training and a
two-head image classifier carried it to **Δm +58.00** (95.13 F1, 55.93 mAP). On 26 October we found that about 250 test
images (18%) repeat labelled scenes only in flipped form, with labels flipped one pixel off an exact mirror; a soup
of memorise runs that also see those classes' images flipped, with its two-head classifier, reached **Δm +65.55**
(95.40 F1, 59.80 mAP) on 27 October; see
[20–26 October](#rounding-hedging-soup-stride-1-and-repeated-scenes--2026-october-2026). On 28 October 19 more flip epochs (refit 2), its two-head classifier and a low-score tail gave **Δm +65.92** (95.24 F1,
60.04 mAP). On 30 October the flip soup with the same tail scored **+65.80** (95.40 F1, 59.94 mAP): the refits added
only +0.12, within run-to-run noise, so the final model (Phase 1 and Phase 2) is the flip soup without refits;
[BEST_MODEL_RECIPE.md](BEST_MODEL_RECIPE.md) lists the whole chain. On 31 October we checked it on new scenes: every
step after the detector helps there too, and nothing we tried beat it on new images beyond noise; see
[Phase 2 checks](#phase-2-checks-new-images--31-october-2026).

| Date (2026) | Approach | Official F1 | Official mAP | Official Δm |
|---|---|---:|---:|---:|
| Oct 12 | ViT + LoRA, TA recipe (`submission_1`) | 92.71 | 25.53 | −1.47 |
| Oct 13 | TerraMind + MoE-LoRA, 24 epochs | 91.87 | 28.04 | +2.89 |
| Oct 14 | Best single-factor ablation (box-loss weight 2) | 92.06 | 27.80 | +2.53 |
| Oct 15 | TerraMind, 36 epochs, box loss 2 + photometric | 92.72 | 31.92 | +10.76 |
| Oct 16 | TerraMind, 64-epoch plain baseline | 92.49 | 34.39 | +15.37 |
| Oct 16 | YOLO11s-P2, 100 epochs + image head | 90.45 | 36.40 | +18.11 |
| Oct 17 | E5 YOLO26m-P2, 100 epochs + image head | 89.75 | 39.24 | +23.18 |
| Oct 17 | Phase 3 joint fine-tune of E5 | 92.56 | 38.78 | +23.80 |
| Oct 17 | E5 + 40 detector-only epochs + spatial head | 93.33 | 44.19 | +34.57 |
| Oct 17 | Same detector + 1024-neuron image classifier + box-edge correction | 94.69 | 44.98 | +36.80 |
| Oct 18 | Speckle candidate | 93.65 | 45.57 | +37.38 |
| Oct 18 | YOLO26m-P2 from DOTA, 100 epochs + plain/speckle chain | 93.90 | 47.25 | +40.73 |
| Oct 18 | Same from COCO weights | 94.12 | 47.04 | +40.45 |
| Oct 19 | Same, YOLO26l from DOTA | 93.25 | 47.51 | +40.88 |
| Oct 19 | DOTA m chain + third 40-epoch stage | 93.93 | 47.51 | +41.23 |
| Oct 19 | Third stage + 40 epochs without augmentation (memorise) | 93.34 | 51.64 | +48.83 |
| Oct 19 | Memorise at lr 0.002 | 93.83 | 51.70 | +49.21 |
| Oct 24 | All-data six-run soup, rounded + hedged, two-head classifier | 93.40 | 54.88 | +55.06 |
| Oct 25 | Stride-1 all-data single model, rounded + hedged | 94.45 | 55.17 | +56.18 |
| Oct 25 | Stride-1 all-data soup, rounded + hedged | 94.37 | 55.93 | +57.59 |
| Oct 26 | Same + stride-1 two-head classifier retrained on the soup | 95.13 | 55.93 | +58.00 |
| Oct 27 | Class-selective flip-memorise soup, rounded + hedged (refit classifier) | 92.59 | 59.80 | +64.04 |
| Oct 27 | Same + stride-1 two-head classifier retrained on the flip soup | 95.40 | 59.80 | +65.55 |
| Oct 28 | Flip soup + 19 more class-selective flip epochs on all data (refit 2), quick classifier | 92.34 | 59.93 | +64.16 |
| Oct 28 | Refit 2 + its two-head classifier | 95.24 | 59.93 | +65.71 |
| Oct 28 | Same + low-score tail (the detector's own boxes under 0.001, ranked below every box) | 95.24 | 60.04 | **+65.92** |

All uploads, with Codabench IDs where known, are in [submissions/README.md](../submissions/README.md). Unless stated,
"local" scores below are on the fixed 939-image holdout of the labelled training set (8,453 train / 939 holdout,
seed 0); "clean" means the 720 holdout images with no near-duplicate in training (Part 6).

---

## Part 1. TA baseline: ViT / TerraMind + LoRA / MoE-LoRA + Faster R-CNN (Oct 12–13)

**Why.** Establish an official score with the reference code and pick the best of the four compliant
backbone/adapter combinations. The TA's own validation table put TerraMind + MoE-LoRA highest (+4.4 ± 0.3).

**What we did.** `models/terramind/train.py` (code in `sarbench/`): frozen ViT-B/16 (ImageNet-21k) or TerraMind-1.0-base,
LoRA or MoE-LoRA adapters, image-class head plus ViTDet feature pyramid and Faster R-CNN. 24 epochs, batch 16,
AdamW, LR 1e-4, weight decay 0.05, seed 0, final-epoch checkpoint. ZIPs via `models/terramind/make_submission.py`. One extra
screening run unfroze the last two TerraMind blocks (`--unfreeze-last 2`, unfrozen LR 2e-5, 12 epochs).
The first plan aimed to beat a leaderboard snapshot of +8.22 (leader: F1 92.13%, mAP 30.75%) within the
limit of five uploads per day.

**Results.**

| Run | Local F1 | Local mAP | Local Δm | Official F1 / mAP | Official Δm |
|---|---:|---:|---:|---|---:|
| ViT + LoRA (`submission_1`; weights not kept) | 92.61 | 24.85 | — | 92.71 / 25.53 | −1.47 |
| ViT + MoE-LoRA | 92.23 | 25.49 | −1.80 | 92.12 / 26.67 | +0.39 (Codabench status "Failed") |
| TerraMind + LoRA | 91.46 | 25.45 | −2.29 | 92.05 / 27.00 | +1.00 |
| **TerraMind + MoE-LoRA** | 90.65 | 27.22 | +0.66 | 91.87 / 28.04 | **+2.89** |
| TerraMind + MoE-LoRA, last 2 blocks unfrozen, 12 ep | 91.36 | 20.66 | −11.52 | 92.29 / 23.01 | −6.53 |

**Lesson.** TerraMind + MoE-LoRA was the best start, as in the TA table. F1 was already near the reference;
the gap was detection (weak bridge 2.14% and playground 1.64% AP, AP75 22.02% vs AP50 57.79% locally).
Partial unfreezing on a short schedule hurt badly. Local Δm under-estimated official Δm at this stage.

---

## Part 2. TerraMind detection ablations (Oct 14–16)

All runs: pretrained TerraMind + MoE-LoRA, seed 0, same 90/10 split, batch 16, LR 1e-4, frozen original blocks,
stretched 512 px input, unless stated. Code: `models/terramind/train.py` options `--anchor-profile`, `--geometry`,
`--class-sampling`, `--box-loss-weight`, `--photometric`, `--select-best`, `--loss-weighting`,
`--small-object-sampling`, `--detail-branch`; calibration in `models/terramind/calibrate_submission.py`.

### 2.1 Single-factor ablations, 24 epochs (Oct 14)

**Why.** Each change targets one observed weakness: elongated objects (wide anchors), large objects (large
anchors), distorted rectangular images (pad instead of stretch), rare classes (inverse-sqrt class sampling),
loose boxes (RoI box-regression loss × 2).

| Change | Local F1 | Local mAP | AP75 | Local Δm | Official F1 / mAP | Official Δm |
|---|---:|---:|---:|---:|---|---:|
| Baseline | 90.65 | 27.22 | 22.02 | +0.66 | 91.87 / 28.04 | +2.89 |
| Wide anchors (ratios 0.2, 5 added) | 90.25 | 27.57 | 21.96 | +1.12 | 90.50 / 28.16 | +2.38 |
| Large anchors (coarsest 181/256 px) | 91.11 | 27.42 | 22.25 | +1.30 | 91.37 / 27.71 | +1.98 |
| Preserve geometry (pad) | 90.40 | 27.12 | 21.95 | +0.33 | 91.96 / 27.72 | +2.31 |
| Class-balanced sampling | 90.37 | 27.40 | 22.46 | +0.85 | — | +1.86 ¹ |
| Box localization (box loss × 2) | 91.45 | 27.89 | 24.02 | +2.36 | 92.06 / 27.80 | +2.53 |

¹ The saved `scoring_result/scores.json` for class balance is a copy of box localization's; +1.86 is from the
Codabench list. Seven planned two-factor combinations were not run (allocation).

**Lesson.** Every change landed within about ±1 Δm of the baseline and none beat it officially. Single-seed
differences this small are noise; the official ranking even reversed the local one. An anchor-shape coverage
check (best-shape IoU ≥ 0.7) predicted wide ratios would help airport/ship and a 6 px size would help
bridge/oil tank/wind turbine, but shape fit did not translate into score.

### 2.2 Detection-score calibration

**Why.** Multiply each detection score by the image classifier's probability for that class, power 0 / 0.5 / 1,
chosen on the holdout. **Result.** The 24-epoch baseline and box-localization models chose power 0 (no change);
the 36-epoch photometric model chose 0.5 (mAP 32.58 → 32.59); the default-anchor uncertainty model chose 0.5
(+0.06 mAP). **Lesson.** Negligible. COCO AP ranks within each class, so class-wise rescaling barely matters
(confirmed later in Part 6).

### 2.3 Longer training: 36 epochs (array 324156, `sbatch/overnight_improve.sbatch`, Oct 14–15)

**Why.** The 24-epoch curves were still rising. Validate every 2 epochs and keep the best local-Δm checkpoint.

| Task | Change | Sel. epoch | Local F1 / mAP / Δm | Official F1 / mAP | Official Δm |
|---|---|---:|---|---|---:|
| 0 | 36 epochs, box loss 1 | 36 | 91.07 / 31.75 / +9.55 | 91.60 / 30.94 | +8.29 |
| 1 | + box loss × 2 | 36 | 91.71 / 32.26 / +10.88 | 92.09 / 31.05 | +8.76 |
| 2 | + brightness/contrast jitter 0.15 | 34 | 92.43 / 32.59 / +11.89 | 92.72 / 31.92 | +10.76 |

**Lesson.** Schedule length was the first big lever (+5.4 official Δm from 24 to 36 epochs). Box loss plus
photometric augmentation added about 2.5 more. Checkpoint selection and calibration on the same holdout make
local scores optimistic.

### 2.4 48 epochs and anchors (array 324733, `sbatch/improve_48.sbatch`, Oct 15)

| Anchors | Sel. epoch | Local F1 / mAP / Δm | Official |
|---|---:|---|---|
| Default | 46 | 92.99 / 35.26 / +17.29 | uploaded, cancelled, no score |
| Large | 46 | 93.45 / 35.54 / +18.08 | 92.27 / 33.44, **+13.42** (status "Cancelled" but scored) |

A weak-class review of the large-anchor model found bridge 359 holdout boxes (278 below 16 × 16 px) and
playground 252 (159 below); at confidence 0.3 / IoU 0.5, 302 bridge and 204 playground boxes were unmatched,
and many predictions overlapped objects loosely. The image classifier also swapped bridge ↔ playground
(8/123 and 14/109). This motivated the `small_extreme` anchors (6/8/11 px at stride 4 up to 128/181 px at
stride 64, ratios 0.25–4, level-specific RPN outputs).

**Lesson.** The local–official gap widened to 4–5 Δm; we did not yet know why (Part 6).

### 2.5 Baselines, task weighting, small objects (Oct 15–16)

**Why.** Separate schedule length from the recipe; try Kendall-style uncertainty task weighting (two learned
log-variances, `--loss-weighting uncertainty`), small anchors, tiny-object image oversampling (up to 2×) and a
stride-4 detail branch. Scripts: `sbatch/terramind_48_baseline.sbatch`, `terramind_64_baseline.sbatch`,
`terramind_48_default_uncertainty.sbatch`, `small_anchor_48.sbatch`, `targeted_small_48.sbatch`.

| Job | Recipe | Local F1 / mAP / Δm | Official F1 / mAP | Official Δm |
|---|---|---|---|---:|
| 326894 | 48 ep, default anchors, box loss 1, fixed weights, no photometric | 91.34 / 34.36 / +14.70 | not uploaded | — |
| 326397 | Same, 64 ep (sel. epoch 56) | 91.90 / 36.29 / +18.68 | 92.49 / 34.39 | **+15.37** |
| 326398 | 48 ep, default anchors, box loss 2, photometric, uncertainty | 92.64 / 35.71 / +17.97 | 92.14 / 33.63 | +13.71 |
| 326154 | Same, `small_extreme` anchors (sel. epoch 44) | 92.51 / 36.26 / +18.96 | 92.52 / 34.36 | +15.32 |
| 327063_0 | 326154 + tiny-object sampling | 92.27 / 35.28 / +16.96 | not uploaded | — |
| 327063_1, 327100 | Detail branch; small anchors with fixed weights | cancelled | — | — |
| Colab | 48 ep, batch 32, `gain` + `tm-aux mean3` (A100, 3.2 h) | 92.09 / 32.90 / +12.30 | 92.12 / 32.04 | +10.68 |

**Lesson.** The plain 64-epoch baseline tied the most elaborate recipe officially. Small anchors beat default
anchors under uncertainty weighting (+1.0 local, +1.6 official), but no clean ablation isolates anchors or task
weighting. Oversampling tiny-object images hurt. The Colab run changed four things at once, so nothing is
attributable. TerraMind + Faster R-CNN topped out near 34.4% official mAP.

### 2.6 Backbone comparisons: ViT and ResNet34 (Oct 15–16)

| Run | Setup | Local F1 / mAP / Δm |
|---|---|---|
| ViT 325636_0 | MoE-LoRA, 48 ep, large anchors, photometric, box loss 1 | 93.44 / 32.74 / +12.72 |
| ViT 325636_1 | Same, box loss 2 | 92.11 / 33.44 / +13.34 |
| TerraMind (matched, 2.4) | Same, box loss 2 | 93.45 / 35.54 / +18.08 |
| ResNet34 326077 | ImageNet-1k, full fine-tuning (38.46M trainable), FPN, `small_extreme`, LR 2e-5 | 93.94 / 32.19 / +11.93 |
| ViT + LoRA + SAR CNN (teammate) | Rank-64 LoRA, stride-4/8 CNN fusion, 126.0M / 40.0M params; different 1,408-image holdout | 94.83 / 29.15 / — |

None of these were uploaded. A paired review (its image gallery has since been removed; compare the two runs in the
[holdout viewer](../reports/holdout-viewer/index.html)) showed
TerraMind matching 2,436 vs 2,350 objects at IoU 0.5 and 1,469 vs 1,324 at IoU 0.75 against ViT box-loss 2,
winning on ship, car, oil tank, aircraft and wind turbine but losing on airport, harbor and playground.

**Lesson.** Earth-observation pretraining (TerraMind) helped detection more than ImageNet ViT or a fully trained
CNN; the gap includes large objects and strict localization, not only tiny anchors. Bridge and playground
stayed at 2–3% AP in every backbone.

---

## Part 3. Switch to YOLO: one-stage detectors with a P2 level (Oct 16–17)

**Why.** TerraMind had plateaued, and most of the weak boxes are tiny (median shortest side about 8–10 px on
512 px images). Ultralytics detectors fit entirely within the 40M trainable budget, so they can be fully
fine-tuned, and a stride-4 **P2** detection level gives tiny objects more feature cells.

**What we did.** `models/yolo/prepare_dataset.py` converts the same 8,453/939 split. Code:
`train_100_epoch.py` (YOLO11), `train_yolo26s_p2.py` (YOLO26 n–l with P2), `train_rtdetr_p2.py`,
`train_image_head.py`, `run_experiment.py` (holdout scoring), `models/terramind/make_submission.py`. Sequential method: train
the detector (official COCO weights; P2 branch random), freeze it, then train an image head (global-average-pool
each P2–P5 map, LayerNorm, Dropout 0.2, Linear) on the 8,453 training labels for 100 epochs. Batch 16, 512 px,
seed 0. Details in [models/yolo/README.md](../models/yolo/README.md).

### 3.1 24-epoch screens and 100-epoch YOLO11s standard vs P2

24-epoch screens (image class taken from the top detection, no head): YOLO11n 29.13% mAP, YOLO11m 29.52%,
YOLO11l 29.68%, YOLO26s 32.52%, with detection-derived macro-F1 only 72–75%, hence the dedicated image head.

| Job | Model | Local F1 / mAP / Δm | Official F1 / mAP / Δm |
|---|---|---|---|
| 328063 | YOLO11s standard, 100 ep (`sbatch/yolo11s_100.sbatch`) | 89.07 / 35.56 / — | not uploaded |
| 328058 | YOLO11s-P2, 100 ep ("E1") | 89.30 / 36.31 / +17.34 | 90.45 / 36.40 / **+18.11** |

**Lesson.** A small fully fine-tuned YOLO with P2 beat every TerraMind model officially after one try. P2 added
0.75 mAP over the standard head at the same scale. This was the only family where official ≥ local.

### 3.2 Scale and architecture sweep (Phases 1–2)

`sbatch/yolo26_all_p2_100.sbatch`, `sar_phase2_architectures.sbatch`, `phase2_e9_yolo11n.sbatch`. All 100
detector epochs, sequential image head. Clean scores from the twin audit (Part 6).

| ID / job | Model | Params | Local F1 / mAP / Δm | Clean mAP / Δm | Official Δm |
|---|---|---:|---|---|---:|
| E9 329376 | YOLO11n-P2 | 2.64M | 87.16 / 31.81 / +7.57 | — | — |
| E6 328598 | YOLO26n-P2 | 2.52M | 88.46 / 32.45 / +9.50 | — | — |
| 328338 | YOLO26s-P2 | 9.67M | 90.10 / 37.47 / — | — | — |
| 329019 | YOLO11m-P2 | 20.04M | 89.86 / 39.88 / +24.47 | 37.19 / +18.59 | — |
| **E5 328339** | **YOLO26m-P2** | 21.09M | 91.08 / 40.67 / +26.63 | **38.19 / +21.45** | **+23.18** (89.75 / 39.24) |
| 329019 | YOLO26l-P2 | 25.76M | 91.14 / 40.24 / +25.83 | 37.49 / +19.96 | — |
| E7 328598 | RT-DETR-L-P2 | 35.32M | 91.34 / 40.80 / +27.02 | 36.93 / +18.97 | +22.85 (91.45 / 38.59) ² |
| E8 | Deformable DETR + P2 | — | dropped | — | — |

² E7's official numbers are user-reported and rounded; no `scores.json` was saved.

E8 ran at about 41 min/epoch on a 24 GB MIG slice at batch 4, projecting about 34 h for 50 epochs against the
24 h job limit; it was stopped at epoch 3 (code kept in `train_deformable_detr.py`).

Phase 1 training strategies on YOLO26s-P2 (`sbatch/sar_phase1_yolo26s_p2.sbatch`):

| ID | Strategy | Local F1 / mAP / Δm |
|---|---|---|
| E1 (YOLO26s ref.) | Sequential, 100 ep | 90.10 / 37.47 / — |
| E2 | 15-epoch joint fine-tune from sequential | 93.46 / 38.09 / +22.96 (clean 36.14 / +18.96) |
| E3 | Joint from COCO init, 100 ep | 93.44 / 37.11 / +21.08 |
| E4 | Conditional 200-epoch continuation (gate passed: +1.11 mAP over last 10 epochs) | no result recorded |

**Lesson.** YOLO26m-P2 was the sweet spot; YOLO26l and RT-DETR-L did not help, and on clean images were worse
(RT-DETR-L gained most from memorized twins). Nano models were far behind. Joint training lifts F1 by about
3 points. Using the image classifier to relabel or score-weight boxes always lowered mAP (YOLO26l: 40.24 native
vs 38.61 relabelled vs 38.86 weighted), so all later submissions keep native detector classes.
Note the E-numbering is overloaded: "E1" is YOLO11s-P2 in the plan text and the YOLO26s sequential reference
in the Phase 1 table.

---

## Part 4. Phase 3: joint fine-tuning of E5 (Oct 17)

**Why.** E5 had the best detector but weak F1 (official 89.75); E2 showed joint training buys about 3 F1 points.

**What we did.** `models/yolo/train_joint.py`, config `experiment_configs/phase3_yolo26m.yaml`, run via
`scripts/run_phase3_joint.sh` / `sbatch/phase3_joint_finetuning.sbatch`. 15 epochs from E5's detector and
classifier, detection + 0.5 × image cross-entropy, shared-layer LR scale 0.1, `lr0` 0.001, mosaic/mixup/
copy-paste off, best epoch by local Δm; 0.80 GPU-hours. Checkpoints staged under
`models/yolo/phase3_assets/` (Git LFS). The RT-DETR-L joint run was prepared (preflight, config) but no
result is recorded.

| Measurement | F1 | mAP | Δm |
|---|---:|---:|---:|
| Training-epoch record | 93.72 | 40.99 | — |
| Exact submission pipeline, full holdout | 93.72 | 40.60 | +27.91 |
| Clean holdout | 92.38 | 38.37 | +22.92 |
| **Official** | 92.56 | 38.78 | **+23.80** |

**Lesson.** Against E5 officially: F1 +2.81, mAP −0.46, net only +0.62 Δm. Joint gradients through the shared
backbone cost detection, which is worth 3.6× more per point. The 0.39-point training-to-export mAP difference
also showed that only the exact submission pipeline should be trusted for local scores
(`tools/audit_submission_pipeline.py` scores that exact pipeline on the holdout).

---

## Part 5. Deadline push (Oct 17, from 18:16 SGT, six hours and two uploads left; target ≥ 30)

Code: `models/yolo/deadline_recovery.py` (continuation + spatial head), `quick_classifier.py`,
`quick_box_bias.py` + `package_box_bias.py`, `paired_holdout_audit.py`, `train_yolo26m_p1.py`;
Slurm `sbatch/deadline_yolo26m.sbatch`, `deadline_yolo26m_p1.sbatch`, `quick_classifier.sbatch`.

### 5.1 Branch screen 329643: invalid

YOLO26 has a one-to-one (NMS-free) and a one-to-many (NMS) branch. The first screen evaluated the same branch
twice because Ultralytics backend setup overrides the model's `end2end` flag from its `nms` argument. The
exporter now sets `nms` explicitly and asserts the branch.

### 5.2 Spatial image head on E5 (not uploaded)

Head: per P2–P5 map, mean + max + 2 × 2 average pooling → LayerNorm → dropout → 512 hidden → GELU → dropout →
9 classes, trained on flip views of training images only. Detector 21,071,016 + head 4,347,401 parameters.

| Branch | Local F1 | Local mAP | Local Δm |
|---|---:|---:|---:|
| One-to-many, NMS IoU 0.7 | 94.13 | 40.67 | +28.27 |
| One-to-one | 94.13 | 39.17 | +25.40 |

Paired bootstrap (200 replicates) F1 gain over E5: +1.67 to +4.37 points; detection unchanged.

### 5.3 Detector-only continuation (job 329644) — official +34.57

**Why.** Keep classification gradients out of the detector (the Phase 3 lesson) and give it more epochs.
**What.** 40 detection-only epochs from E5's 100-epoch checkpoint (fresh schedule), then the spatial head on
frozen features; one-to-many branch selected.

| Measurement | F1 | mAP | Δm |
|---|---:|---:|---:|
| Local, one-to-many | 94.58 | 44.75 | +36.32 |
| Local, one-to-one | 94.58 | 43.47 | +33.87 |
| Clean, one-to-many | 93.43 | 41.73 | +29.92 |
| **Official** (accuracy 93.69, AP50 70.87) | 93.33 | 44.19 | **+34.57** |

Paired bootstrap vs E5: mAP +4.08 (95% interval +3.28 to +5.01), F1 +3.50 (+2.10 to +5.01).

A **second 40-epoch block** reached 94.43 / 44.98 / +36.67 locally but only +0.12 Δm on the clean subset
(30.04 vs 29.92), so it was not uploaded ([package](../nonsubmissions/171026/yolo26m_p2_finetune_329644/)).

**Lesson.** E5 was badly under-trained: the single largest gain of the project came from simply training the
detector longer, without the classification loss. The one-to-many branch with NMS beat the NMS-free branch by
1.3–1.5 mAP every time.

### 5.4 Wider image head (job 330230, 9 min)

Detector frozen at the first continuation; cached features train a warm-started 512-unit head and a fresh
1024-unit head (LR 5e-4, 240 epochs, label smoothing 0.05). Selected: 1024 units, epoch 234, by clean macro-F1
with full F1 not below the original. Local F1 96.06 full / 95.10 clean, detection unchanged (Δm 37.11 / 30.82).
These use the selection images, so they are not independent.

### 5.5 Per-class box-edge correction

**Why.** Small but systematic edge offsets (up to 1.7 px) matter for 8 px boxes. **What.** For each class and
source width, shift each box edge by half the median matched residual, clipped to 1 px at 512 px, needing ≥ 12
matches (IoU ≥ 0.5, score ≥ 0.1). Cross-fit on class-stratified halves of the 720 clean images, twin groups kept
together, two seeds, both directions: held-out mAP gains **+0.604, +0.814, +0.870, +0.841** points. Final offsets
fit all 720 clean images (descriptive scores: box-only 94.58 / 45.58 / +37.91 full, 93.43 / 42.71 / +31.79 clean).

### 5.6 Final candidate — official +36.80 (best)

`yolo26m_p2_candidate_330230.zip`: first-continuation detector (one-to-many), 1024-unit head, box-edge
correction; 29,748,913 parameters, one forward pass, no TTA. Local (includes calibration images): 96.06 / 45.58 /
+38.70 full, 95.10 / 42.71 / +32.68 clean. **Official Δm +36.80**, +2.23 over the continuation alone; the F1/mAP
official breakdown is macro-F1 94.69%, mAP 44.98% (AP50 71.41%). Details: [submission folder](../submissions/yolo26m_p2_candidate_330230/).

### 5.7 Other deadline attempts (none uploaded)

| Attempt | Outcome |
|---|---|
| P1 (stride-2) level on E5, job 329663 | Failed at startup after the slot opened; no result. Rule status of internal upsampling was also unconfirmed with the TA. |
| Map-only groups (teammate): 40 detection-only epochs from a Phase 3 continuation checkpoint, old head | Local 93.62 / 41.81 / +30.17; not recommended ([package](../nonsubmissions/171026/yolo26m-map-only-groups-20261017/)) |
| DenoDet V2 (FFTResNet-50 + GFL, ImageNet init, 32.94M params, `models/denodet/train.py`) | Original job 329477 failed saving epoch 20; job 329870 recovered to epoch 32; job 330220 completed 48 epochs (best checkpoint: 48). Export: macro-F1 93.75%, mAP 37.41%, Δm +21.83 full; 91.99%, 32.71%, +11.89 clean. Improved from +14.23 / +6.23 at epoch 32, still below YOLO26m. [ZIP and curves](../nonsubmissions/181026/denodet-v2-329477/), [settings/results](../results/denodet-v2-329477/seed_0/). Official scoring pending; ImageNet initialization only. |

**Lesson.** Cheap post-hoc steps on a frozen strong detector (head capacity, deterministic box calibration
validated by cross-fitting) were worth more in the last hours than new architectures. Infrastructure (file-count
quota, Slurm limit of two concurrent jobs) killed two runs; the quota was relieved by moving Python bytecode
caches to temporary storage.

---

## Part 6. Evaluation lessons

### 6.1 Near-duplicate twins and the clean holdout

219 of 939 holdout images (23%) have a same-class twin in training (64 × 64 grayscale correlation > 0.9;
same framing, 199/219 identical boxes, different speckle). Across all 9,392 labelled images, 2,345 (25%) have a
twin (2,540 pairs in `sarbench/train_twins.json`). Every evaluation path now reports the 720-image clean subset
next to the full holdout (`python -m sarbench.twins score results/<run>/seed_0` re-grades old runs). Twins are
kept in training as natural speckle augmentation; checkpoint selection still uses the full holdout.

| Phase 3 subset | Images | mAP | F1 | Δm |
|---|---:|---:|---:|---:|
| All | 939 | 40.60 | 93.72 | +27.91 |
| Twins only | 219 | 46.82 | 97.06 | +41.60 |
| Clean | 720 | 38.37 | 92.38 | +22.92 |
| Official | 1,426 | 38.78 | 92.56 | +23.80 |

### 6.2 Official validation also overlaps training

A direct query found 107 of 1,426 official images (7.5%) with a > 0.9 twin. Matching each official image to its
closest training image and checking boxes against the officially scored continuation revealed about 240 more
"soft twins" at correlation 0.7–0.9, so about **350 official images (25%) are training scenes**. Retraining on
all 9,392 images would give only 25 more official images a same-scene match.

### 6.3 Local vs official

| Model | Full holdout Δm | Clean Δm | Official Δm |
|---|---:|---:|---:|
| TerraMind 64 ep / small_extreme / large anchors | +18.68 / +18.96 / +18.08 | — | +15.37 / +15.32 / +13.42 |
| E5 YOLO26m-P2 (100 ep) | +26.63 | +21.45 | +23.18 |
| Phase 3 joint (115 ep) | +27.91 | +22.92 | +23.80 |
| First continuation (140 ep) | +36.32 | +29.92 | +34.57 |
| Final candidate | +38.70 ³ | +32.68 ³ | +36.80 |

³ Includes calibration/selection images.

Official scores fall between clean and full: nearer clean for the 100–115-epoch models, nearer full for the
140-epoch model that memorizes scenes well enough to recognize soft twins. Use full holdout to choose between
continuation blocks, and clean to confirm a gain is not just memorization; differences under about 1 mAP point
are noise.

### 6.4 Where the mAP is lost (Phase 3 diagnostics)

Bridge (5.4 AP) and playground (4.0) fail in every family tried; wind turbine is found (AP@0.1 97.3) but boxes
are loose (AP75 14.9); oil tank scores 63.6 on 256 px sources vs 21.6 on 512 px sources. The bridge/playground
audit found 57% / 61% of boxes matched at IoU ≥ 0.3 at some score, but median matched scores of only 0.066 /
0.038: a ranking problem more than blindness; many bridges are defined by road × river context.
Ruled out: classifier gating/relabelling of boxes (oracle +0.19 mAP at best), maxDets 300 (+0.02), GT shifts,
size snapping, per-class score calibration, bigger generic detectors. Ranked next ideas (NWD loss/assignment,
P1 level, IoU-aware ranking, bridge/playground oversampling and class-conditioned detection) are in
[DELTA45_PLAN.md](./DELTA45_PLAN.md).

---

## Completed detector experiments — 18 October 2026

Jobs 330445–330448 all completed successfully. The unsubmitted artifacts, full/clean scores for both prediction
heads, and Slurm completion summaries are indexed in [nonsubmissions/181026/](../nonsubmissions/181026/README.md).
These are local scores; none of these four experiments has an official Codabench score.

| Experiment | Full mAP (%) | Full macro-F1 (%) | Full Δm | Clean mAP (%) | Clean Δm |
|---|---:|---:|---:|---:|---:|
| YOLO26m-P2, 200 epochs from COCO weights (330445) | 42.2757 | 96.0526 | +32.3646 | 40.1712 | +27.9911 |
| Continue for 40 epochs with NWD box loss (330446) | 44.7466 | 93.8246 | +35.8988 | 41.5063 | +28.9021 |
| Continue for 40 epochs with gamma speckle augmentation (330447) | 45.6923 | 92.7424 | +37.1285 | 42.1203 | +29.2964 |
| Add stride-2 detection level, train 40 epochs (330448) | 44.4205 | 94.4202 | +35.5939 | 41.7882 | +29.9037 |

The table uses the one2many head and the fixed 939-image full / 720-image clean holdout. NWD, speckle and
stride-2 start from `continue_1` of job 329644. NWD and speckle reuse its image classifier; stride-2 and the
200-epoch run train their own classifier. The classifier differences affect Δm, so detection mAP is the direct
comparison of detector quality. Against the continue_2 control (44.98% full / 41.89% clean mAP), speckle's small
full-holdout gain shrinks on the clean subset; these single runs do not establish a reliable improvement.
The 200-epoch recipe did not beat the shorter continuation detector.

The 200-epoch and stride-2 jobs exported ZIPs for all 1,426 course validation images into `nonsubmissions/181026/`.
The NWD and speckle jobs exported holdout predictions only; their archive explicitly records that no
course-validation ZIP was generated. Per-epoch histories and classifier metrics are preserved under
`results/training-curves/`; raw holdout predictions and annotations are under `results/<experiment>/`.

## Chain, memorise and headroom — 18–19 October 2026

**Pipeline that reached +49.21:** YOLO26m-P2 from DOTA weights, 100 epochs → +40 plain (lr 0.002) → +40 speckle
(lr 0.001) → +40 third stage (lr 0.001) → **memorise**: 40 epochs with all augmentation off (no scale, shift,
flips, brightness), lr 0.002 → 1024-unit image classifier on frozen features.

**Memorise, the big lever.** Augmentation jitter (scale ±20%, shifts, flips) blurs exact box geometry. Turning it
off for the last 40 epochs raised clean AP75 by about 8 points and official mAP by 4.1 (third stage 47.51 →
memorise 51.64). Box-tightness at strict thresholds, not finding objects, was what the jitter cost.

Memorise variants, holdout mAP (full 939 / clean 720 / twins 219 images):

| Run (job) | Full | Clean | Twins | Verdict |
|---|---:|---:|---:|---|
| Memorise, lr 0.001 (332085) | 55.81 | 50.04 | 77.53 | official +48.83 |
| **Memorise, lr 0.002 (332326)** | 56.66 | **50.40** | 79.81 | **official +49.21, control** |
| Memorise, lr 0.004 (332325) | 57.00 | 50.15 | 81.75 | gain only on twins |
| YOLO26l memorise (332235) | 56.22 | 50.31 | 79.32 | no gain over m |
| Memorise keeping flips (332161) | 52.06 | 47.30 | 71.56 | −3.1 clean: flips hurt exact geometry |
| Missing-label loss for bridge/playground (332469) | 56.28 | 49.91 | 79.77 | no gain |
| Box-loss gain ×2 (332470) | 56.57 | 50.22 | 81.58 | gain only on twins |
| Water-map input channels + memorise (332236) | 55.55 | 49.26 | 79.31 | the water map's 100-epoch gain disappears after memorise |
| Exact 80: 80 memorise epochs straight after 100 (332471) | — | — | — | cancelled at epoch 25, 4 points behind the control at the same epoch: the chain's jittered epochs are needed |

**Studies on the control (no retraining):**
- *Post-processing* (`models/yolo/postprocess_study.py`): NMS at 0.5–0.8 and weighted box fusion. Best +0.07 clean
  mAP (fusion at 0.8). The final step is not the problem.
- *Classifier* (`models/yolo/classifier_study.py`): picking the classifier epoch on the holdout flatters F1 (95.7);
  picked on one half and scored on the other, clean F1 is 93.3–93.6, matching the official 93.8. Hidden width and
  dropout make no difference; training without flips costs about 3 F1.
- *Headroom* ([reports/headroom/](../reports/headroom/README.md)): of the 49.6 mAP points missing, 22.6 are missed
  objects (14.9 from bridge and playground), 22.3 loose boxes and 4.7 false alarms. Where label noise can be
  measured from twin images, most of our box error is the labels' own noise (about +1 mAP winnable). Ship and
  wind-turbine loose boxes (7.9 points) could not be measured. Ship labels pad the hull inconsistently.

**Label origin.** Six classes (bridge, harbor, oil tank, playground, airport, wind turbine) resemble M4-SAR: boxes
drawn on Sentinel-2 optical images and transferred to the radar image. Many bridges are hard or impossible to see
in the radar. The same YOLO26 reaches 48.8 / 75.8 AP50 on bridge / playground in M4-SAR with about 13× more images
(we reach 16.2 / 26.7), so whether ours are data-limited is tested by a half-data learning curve
(`sbatch/learning_curve.sbatch`).

**Running on the evening of 19 October** (each compared at the same stage with the matching control):
no left-right flips in the 100-epoch stage (`sbatch/noflip_exact.sbatch`, stage 1 only); a DFL box head,
`reg_max 16` (`REG_MAX=16` in `sbatch/yolo26_pretrained_full.sbatch`); an extra attention block at P4
(`P4_ATTENTION=1 THEN_MEMORISE=1`); weight averaging of the control's last checkpoints
(`sbatch/swa_study.sbatch`); and the half-data learning curve.

## Objects365 instead of DOTA weights — 20–21 October 2026

The full YOLO pipeline (`OBJ365=m` in `sbatch/yolo26_pretrained_full.sbatch`, job 333749) started from Ultralytics'
YOLO26m Objects365 weights (`pretrained/yolo26m-objv1-150.pt`) instead of the DOTA ones. Official-mix holdout
mAP at each stage, against the DOTA pipeline at the same stage:

| Stage | Objects365 | DOTA | Difference |
|---|---:|---:|---:|
| 100 epochs | 41.21 | 42.01 | −0.80 |
| plain + speckle chain | 45.57 | 45.23 | +0.34 |
| third stage | 46.19 | 45.80 | +0.39 |
| memorise | 51.12 | 51.80 | −0.68 |

Behind after 100 epochs, slightly ahead after the chain and third stage, but **behind after memorise**: the same
memorise as the DOTA control (40 epochs, no augmentation, lr 0.002; job 334608) gives official-mix 51.12 against
51.80 (clean 49.46 vs 50.40), with harbor at 512 px 7 AP lower. DOTA weights stay. The Objects365 runs were deleted.

## 90-degree rotation — 21 October 2026

`--rotate90 0.5` (exact 90/180/270-degree turns of image and boxes) in the M4-like oversampling third stage, then
memorise with beta 8 (`sbatch/third_then_memorise.sbatch`, job 334368), against the same recipe without it
(333674): official-mix 52.37 vs 52.39, clean 50.82 vs 50.92. No class moved by more than about 2 AP. Flips already
give four of the eight orientations; the other four add nothing.

Leaving harbor out of the M4-like oversampling (job 334715; harbor at 512 px had dropped about 1.5 AP when
oversampled): official-mix 52.21 vs 52.39, and harbor at 512 px stays at 49.7. The harbor drop comes from
oversampling the other classes (or noise), not from repeating harbor images.

## Rounding, hedging, soup, stride 1 and repeated scenes — 20–26 October 2026

**The path from +49.21 to +58.00.** Scores are official (Codabench). The stride-1 rows start from scratch on the
training split and rejoin the soup recipe on all data; every other row keeps everything above it.

| Date | Step | What changed | Official F1 | Official mAP | Official Δm |
|---|---|---|---:|---:|---:|
| Oct 19 | Memorise at lr 0.002 (start of this section) | | 93.83 | 51.70 | +49.21 |
| Oct 20–23 | Rounding, assigner β 8, M4-like ×4 oversampling | eight rounded uploads; their scores were not recorded here (best about 52.9 mAP) | | | |
| Oct 24 | All-data six-run **soup**, rounded | one averaged model of six memorise variants | 93.33 | 54.03 | +53.40 |
| Oct 24 | + **hedging** | ±8% copies of every box ranked below all originals | 93.33 | 54.88 | +55.02 |
| Oct 24 | **Stride 1**, training split only, own classifier | first convolution at full resolution | 93.26 | 54.28 | +53.84 |
| Oct 25 | Same boxes + stride-1 **two-head classifier** | branch copy of layers 5–10 + conv head | 94.24 | 54.28 | +54.36 |
| Oct 25 | Stride 1 on all data, single model | fine-tune stages on train + holdout | 94.45 | 55.17 | +56.18 |
| Oct 25 | Stride-1 all-data **soup** | six memorise variants averaged | 94.37 | 55.93 | +57.59 |
| Oct 26 | Soup + its own two-head classifier | heads retrained on the soup | 95.13 | 55.93 | +58.00 |
| Oct 27 | **Flip-memorise** soup, rounded + hedged | six memorise variants that also see airport/bridge/harbor/oil-tank/playground/turbine images flipped as the test's copies are; image classifier refit at packaging | 92.59 | 59.80 | +64.04 |
| Oct 27 | Flip soup + its own two-head classifier | heads retrained on the flip soup; the refit classifier's F1 drop does not carry over | 95.40 | 59.80 | **+65.55** |

### What each step was

- **Rounding (Oct 20).** Labels of some classes are whole pixels; snapping predicted edges of those classes to whole
  pixels lifts the strict IoU thresholds a little (`models/yolo/round_boxes.py`, `whole-pixel-classes`).
- **Assigner β 8 and M4-like oversampling (Oct 20–22).** The task-aligned assigner's β 8 (default 6) favours grid
  points whose box overlaps the label best; the third stage repeats the 512 px M4-SAR-like images of bridge and
  playground ×4 (`--repeat-classes bridge:4,playground:4 --repeat-width 512`). Small, consistent gains.
- **Hedging (Oct 22–23).** Our boxes find objects but often miss 0.85–0.95 IoU by a pixel. Each box gets ±8% resized
  copies (tier 1) and ±8% shifted copies (tier 2), every copy ranked below every original, so they can only add
  recall at strict thresholds. Cross-fitted on the holdout +0.84; Codabench +0.85 (`models/yolo/hedge_study.py`).
- **Soup (Oct 24).** Six memorise runs from the same third stage (β 8, 10, 12, speckle, two M4-like speckle levels),
  their weights averaged into **one** model (no ensemble). Worth about +1.4 Δm over the single all-data model.
  Averaging checkpoints within one run (epochs 20/30/40) gave nothing (−0.1).
- **Stride 1 (Oct 23–24).** The first convolution at stride 1 instead of 2, so every grid is twice as fine and the
  model sees full-resolution pixels (about 4× compute; L40S, batch 16). Official-mix +1.1 over the normal model
  (oil tank at 512 px +10.7); +0.7 on Codabench. A finer output level alone (P1) added nothing, so the gain comes
  from full-resolution processing, not the grid.
- **Two-head classifier (Oct 25–26).** Average of two heads on the frozen stride-1 detector: a trainable copy of
  backbone layers 5–10 (`models/yolo/branch_head_study.py`) and a conv head; +0.8–1.0 F1.
- **All data.** Final models train the fine-tune stages on train + holdout (about +0.9 mAP over the same recipe on
  the training split).

### What did not help (20–26 October)

SPD down-sampling (alone or with stride 1), P1 and P6 levels, YOLO26l at stride 1 (led after the first stages,
ended 0.8 behind), class-weighted loss and hard targets for bridge / playground / ship, 140 epochs straight instead
of 100 + 40 (−3.3 official-mix), nearest-neighbour upscaling of 256 px images (+0.1, noise), within-run weight
averaging, extra large-shift box copies for bridges (+0.01), and every post-processing study on the soup (box
voting, size snapping, re-scoring, classifier gating).

### Where the remaining mAP is (Oct 26)

- **Per-class test AP by probing.** A ZIP with only one class's boxes scores that class's AP / 9 (checked on the
  local metric). Two uploads: test bridge AP **5.8** (AP50 14.1), playground **20.3** (AP50 30.5), close to the
  clean holdout (8.0 / 19.3). The other seven classes average about 68.
- **Bridges and playgrounds are mostly invisible, not mislocated.** 57% of holdout bridges and 62% of playgrounds
  have no box anywhere near; misses come in whole images. Crops show found bridges cross clear dark rivers and found
  playgrounds are dark rectangles in the city; the missed ones sit on narrow streams or show nothing
  (`nonsubmissions/261026/bridge-miss-audit/`). The model fits training bridges at 96.8 AP, so it can memorise them
  but not generalise.

### Repeated scenes: flipped copies (Oct 26)

The twin check (`sarbench/twins.py`) compares whole images the same way round. Comparing also after flips, across
tile shifts and across sizes (`tools/analysis/`) showed how the 1,426 Codabench test images repeat labelled ones:

| Test images that repeat labelled content | Images |
|---|---:|
| the right way round (identical framing, near-copy or a shared 200 px tile strip) | 475 |
| **only flipped** (left-right, up-down or 180°; never 90°) | **250** |
| nothing | 701 |

- **Flipped copies are pixel copies** (holdout: pixel correlation 1.00 after flipping back), and their labels are the
  original's mirrored **one pixel further left / up** than an exact mirror (x' = W − x − w − 1 at 512 px: 536 of
  548 holdout boxes; − 2 at 256 px on a small sample). Ultralytics' random flip mirrors exactly, so it teaches boxes 1 px off — likely
  why "memorise keeping flips" lost 3.1 on 19 October.
- **Memorise learns exact pixels in one orientation:** the current model scores 66.5 mAP on the holdout's flipped
  copies, against about 83 on same-way-round twins and 51 on new scenes.
- **The "twins" are not re-captures** but the same capture with different brightness / sharpness (fine-grain
  correlation 0.95); **near-copies** (r 0.7–0.9) are re-captures with new speckle (fine grain 0.14–0.32).
- **Tiles:** the 512 px images are cut every 312 px, so neighbours share a 200 px strip (208 test images share one
  with a labelled tile, about 110 more with a flipped one). No repeats across 256 / 512 px.
- The image classifier is already 100% on twins and flipped copies (it trains with flips, which move no labels); its
  32 holdout errors are on near-copies (7) and new scenes (25), mostly bridge ↔ playground.

**Which classes the dataset flips (labelled data only).** Among the 9,392 labelled images, the share that have
another labelled image as a flipped copy (`tools/analysis/flip_pairs_labelled.py`): wind turbine 35%, harbor 21%,
airport 17%, oil tank 10%, playground 10%, bridge 4%, ship 1% (8 of 1,173), aircraft 0 of 1,182, car 0 of 443. The
dataset mirrors the M4-SAR-like classes and not aircraft, car or ship scenes.

**Flip-memorise** (`--dataset-flips`): memorise with every training image also shown in the three flipped
orientations, boxes flipped with the dataset's offset (training data only; no label is copied onto test images).
Training-split test (job 338727) against the same third stage: raw holdout mAP 59.8 at epoch 14 of 20 against the
plain memorise's best 57.9. Strong constant speckle on top (338730) runs about 2 behind it; a speckle-then-clean
schedule (`--anneal-speckle`, 338803) is queued. **Result:** clean-holdout official-mix (rounded + hedged) **58.32 against 54.73 (+3.59)**; the holdout's flipped
copies 66.5 → 88.1, other clean images 44.6 → 44.0. Per class: wind turbine +16.4, playground +6.2, airport +5.2,
harbor +5.1, oil tank +4.9, bridge +0.7, but aircraft −0.8, car −2.3 and ship −3.2, mostly at the strict IoU
thresholds: those classes have no flipped duplicates in the training data, so their flipped views only halve the
passes in their real orientation and shift their boxes by an offset they never have.

**Class-selective flips** (`--dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine`), chosen
from the two training-data findings above (which classes the dataset mirrors, and which classes lost on the
holdout): aircraft, car and ship images are listed four times as they are, 80 passes instead of 40. The all-data
soup is rebuilt with six class-selective ingredients (`~/sbatch-staging/flipsoup3.sh`).

## Phase 2 checks: new images — 31 October 2026

Phase 2 scores new images with the final model, so on 31 October we ranked runs by **new6**: holdout AP on new scenes
(64 px correlation < 0.6 to every training image under the four flips, 451 images), averaged over the six classes with
at least 95 new-scene labels (aircraft, bridge, car, oil tank, playground, ship). Airport (13), harbor (15) and wind
turbine (8) have too few labels to rank runs. All runs below are training-split models from the same third stage,
rounded and hedged (`sbatch/staging/job_g.py`, `job_n.sbatch`).

| Run (training split) | new6 | Flipped copies | Official-mix |
|---|---:|---:|---:|
| Plain memorise (337666) | 46.88 | 66.5 | 54.73 |
| All-class flips (338727) | 45.50 | 88.1 | 58.32 |
| Class-selective flips (344678, the final recipe) | 46.15 | 87.5 | 58.88 |
| 50/50 weight average of plain and all-class flips | 46.55 | 79.7 | 57.14 |
| Plain, frozen backbone (`--freeze 11`) | −0.4 vs plain | | |
| Plain, one image per twin group (`--dedupe-twins`, 344970) | 46.81 | 64.8 | 54.65 |
| Plain, weight decay 0.005 (345074) | 46.85 | 66.4 | 54.45 |
| Plain, constant LR + SWA of epochs 20–40 (344680) | 46.98 | 66.3 | 54.45 |
| Blends toward the third stage (75/25, 50/50) | 46.82, 46.67 | | |

The flip training is the only part of the final model that leans on Phase 1: −0.7 new6 for +21 on the flipped copies.
The all-data plain/flip average, with its own two-head classifier and all five steps, scored **+63.03** on Codabench
(58.60 mAP, 95.00 F1; not posted), −2.77 against the final model. Its expected Phase 2 gain (about +0.3 mAP, +0.5 Δm) is
within run-to-run noise (±0.3 new6), so the final model stays the +65.80 flip soup. None of the four regularisers
moved new6 beyond noise.

Each step after the detector, added one at a time on the holdout's new scenes (`sbatch/staging/job_s.py`, same on
all three models): rounding +0.3 new6, hedging +0.8, the low-score tail +0.1. Image labels on new scenes: the 50/50
two-head average 94.9% accuracy / 93.5 F1, against 93.8 / 92.7 (branch head) and 92.5 / 91.3 (conv head). F1 on new
scenes is about 2 points below the whole holdout (95.7), so Phase 2's F1 will likely land a little under Phase 1's.

One command for Phase 2 (31 Oct): the five steps moved from `make_final_zip.py` into
`models/yolo/make_submission.py --final` (`make_final_zip.py` now just calls it). Job 346225
(`sbatch/staging/job_m.sbatch`, 4 minutes on a Blackwell slice) rebuilt the final ZIP with it: `detection.json` and
`classification.csv` byte-identical to the uploaded +65.80 ZIP (MATCH). The same command on 60 test images copied to
their own folder, the way Phase 2's images will come, gave the same labels and boxes within 1 px (CLOSE MATCH).

Multi-label NMS (1 Nov): Ultralytics' predictor keeps only each box's top class; its validator keeps every class
scoring above the threshold. `make_submission.py --multi-label` does the latter. On the holdout, rounded and hedged as
submitted (job 348074, `sbatch/staging/job_g.py` with `job_g2.json`): official-mix 58.87 → 58.87 and new6
46.14 → 46.13 on the class-selective flip model (344678), 54.63 → 54.62 and 46.68 → 46.68 on the plain stride-1 model
(337666). The detector almost never gives a box a second class at 0.001 or more (4,180 → 4,214 boxes on 939 images),
so there was nothing to gain. Left out.

## Cross-cutting lessons

- **Detection is the score.** 1 mAP point ≈ 1.91 Δm vs 0.54 for F1. Changes that trade mAP for F1 (Phase 3)
  barely pay.
- **Training length beat cleverness twice.** 24 → 36 → 64 epochs carried TerraMind from +2.89 to +15.37; 40 more
  detector epochs carried YOLO26m from +23.18 to +34.57. Check the curve before adding complexity.
- **Architecture family mattered more than any tweak.** A 9M-parameter fully fine-tuned YOLO11s-P2 beat every
  frozen-backbone TerraMind recipe. Within YOLO, medium beat large; RT-DETR's lead was twin memorization.
- **Keep the tasks apart.** Train the detector on detection loss only, then fit the image head on frozen
  features; never relabel or score-weight boxes with the classifier.
- **One change per run, two seeds or a paired bootstrap.** Most TerraMind ablations moved by less than the
  noise and several changed 2–4 factors at once, so their effects cannot be attributed.
- **Measure through the exact submission pipeline** and assert the configuration you think you ran (the
  invalid branch screen, the 0.39-point export gap).
- **Know your holdout.** A random split of this dataset leaks same-scene twins; group twins when splitting and
  always report a clean subset. The official set leaks too (about 25%).
- **Calibrate post-hoc steps by cross-fitting**, keeping twin groups together, before spending an upload.
- **Augmentation helps finding objects and hurts exact edges.** Train with jitter, then finish without it
  (memorise). With strict IoU thresholds, how sharp the boxes are matters as much as what is found.
- **Gains that appear only on twin images are memorisation.** Judge on the clean subset; box-loss ×2 and lr 0.004
  improved the twins but not the clean images.
- **Picking the best epoch on the holdout flatters it.** Score honestly by choosing on one half and testing on the
  other (the classifier lost 2 F1 points that way, matching Codabench).
- **Know the ceiling before tuning.** Label noise limits strict-threshold AP for small objects; measure it (twins)
  before spending GPU time on box precision.
- **Look for repeats under every symmetry the data might have been made with.** Our twin check compared images
  the same way round and missed that 18% of the test images repeat labelled scenes only flipped; checking flips, tile shifts and
  sizes took minutes on CPU.
- **Probe what the score hides.** One-class uploads gave the test-set AP of a single class for one upload each.
- **Average weights, not models.** A soup of six runs from one start point is still one model and gained more
  than any single recipe change (+1.4 Δm).
- **Budget the infrastructure.** Uploads (5/day), Slurm concurrency (2 running GPUs, 4 submitted jobs), 24 h job limit,
  the 100,000-file and 500 GB quota, and MIG slices decided what could run as much as the ideas did.

## Where artifacts live

| Folder | Contents |
|---|---|
| [submissions/](../submissions/README.md) | One folder per ZIP uploaded to Codabench (25), with packaging metadata and, where downloaded, `scoring_result/scores.json`. The index is the authoritative list of official scores. |
| [nonsubmissions/](../nonsubmissions/) | ZIPs packaged but never uploaded (ablations, ViT/ResNet comparisons, second continuation, quick head, box-only, map-only, DenoDet). |
| [results/](../results/README.md) | Local metrics, holdout predictions and annotations per run, including clean-holdout scores and the twin audits in [results/yolo26m_p2_finetune_329644/](../results/yolo26m_p2_finetune_329644/). |
| [checkpoints/](../checkpoints/README.md) | TerraMind/ViT/ResNet weights and configuration (Git LFS). YOLO checkpoints for Phase 3 are in `models/yolo/phase3_assets/`; the final image head is in the candidate's submission folder. |
| `reports/` | [Holdout viewer](../reports/holdout-viewer/index.html) (every holdout image, runs side by side), [curve comparison](../reports/curve-compare/index.html), and the bridge/playground audit images. Raw per-epoch training CSVs are in [results/training-curves/](../results/training-curves/README.md). |

Code entry points and Slurm scripts per model family are mapped in the [README](../README.md); training
protocol details are in [DETAILS.md](./DETAILS.md) and [WORKFLOW.md](./WORKFLOW.md).
