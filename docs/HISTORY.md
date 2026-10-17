# Experiment history: what we tried, why, and what we learned

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
which is why official scores land between our full and twin-free ("clean") local scores.

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
| Oct 17 | Same detector + 1024-neuron image classifier + box-edge correction | 94.69 | 44.98 | **+36.80** |

All 25 uploads, with Codabench IDs, are in [submissions/README.md](../submissions/README.md). Unless stated,
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

None of these were uploaded. The paired review ([findings](../reports/vit-terramind-48/FINDINGS.md)) showed
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
([training-comparison report](../reports/training-comparison/README.md)).

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
(30.04 vs 29.92), so it was not uploaded ([package](../nonsubmissions/yolo26m_p2_finetune_329644/)).

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
| Map-only groups (teammate): 40 detection-only epochs from a Phase 3 continuation checkpoint, old head | Local 93.62 / 41.81 / +30.17; not recommended ([package](../nonsubmissions/yolo26m-map-only-groups-20261017/)) |
| DenoDet V2 (FFTResNet-50 + GFL, ImageNet init, 32.94M params, `models/denodet/train.py`) | 329456 → 329477 (48 ep) failed at epoch 20 on disk quota; 329870 recovered to epoch 32; resume 330220 saved no epoch-48 evaluation. Epoch 32: 92.66 / 33.75 / +14.23 full, 90.86 / 30.07 / +6.23 clean ([models/denodet/README.md](../models/denodet/README.md)) |

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
[DELTA35_BRAINSTORM_20261017.md](./DELTA35_BRAINSTORM_20261017.md).

---

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
- **Budget the infrastructure.** Uploads (5/day), Slurm concurrency (2 jobs), 24 h job limit, disk/file-count
  quota and MIG slices decided what could run as much as the ideas did.

## Where artifacts live

| Folder | Contents |
|---|---|
| [submissions/](../submissions/README.md) | One folder per ZIP uploaded to Codabench (25), with packaging metadata and, where downloaded, `scoring_result/scores.json`. The index is the authoritative list of official scores. |
| [nonsubmissions/](../nonsubmissions/) | ZIPs packaged but never uploaded (ablations, ViT/ResNet comparisons, second continuation, quick head, box-only, map-only, DenoDet). |
| [results/](../results/README.md) | Local metrics, holdout predictions and annotations per run, including clean-holdout scores and the twin audits in [results/yolo26m_p2_finetune_329644/](../results/yolo26m_p2_finetune_329644/). |
| [checkpoints/](../checkpoints/README.md) | TerraMind/ViT/ResNet weights and configuration (Git LFS). YOLO checkpoints for Phase 3 are in `models/yolo/phase3_assets/`; the final image head is in the candidate's submission folder. |
| `reports/` | [Training comparison](../reports/training-comparison/README.md) (curves, official-vs-local audit) and the [ViT/TerraMind paired review](../reports/vit-terramind-48/FINDINGS.md). |

Code entry points and Slurm scripts per model family are mapped in the [README](../README.md); training
protocol details are in [DETAILS.md](./DETAILS.md) and [WORKFLOW.md](./WORKFLOW.md).
