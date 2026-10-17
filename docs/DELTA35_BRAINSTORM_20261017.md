# Beating official Δm +35: brainstorm (2026-10-17)

Latest recorded official result: YOLO26m-P2 first continuation, macro-F1 93.3310%, mAP 44.1919%,
**Δm +34.5697**. The original brainstorm used the older Phase 3 joint model
(macro-F1 92.56%, mAP 38.78%, Δm +23.80); its historical diagnostics and scenarios remain below.
Those diagnostics were run on the 939-image local holdout using that model's exact submission-pipeline
predictions (`results/phase3-yolo26m-joint-20261017-1516/submission-pipeline/`). They are local evidence,
not official scores. Gain figures in the "headroom" table are scenarios, not predictions.

## Quick follow-up: latest continuations and clean images

Rescoring saved one-to-many predictions requires no retraining:

| Run | Full holdout Δm (939) | Clean mAP (720) | Clean macro-F1 | Clean Δm |
|---|---:|---:|---:|---:|
| First continuation | 36.3202 | 41.7335% | 93.4303% | 29.9219 |
| Second continuation | 36.6724 | 41.8928% | 93.0817% | 30.0398 |

The second continuation gains only **0.1180 clean Δm**, compared with 0.3522 on
the full holdout. This single comparison does not establish a reliable gain.
These local clean scores are not forecasts of official scores: the first
continuation already has official Δm 34.5697 despite clean Δm 29.9219.

A direct query found **107 candidate training twins among 1,426 official
validation images (7.50%)**, compared with 219/939 (23.32%) in the local
holdout. The validation query uses 64×64 grayscale thumbnail Pearson
correlation > 0.9, without class filtering because official labels are
unavailable; matches have not been manually verified. This supports fewer
training twins in official validation but does not isolate the cause of score gaps.

Saved evidence: [clean continuation comparison](../results/yolo26m_p2_finetune_329644/clean_continuation_comparison.json)
and [candidate validation twins](../results/yolo26m_p2_finetune_329644/official_validation_training_twins.json).
Clean scores are also recorded beside each continuation's full holdout metrics.

Keep training twins as useful speckle variants. For a new training comparison,
select checkpoints on clean validation scores. Future fresh splits should keep
connected same-scene twin groups together in training or validation; changing
the split cannot remove information already learned by existing checkpoints.
This improves evaluation reliability rather than directly increasing official AP.

### Soft twins in official validation

The 0.9 cutoff misses many same-scene images. Each official validation image was matched to its most correlated
labeled training image (all 9,392). A match counts as the same scene when the training image's boxes coincide
(same class, IoU ≥ 0.5) with the officially scored first continuation's boxes (score ≥ 0.3) on the validation
image. Its predicted image class also checks the match's class:

| Best-match correlation | Official images | Predicted class = match's class | Match's boxes found |
|---|---:|---:|---:|
| > 0.97 | 57 | 100% | 90% |
| 0.9–0.97 | 50 | 98% | 93% |
| 0.8–0.9 | 109 | 95% | 82% |
| 0.7–0.8 | 133 | 87% | 83% |
| 0.6–0.7 | 127 | 76% | 66% |
| ≤ 0.6 | 950 | 27% | 13% |

The 107 candidates above 0.9 are confirmed same-scene twins (106/107 class agreement, 92% of boxes found). About
240 more images at 0.7–0.9 are "soft twins": same framing and boxes, but more different in appearance. In total
about 350 of the 1,426 official images (25%) are training scenes. So the 7.50% figure above understates the
overlap. The local holdout is about one third same-scene (219 twins plus 92 images at 0.7–0.9; that local band
was not box-checked). The 720-image clean subset still contains those 92 soft twins.

Consequences:

- Official scores sit between the clean and full holdout scores. They land nearer the full score for models that
  memorize scenes robustly enough to recognize soft twins. That fits all three officially scored runs: E5 and
  Phase 3 (100–115 detector epochs) landed nearer clean, and the 140-epoch first continuation nearer full.
- When choosing between continuation blocks, the full holdout is the closer proxy for official scores. Use the
  clean score to check that a gain is not only memorization.
- Retraining on all 9,392 images gives only 25 official images (1.8%) a same-scene match they lack today
  (correlation > 0.7 only with a holdout image). Expect little beyond the general effect of 11% more data.
- Photometric or speckle augmentation that mimics the differences between twins could help memorized scenes
  transfer to soft twins. Untested.

### Thirty-minute follow-up: deterministic box-edge correction

The first continuation's frozen predictions support a quick calibration candidate:
fit half the median edge residual for each class/source width, clipped to one
pixel at 512px and requiring at least 12 matched boxes. Two fixed seeds split
the 720 clean images into class-stratified halves, keeping connected twin groups
together. Fitting on one half and evaluating on the other, then reversing,
gives mAP gains of **0.6043, 0.8138, 0.8698, and 0.8408 percentage points**.

The final offsets fit all 720 clean images. Their final fitted full/clean scores
are descriptive because those images participate in calibration; the four
held-out checks above are the evidence for transfer. Classification is unchanged.
The [candidate ZIP and calibration audit](../submissions/yolo26m_p2_box_bias_20261017/)
are ready; official scoring is pending. Reproduce with
`yolo_baseline/quick_box_bias.py` and `yolo_baseline/package_box_bias.py`.

## 1. What +35 requires

`Δm = 50 × [(F1 / 0.9331 − 1) + (mAP / 0.2613 − 1)]`

- 1 mAP point = **+1.91 Δm**; 1 F1 point = **+0.54 Δm**. mAP is worth 3.6× more per point.
- Δm +35 needs +11.20 over today.

| Official macro-F1 | Official mAP needed for Δm 35 |
|---:|---:|
| 92.56% (today) | 44.63% (+5.85) |
| 95.00% | 43.95% (+5.17) |
| 97.00% | 43.39% (+4.61) |

The current model loses 1.82 mAP points from local to official, but most of that gap is near-duplicate leakage in
the local holdout (section 8). For E5 and Phase 3, the 720 clean holdout images land 0.4–1.1 mAP below
official, while the full holdout lands 1.4–1.8 above. Judge changes on the clean subset and aim for about
**44.6% mAP** there, which is slightly conservative.

## 2. Where the mAP is lost

| Class | AP | AP@0.1 | AP50 | AP75 | Diagnosis |
|---|---:|---:|---:|---:|---|
| aircraft | 64.5 | 93.5 | 91.6 | 74.1 | healthy |
| airport | 63.0 | 90.2 | 88.6 | 71.5 | healthy |
| **bridge** | **5.4** | 24.2 | 14.5 | 3.4 | fails to find objects (recall) |
| car | 61.8 | 93.5 | 92.6 | 72.9 | healthy |
| harbor | 51.5 | 81.2 | 78.2 | 59.3 | ok |
| oil_tank | 39.5 | 70.1 | 62.7 | 40.9 | 512 px source is weak (below) |
| **playground** | **4.0** | 14.6 | 10.0 | 3.7 | fails to find objects (recall) |
| ship | 44.5 | 90.2 | 83.4 | 42.4 | localization |
| **wind_turbine** | **31.3** | **97.3** | 82.6 | **14.9** | found, but boxes too imprecise |

Split by image source (width): oil tank AP **63.6% on 256 px images vs 21.6% on 512 px images**; bridge 15.9% vs
4.5%. The 512 px images contain the tiny objects: GT median box side is 8.0 px (wind turbine), 9.9 px (bridge),
10.6 px (oil tank), 14.8 px (playground). Matched wind-turbine boxes average IoU 0.67. At 8 px, a 1 px error
already costs about 0.2 IoU.

Bridge/playground are 1–7% AP in **every** model family tried (ViT, TerraMind, ResNet34, YOLO11/26 n–l, RT-DETR).
The bottleneck is the data regime, not a particular architecture.

## 3. Ruled out by today's checks

| Hypothesis | Check | Result |
|---|---|---|
| Wrong-class boxes hurt mAP; use image classifier to gate/relabel | Oracle: delete every box not of the true image class | **+0.19 mAP** at best. The classifier can only buy F1 |
| maxDets limit | maxDets 300 vs 100 | +0.02 mAP |
| Bridge/playground labels shifted or scaled | GT centre distribution | uniform over the image; no shift |
| Systematic box offset | mean matched-box edge error | ≤ 1.7 px (car x1/y1, airport x2); small win only |
| Quantized GT sizes → snap box sizes | wh histogram | 252 distinct wind-turbine sizes; no |
| Per-class score calibration | — | COCO AP ranks within each class, so per-class rescaling cannot change mAP |
| Bigger general detector | existing runs | YOLO26l-P2 40.2 < YOLO26m-P2 40.7; RT-DETR-L-P2 40.8. On clean images: YOLO26m 38.2 > YOLO26l 37.5 > RT-DETR-L 36.9. Capacity is not the limit |

## 4. Headroom scenarios

| Lever | Scenario | mAP | Δm |
|---|---|---:|---:|
| Wind-turbine localization | class AP 31 → 45 | +1.6 | +3.0 |
| Bridge + playground recall | 5.4 / 4.0 → 20 each | +3.4 | +6.5 |
| Oil tank on 512 px images | class AP 39.5 → 50 | +1.2 | +2.2 |
| Ship localization | 44.5 → 50 | +0.6 | +1.2 |
| F1 | 92.56 → 95 | — | +1.3 |
| **Total if all land** | | **+6.7** | **≈ +14** |

There is no single +11 lever. Getting to 35 needs tiny-object localization **and** some bridge/playground recall.

## 5. Ranked ideas

### A. Cheap, low risk (hours)

1. **Train the final model on all 9,392 labelled images** (add the 939 holdout) with the epoch count picked on the
   holdout. +11% data. Expected small but non-negative.
2. **Keep E5 detection, add Phase 3 F1 (ceiling +0.9 Δm, uncertain).** Joint fine-tuning gave +2.81 official
   F1 but −0.46 official mAP. Locally the mAP drop was only −0.07, so part of the official drop may be noise.
   A plain classifier on frozen features is what E5 already is (official F1 89.75), so it would only reproduce
   E5. The Phase 3 F1 gain came from adapting the shared features. To keep that gain with the detector frozen,
   the classifier needs its own trainable features: for example, a private copy of the last backbone stage
   feeding the classifier head, trained on classification only. E5 mAP (39.24) + Phase 3 F1 (92.56) ≈
   **Δm +24.7** is the ceiling, not an expectation. Keep one shared backbone and one forward pass; the TA's
   MoE-LoRA reference also uses task-specific feature paths. Joining two full trained backbones would be an
   ensemble. Cheaper alternative: rerun the joint fine-tune with a lower shared-layer learning rate or
   classification weight 0.1–0.2. Do this last, on top of the final detector.
3. Per-class box-edge bias correction fitted on the holdout (car, airport, harbor@256). +0.1–0.3 mAP.

### B. Tiny-object localization: the main controllable lever (1–2 GPU days)

4. **Higher-resolution internal features at the same 512 px input**: stem stride 1 or an extra stride-2 (P1) head
   on YOLO26m-P2. Every 5–10 px object gets 2× more feature cells. A stride-1 first layer costs about 4× compute (every later
   grid doubles in both directions, like a 1024 px input). An extra stride-2 level alone costs less, roughly
   1.5–2× by estimate; Ultralytics prints the exact GFLOPs when it builds the model.
   **Rule check first:** confirm with the TA that internal resolution changes are allowed under "each image in once at ≤ 512 × 512".
5. **Tiny-box loss/assignment**: Normalized Wasserstein Distance (NWD) loss and NWD- or RFLA-based label
   assignment. IoU loss and IoU assignment behave badly below 16 px; these were designed for AI-TOD-style objects.
   Target: wind-turbine AP75 15 → 35+, oil tank@512 AP 22 → 35+.
6. **Localization-aware ranking**: an IoU-prediction branch (or VFL-style IoU-aware class target) so tightly
   localized boxes rank first. This raises AP75–AP95 without changing recall.
7. (Heavier) Second-stage box refinement for boxes < 16 px from P1/P2 features.

### C. Bridge / playground recall (largest headroom, most uncertain)

8. **Visual audit first** (half a day): bridge@512 has 326 GT boxes, but only ~38 are matched at IoU ≥ 0.3 with
   score ≥ 0.25. Use the report's class filter to see whether these objects are visible and consistently labelled.
   If not, the ceiling is low, so stop investing and put the time into B.
9. Image-class-conditioned detection: feed the image-class logits into the detection head (FiLM or gating) so a
   "bridge image" primes the bridge channel. Unlike the ruled-out gating, this targets **recall**, not false positives.
10. Repeat-factor sampling of bridge@512 / playground@512 images, plus copy-paste of their tiny instances
    (Phase 3 disabled copy-paste globally). The earlier TerraMind small-object sampling result was mixed, so treat
    this as an ablation, not a default.

### D. Classification (≤ +1.5 Δm realistic)

11. Bridge ↔ playground confusion is 23 of 65 holdout classification errors. A multi-scale (P2–P5) pooled
    classifier head is the obvious try. F1 is capped at a small Δm contribution, so it ranks below B and C.

### Don't spend more time on

Classifier relabel/score-weighting of boxes, per-class score calibration, maxDets, larger generic detectors.

## 6. Suggested order

1. Bridge/playground visual audit (8) and TA rule question for (4). Both are free.
2. One GPU-day ablation pair on YOLO26m-P2: **P1 / stem-stride-1** vs **NWD loss + assignment**. Measure
   wind-turbine AP75, oil tank@512 AP and bridge/playground AP on the holdout.
3. Final: winner + (9) if the audit says the labels are learnable, retrained on all 9,392 images, then the
   frozen-detector classifier-branch step (2).

## 7. One change at a time

Rules:

- Score every run on the 720 clean holdout images (section 8), not the full 939.
- Ablate detection changes with detection-only fine-tunes (no classification loss), so F1 shifts don't confound
  them. Handle classification once, at the end.
- Loss, assignment and sampling changes: 15-epoch fine-tune from E5 (same length as Phase 3), compared with a
  matched control. Architecture changes: full 100-epoch run, compared with E5.
- Run the control twice with different seeds. The gap between the two controls is the noise level; a change
  smaller than that is not a result.
- Keep a change only if its target metric improves beyond noise and overall holdout mAP does not fall below the
  control. Stack winners in order and re-check after each addition.

| # | Change | Watch | Cost |
|---|---|---|---|
| 0 | Visual audit of bridge/playground @512; ask the TA about internal resolution | whether 5–7 are worth it; whether 4 is allowed | no GPU |
| 1 | Control: 15-epoch detection-only fine-tune from E5, two seeds | noise level | 2 short runs |
| 2 | NWD term in the box loss | wind-turbine AP75, oil tank @512 AP | short run |
| 3 | NWD in label assignment (TaskAlignedAssigner) | same as 2 | short run |
| 4 | Stride-2 (P1) head or stride-1 stem, only if allowed | wind-turbine AP75, oil tank @512, bridge/playground AP | full run; stride-1 stem ≈ 4× compute, extra stride-2 level ≈ 1.5–2× |
| 5 | Oversample bridge/playground images (repeat-factor sampling), only if the audit says the labels are learnable | bridge, playground AP | short run |
| 6 | Copy-paste tiny bridge/playground instances (same condition) | bridge, playground AP | short run |
| 7 | Feed image-class logits into the detection head (same condition) | bridge, playground AP | more code, short run |
| 8 | Final: stack winners, retrain on all 9,392 images, classifier-branch step (2), fit box-edge bias correction (3) on the final holdout predictions, submit | official Δm | full run + submission |

## 8. Step 0 results

Contact sheets and lists are in `reports/bridge-playground-audit/`. Green = ground truth, red = predictions with
score; crops span 2.5× the box and are upscaled. The two `control_*` sheets show wind turbines and oil tanks for
comparison.

### Bridge and playground @512

- **Bridge:** boxes mark points where a road crosses a dark river or canal, often in chains along one river.
  About a quarter of the sampled crops show a visible bright deck; the rest show only the dark water line or
  nothing distinctive at this resolution. The model finds mainly the visible ones. For the rest the cue is
  context (road × river), not appearance.
- **Playground:** many boxes enclose a dark, smooth ~15×20 px patch (a sports field) in bright urban clutter, so
  roughly half look learnable. Similar dark patches elsewhere are unlabelled, which caps precision.

| Holdout subset | GT boxes | Matched (IoU ≥ 0.3), any score | score ≥ 0.1 | score ≥ 0.25 | Median score of matched box |
|---|---:|---:|---:|---:|---:|
| bridge @512 | 326 | 57% | 24% | 12% | 0.066 |
| playground @512 | 252 | 61% | 22% | 11% | 0.038 |
| wind turbine @512 (comparison) | 923 | 99% | 98% | 96% | 0.575 |
| oil tank @512 | 359 | 84% | 64% | 43% | 0.256 |

The model often places a box on the right spot but scores it below false positives: a ranking problem more than
blindness. Verdict:

- Run steps 5 (oversampling) and 7 (image-class conditioning) for both classes.
- Run copy-paste (step 6) for playground only. A pasted 10 px bridge patch loses the river-road context that
  defines it.
- Expect roughly 10–15% AP per class, not 20%. The section 4 recall scenario is optimistic: closer to +2 mAP
  (+3–4 Δm) than +3.4 (+6.5).

### Near-duplicate twins between holdout and training

219 of the 939 holdout images (23%) have a same-class twin among the 8,453 training images (64×64 grayscale
correlation > 0.9, exhaustive search). They are not exact copies: no file is byte-identical, and the speckle
differs (median full-resolution correlation 0.94; only 17 of 185 pairs checked differ by brightness/contrast
alone). But the framing is the same, and 199 of the 219 have identical box labels: the same place, captured or
processed twice and labelled once. For the model, such an image is nearly a training image.

| Holdout subset | Images | mAP | F1 | Δm |
|---|---:|---:|---:|---:|
| All | 939 | 40.60 | 93.72 | +27.91 |
| Twins only | 219 | 46.82 | 97.06 | +41.60 |
| **Clean (no twin in training)** | **720** | **38.37** | **92.38** | **+22.92** |
| Official validation | 1,426 | 38.78 | 92.56 | +23.80 |

E5 scores Δm +26.63 on the full holdout, +21.45 clean and +23.18 official. Across the two officially scored
runs, the full holdout overstates official Δm by 3.5–4.1 and the clean subset understates it by 0.9–1.7. Clean is
much closer, but differences under about 1 mAP point are noise. The official validation set seems to contain
fewer twins of training images than the local holdout. This is an inference; check it directly (fix 3).

Across the whole labeled set, 2,345 of 9,392 images (25%) have a twin: 2,540 pairs in `sarbench/train_twins.json`.

Fix (implemented):

1. Every evaluation path now reports the clean subset next to the full holdout: `train.py` (`val['clean']` in
   each validation record and in the log line), cross-validation summaries, `calibrate_submission.py`,
   `yolo_baseline/run_experiment.py` (`val.clean_holdout`), `yolo_baseline/train_deformable_detr.py`,
   `denodet_baseline/export.py` and the submission audit (`clean_holdout`, `clean_minus_official_percentage_points`).
   Checkpoint selection is unchanged and still uses the full holdout.
2. Old runs need no retraining: `python -m sarbench.twins score results/<run>/seed_0` re-grades their saved
   predictions. Macro-F1 and Δm are included when per-image classifications were saved.
3. On the GPU machine: `python -m sarbench.twins query --images <official validation image folder>` counts official
   validation images with a training twin.
4. Leave twins in the training data. Within training they act like speckle augmentation of the same scene.

Re-graded completed runs (mAP %, Δm where per-image classifications exist):

| Run | mAP all 939 | mAP clean 720 | Δm all | Δm clean |
|---|---:|---:|---:|---:|
| Phase 3 YOLO26m joint | 40.60 | 38.37 | +27.91 | +22.92 |
| E5 YOLO26m-P2 | 40.67 | 38.19 | +26.63 | +21.45 |
| YOLO26l-P2 | 40.24 | 37.49 | +25.83 | +19.96 |
| YOLO11m-P2 | 39.88 | 37.19 | +24.47 | +18.59 |
| RT-DETR-L-P2 (E7) | 40.80 | 36.93 | +27.02 | +18.97 |
| YOLO26s-P2 E2 joint | 38.09 | 36.14 | +22.96 | +18.96 |
| Best TerraMind (64-epoch) | 36.29 | 33.70 | — | — |

RT-DETR-L looked like the best detector on the full holdout but is fourth on clean images: it gained most from
memorized twins.

Clean per-class AP also changes priorities: airport 47.2 (63.0 on the full holdout) and harbor 38.7 (51.5) are
weaker than they looked, and oil tank is 51.8 (39.5). Re-rank the section 4 headroom on clean numbers before
choosing between steps.

### Rule check

The Codabench API page returned unrelated text, so the exact Terms wording is unverified. Question for the TA:

> 1. Our model takes each image once at 512 × 512. May the network keep higher-resolution feature maps inside,
>    for example a stride-1 first convolution instead of stride 2, or an extra stride-2 detection level? We do not
>    upsample or tile the input. Would an explicit upsampling layer inside the model be treated differently?
> 2. About 23% of the images in a random 10% split of the training set have a same-scene twin elsewhere in
>    training (same framing and boxes, different speckle). Are the validation and test sets deduplicated against
>    training?

Analysis scripts used for these numbers are short pycocotools checks on the files named above (not committed).
