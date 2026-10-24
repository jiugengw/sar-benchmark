# Delta 45: remaining 48 hours, 18 October 2026

Keep YOLO as the main incremental track. The user is already testing Dome-DETR independently; do not duplicate that work. A different architecture is an optional fallback, not a requirement for reaching the target.

## Target and current evidence

The best official submission has mAP 44.9758% and macro-F1 94.6932%. Holding F1 fixed, delta 45 requires mAP **49.2596%**, a gain of **4.2839 percentage points**. Reaching 50% provides some margin. No experiment below has established that gain.

Latest uncalibrated detector comparisons, using saved one2many scores:

| Recipe | Full mAP | Clean mAP |
|---|---:|---:|
| First continuation, source of best submission | 44.75 | 41.73 |
| Second continuation, matched recipe control | 44.98 | 41.89 |
| NWD continuation | 44.75 | 41.51 |
| Speckle continuation | 45.69 | 42.12 |
| Stride-2 head continuation | 44.42 | 41.79 |
| Fresh 200 epochs | 42.28 | 40.17 |

These results deprioritize repeating those exact recipes. They do not establish a ceiling for YOLO. Speckle's improvement over the second continuation is only 0.71 full / 0.23 clean points; its classifier was reused across changed detector features and needs refitting before comparing final delta.

Sources: [completed runs](../nonsubmissions/181026/README.md), [official score](../submissions/yolo26m_p2_candidate_330230/scoring_result/scores.json).

## Current class failures

Fresh analysis of saved calibrated predictions on all 939 holdout images:

| Class | Class AP | Matched recall at IoU .50, score >= .001 | Matched recall at IoU .75, score >= .001 | Matched recall at IoU .50, score >= .25 |
|---|---:|---:|---:|---:|
| Bridge | 6.24 | 50.7 | 17.3 | 10.9 |
| Playground | 9.95 | 58.7 | 26.2 | 20.6 |
| Wind turbine | 40.64 | 97.3 | 56.4 | 91.2 |
| Oil tank | 42.97 | 89.7 | 61.6 | 63.4 |
| Ship | 45.80 | 94.7 | 56.7 | 86.7 |

Recall uses greedy same-class matching on already exported predictions; it is diagnostic, not COCO AP or an oracle ceiling. These images include calibration-fit images. AP comes from the existing metrics, not a new AP computation. [Diagnostic counts](../results/delta45_20261018/saved_prediction_diagnostics.json).

Bridge/playground have a mixture of missed objects, poor ranking and loose boxes. Wind turbines are mainly a localization problem. Lowering display confidence cannot recover AP already evaluated with low-confidence predictions.

Class AP is equally weighted across nine classes. A 15-point improvement in each of bridge and playground adds 3.33 overall mAP points if every other class stays constant. This is a headroom calculation, not an expected result.

## YOLO experiments in priority order

1. **One targeted sampling screen from the first continuation.** Repeat bridge/playground training images twice using the existing `--repeat-classes bridge,playground --repeat-factor 2` option. Preserve full scenes, which contain the context that defines bridges. Keep the remaining recipe fixed. These classes are not rare by image count; this deliberately reallocates training emphasis and might also overfit noisy labels. Track all nine APs, full and clean mAP, and optimizer steps: repeating images increases steps per epoch, so the old continuation is a practical reference rather than a compute-matched causal control.
2. **If sampling fails, test context-preserving training crops.** Mix full-image training with moderate crops resized to 512, preserving substantial surroundings and transforming all intersecting labels correctly. This aims to teach weak small objects at a more learnable scale. Avoid tiny isolated bridge cutouts; they remove road/river context. Validation and submission remain one full image at 512. This augmentation is a proposal and is not implemented in the current trainer.
3. **Finish the strongest detector properly.** Refit the 1024-unit image classifier on its own frozen features and cross-fit new box corrections. Do not reuse the old detector's offsets without evaluation. The speckle checkpoint is a candidate for this inexpensive completion step, not a proven replacement.

Existing cluster command for the sampling screen (not launched in this session):

```bash
sbatch sbatch/finetune_detector.sbatch bridge-playground --repeat-classes bridge,playground --repeat-factor 2
```

Use detection AP for screening; the scoring classifier is reused. As a practical continuation gate, seek approximately +1 full mAP point with a positive clean gain and no large loss in strong classes. This gate is a time-allocation heuristic, not a significance test. A marginal result should not trigger a whole new long schedule.

## Optional alternative models

Published parameter counts below cover the detector; count the actual nine-class model plus image head before training. Reported optical-dataset AP is not a forecast of SAR AP at 512.

| Priority | Candidate | Published detector parameters | Initialization and reason |
|---|---|---:|---|
| First alternative | TinyFormer-M-PBM | 20.1M, VisDrone variant | Author lists COCO-to-VisDrone weights; tiny-detail architecture and aerial initialization |
| Fallback | D-FINE-M | 19M | Objects365 or Objects365-to-COCO weights; distribution-based box refinement |
| Reserve | DEIMv2-M / L | 18.1M / 32.2M | DINOv3-derived backbone and COCO detector; spatial adapter makes patch size alone insufficient to dismiss it |

[TinyFormer repository](https://github.com/mmpmmpmmpjosh/TinyFormer) lists the VisDrone checkpoints and a Google Drive mirror. Its [M-PBM configuration](https://github.com/mmpmmpmmpjosh/TinyFormer/blob/main/configs/tinyformer/tinyformer_dinov3_m_coco_pbm.yml) uses stride-4 features in its encoder. Checkpoint links were found but weights were not downloaded or loaded; the browsing tool could not retrieve the VisDrone configuration or Drive folder. Treat startup as unresolved. Its requirements pin an older Torch version; preserve the cluster's working Blackwell-compatible Torch environment and check compatibility in isolation.

[D-FINE model zoo](https://github.com/Peterande/D-FINE) provides a direct `dfine_m_obj2coco.pth` release and custom-data fine-tuning support. [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) provides released M/L models and input-size configuration guidance. All require a 512 input configuration and a shared-feature image classification head for this benchmark.

Give a new framework a two-hour setup budget; then use a working fallback if checkpoint loading or dependencies remain blocked. Do not run all these alternatives while Dome occupies one of the two cluster slots.

## Time allocation

- First 6–8 hours: one targeted YOLO screen and completion of the speckle candidate's head/calibration where resources permit. Read exact exported full/clean metrics before deciding.
- Following 16–24 hours: spend the available slot on the most promising targeted YOLO change, or one alternative if YOLO shows no meaningful gain. Dome remains the user's independent track.
- Final 8–12 hours: freeze experiments, fit the final classifier/calibration, audit parameter counts and one-pass export, and obtain an official score with time left to fix packaging.

Do not add holdout images to training until model/recipe selection is finished. If fitting on all labelled data, holdout scores cease to be validation evidence, and the final recipe must use a fixed training duration rather than selecting on those scores.

Status: repository results reviewed and saved predictions analyzed; external model documentation researched. No GPU jobs, checkpoint downloads, or submissions were launched.
