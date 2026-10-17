# Ultralytics YOLO SAR comparisons

The best submission (YOLO26m-P2, official Δm +36.80) comes from this folder. The [code map](../../README.md#code-structure) lists the four scripts that built it, in order; [docs/HISTORY.md](../../docs/HISTORY.md) explains each step. The sections below document the earlier YOLO comparison phases.

These runs compare the native Ultralytics detector against TerraMind + RPN/ROI. Every run uses the same saved 8,453/939 train/holdout split, 512 × 512 grayscale SAR images, nine classes, 24 epochs, batch size 16, and pretrained weights. Ultralytics expands grayscale images to three equal channels for its pretrained model.

## Selected models

- YOLO11s is the first run, as initially requested.
- YOLO11m, YOLO11n, and YOLO11l extend the size comparison. YOLO11x is excluded because its published 56.9M parameters exceed the 40M limit.
- YOLO26s, YOLO26m, and YOLO26l compare the newer model generation at three capacities. YOLO26l is within the project's 40M parameter cap.
- YOLOv8 and YOLO12 are excluded from this run set.

All selected models are fully fine-tuned from their pretrained weights: the backbone, neck, and detection head train, with no LoRA or frozen layers. The detection head is adapted to the project's nine classes. The experiment records total and trainable parameter counts, Ultralytics version, and training settings.

## Data and outputs

`prepare_dataset.py` reads COCO annotations, `labels.csv`, and the saved TerraMind split at `runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json`. It writes the same train and holdout image IDs to `data/yolo11s-sar-512`. Images use grayscale bilinear stretch resizing to 512 × 512, matching the project loader. Normalized YOLO boxes are checked by round-trip conversion against their source COCO boxes.

Each run saves under `runs/<model>-baseline-<job-id>/` and `results/<model>-baseline-<job-id>/seed_0/`. Metrics include detection mAP and per-class AP, precision/recall at confidence 0.25 and IoU 0.50, image classification derived from the highest-confidence detection, batch-one holdout speed against the TerraMind baseline, and a same-image visual comparison. The comparison report discovers completed runs and allows selecting which models to compare. `make_all_submissions.py` creates a separate project-format predictions ZIP for every completed YOLO model using all 1,426 official course validation images; it does not submit them.

`sbatch/yolo11s_baseline.sbatch` launches the first run. `sbatch/yolo_variant.sbatch` launches other selected variants with `YOLO_MODEL` set through Slurm's `--export` option. The fixed-weight 48-epoch TerraMind run is scheduled after the final YOLO job.

## Matched 100-epoch YOLO11s standard/P2 comparison

`sbatch/yolo11s_100.sbatch` runs either `YOLO_VARIANT=standard` or `YOLO_VARIANT=p2`. Both use the same official YOLO11s initialization, saved 8,453/939 split, 512 × 512 inputs, batch size 16, Ultralytics optimizer and augmentation defaults, seed 0, and 100 detector epochs. P2 adds a stride-4 detection level; the shared pretrained backbone, neck, and P3-P5 detection towers transfer from the standard checkpoint. Each detector then gets a supervised image-classification head trained on the 8,453 training labels for a fixed 100 epochs. The 939-image holdout is evaluated once for image macro-F1 and for native box classes versus assigning the classifier's top class to every box, with and without multiplying box scores by the classifier confidence. No policy is selected automatically from the holdout.

The results include COCO mAP, AP50/AP75, AP small, AR small, per-class AP for every box-class policy, and each policy's change from native classes. The comparison page displays these values alongside the classifier head's image-level macro-F1. To use a classifier in a course-format ZIP after choosing a policy, pass `--classifier-checkpoint` and one of `--box-class-policy classifier_class_all_boxes` or `classifier_class_score_weighted` to `models/terramind/make_submission.py`.

## Matched 100-epoch YOLO26 P2 comparison

`sbatch/yolo26_all_p2_100.sbatch` trains YOLO26s, YOLO26m, and YOLO26l with a stride-4 P2 level, up to two scales concurrently. Every run uses the same 8,453/939 split, 512 × 512 inputs, batch size 16, seed 0, 100 detector epochs, and classifier/evaluation stages as YOLO11s-P2. The shared pretrained backbone, P3 neck, P4/P5 neck, and P3-P5 detection heads transfer from the matching official YOLO26 scale; the new P2 branch starts with random weights. Training code is in `train_yolo26s_p2.py`; each run saves under `runs/yolo26<scale>-100-p2-<job-id>/` and `results/yolo26<scale>-100-p2-<job-id>/seed_0/`.

`sbatch/yolo11n_p2_100.sbatch` trains YOLO11n-P2 for 100 epochs using the local official YOLO11n checkpoint. `sbatch/yolo26n_p2_100.sbatch` trains YOLO26n-P2 for 100 epochs and resolves the official YOLO26n checkpoint through Ultralytics on the compute node because it is not staged in `pretrained/`.

The unqueued 200-epoch detector templates are `sbatch/yolo11_p2_200.sbatch` and `sbatch/yolo26_p2_200.sbatch`. Each accepts one `MODEL_SCALE` and runs only that selected model; both keep the separate image classifier at 100 epochs and package a submission ZIP after evaluation.

## SAR classification and detection experiment plan

This plan tracks the proposed comparison of classification macro-F1, detection COCO mAP, and the benchmark's combined `Δm`. Keep the existing YOLO11s-P2 checkpoints, settings, metrics, and logs as the E1 baseline; write every new run to a unique run directory.

### Confirmed E1 implementation

E1 is the 100-epoch YOLO11s-P2 detector followed by a separately trained image-classification head. The detector starts from `pretrained/yolo11s.pt`, transfers compatible backbone, neck, and P3-P5 detection weights, and initializes the new P2 branch randomly. The saved detector is frozen while the classifier is trained from the same image labels.

`train_image_head.py` hooks the inputs to YOLO's final `Detect` module. For P2, these are the four stride-4/8/16/32 maps. It global-average-pools each map, concatenates the channel vectors, and feeds a `LayerNorm + Dropout(0.2) + Linear` head. E1 artifacts are under `runs/yolo11s-100-p2-328058/` and `results/yolo11s-100-p2-328058/seed_0/`; its submission ZIP is `submissions/yolo11s_100_p2_328058/yolo11s_100_p2_328058.zip`. Local holdout scores are macro-F1 **89.30%**, mAP **36.31%**, and `Δm` **+17.34%** against the repository's fixed references. The value is a one-seed local comparison, not an official benchmark score. The saved classifier currently uses the fixed final epoch; future experiments should select a best-validation checkpoint using the combined metric.

### Metric and controls

Use the benchmark formula already documented in the repository:

```text
Δm = 100 × ½ × [(macro-F1 − 93.31%) / 93.31% + (mAP − 26.13%) / 26.13%]
```

Report scores in percentage points when applying this form. The fixed references are from the TA's validation benchmark. A score computed on the project's 939-image labeled holdout is a **local comparison value**, not an official benchmark score; keep its split label beside the result. Apply the same formula and evaluation implementation to every experiment.

Hold constant the saved 8,453/939 image-ID split, 512 × 512 grayscale inputs where supported, nine-class mapping, and seed 0 for controlled comparisons. Record pretrained checkpoint paths and hashes, architecture and stride levels, total/trainable parameters, learning-rate groups, augmentations, epochs, per-class F1/AP, peak VRAM, and GPU-hours. Use architecture-appropriate optimizers and schedules. Select and preserve the best validation checkpoint by `Δm`, while reporting both task metrics so negative transfer is visible.

Every completed experiment must also create its own course-format prediction ZIP for all 1,426 official validation images, with the matching metrics file and experiment notes beside it under `submissions/`. ZIP creation is packaging only; it does not submit the model.

### Phase 1 — YOLO26s-P2 training strategies

The requested grouping is one Phase 1 Slurm job bundle for E2, E3, and conditional E4. The user selected uninterrupted execution within each bundle, so the job runs E2 then E3 and proceeds to E4 when its predeclared mAP criterion is met. Run each stage in a distinct output directory and retain its best checkpoint, logs, metrics, and submission ZIP.

| ID | Run | Implementation and comparison |
|---|---|---|
| E1 | Existing YOLO26s sequential baseline | Use the completed 100-epoch YOLO26s-P2 detector and frozen-feature classifier at `runs/yolo26s-100-p2-328338/` as the Phase 1 reference. |
| E2 | Joint fine-tuning | Initialize both trained YOLO26s-P2 E1 heads. Unfreeze the shared backbone and neck; apply detection plus weighted classification loss, with a lower learning rate for shared pretrained layers than either head. Train for 15 epochs and select by validation `Δm`. |
| E3 | Joint from initialization | Start from official pretrained YOLO26s weights, transfer matching layers and P3-P5 detection heads into YOLO26s-P2, randomly initialize P2 and the classifier, and train both tasks together for 100 epochs. Compare against E1 and E2. |
| E4 | Extended sequential | Continue from the YOLO26s-P2 E1 final detector weights only if its validation mAP curve suggests further improvement, then train 100 more epochs and retrain the classifier on frozen features. This completed E1 checkpoint has no optimizer state, so E4 initializes a fresh optimizer; it is a 200-epoch cumulative weight continuation, not an optimizer-state resume. |

Joint training must align each detector batch with image-level labels, omit classification loss for samples without valid class labels, and verify that gradients from both losses reach the shared backbone. E4's automatic continuation gate is: continue only when E1 mAP50-95 increased by at least 0.5 percentage points over its final 10 training epochs. The YOLO26s-P2 E1 run improved from 36.962% at epoch 91 to 38.070% at epoch 100 (+1.108 points), so it passes the gate.

### Phase 2 — architecture comparison

The requested grouping is one architecture-comparison Slurm job bundle. E5 (YOLO26m-P2) completed as array task `328337_1`; its local holdout metrics, compute record, and course-format ZIP are archived under `submissions/E5_yolo26m_p2_328339/`.

| ID | Architecture | Priority | Plan |
|---|---|---|---|
| E5 | YOLO26m + P2 | High | Completed as array task `328337_1`; result artifacts are archived under `submissions/E5_yolo26m_p2_328339/`. |
| E6 | YOLO26n + P2 | Medium | Use the matching pretrained checkpoint and confirm P2 use. |
| E7 | RT-DETR + P2 | Medium | Completed as job `328598`; result artifacts are archived under `submissions/E7_328598/`. |
| E8 | Deformable DETR + P2 | Dropped | Training code is retained. The 50-epoch run was stopped at epoch 3 after the measured pace was about 41 minutes per epoch, projecting roughly 34 hours and exceeding the student partition's 24-hour job limit. The available GPU was a 24 GB MIG slice; at 512 × 512 with four feature levels, batch size 4 was needed to fit, leaving too little VRAM for a larger batch. Partial generated run files were removed and are not treated as results. |
| E9 | YOLO11n + P2 | Completed, job `329376` | Use the matching pretrained checkpoint and the same sequential evaluation method. The E9-only runner is `sbatch/phase2_e9_yolo11n.sbatch`. |

Each architecture starts with sequential detector training, frozen detector features, and a separately trained classifier. Do not assume YOLO hooks or feature dimensions transfer to either DETR implementation. The RT-DETR and Deformable DETR stages must save their standard-pretrained validation results before training their P2 variants.

An additional completed comparison run, YOLO26s-P2 array task `328337_0`, scored **90.10% macro-F1** and **37.47% mAP** on the saved 939-image holdout. The metrics are at `results/yolo26s-100-p2-328338/seed_0/metrics.json`; YOLO26s is an additional result, not E5. The completed E5 YOLO26m-P2 run scored **91.08% macro-F1**, **40.67% mAP**, and **+26.63% local Δm**. Its detector and classifier checkpoints are in `runs/yolo26m-100-p2-328339/`; metrics and ZIP are under `submissions/E5_yolo26m_p2_328339/`. E7 RT-DETR-L-P2 scored **91.34% macro-F1**, **40.80% mAP**, and **+27.02% local Δm**; its metrics, compute record, and ZIP are under `submissions/E7_328598/`. A separate YOLO26l-P2 run, job `329019`, completed outside the Phase 2 architecture comparison with **91.14% macro-F1**, **40.24% mAP**, and **+25.83% local Δm**; its artifacts are under `nonsubmissions/sar-experiments/yolo26l-100-p2-329019/`. Array element `_2` was cancelled before starting, so YOLO26l remains a supplementary run.

### Phase 3 — joint fine-tuning of YOLO26m-P2 and RT-DETR-L-P2

**Outcome:** the YOLO26m joint fine-tune scored official Δm +23.80 (F1 up 2.8 points, mAP down 0.5 against E5). It was superseded by detector-only continuation plus a separate image head (`deadline_recovery.py`, official +34.57, then +36.80 with a wider head and box-edge correction). The plan below is kept as a record; see [docs/HISTORY.md](../../docs/HISTORY.md).

The current leading completed candidates are E7 RT-DETR-L-P2 (**91.34% macro-F1**, **40.80% mAP**, **+27.02% local Δm**) and E5 YOLO26m-P2 (**91.08%**, **40.67%**, **+26.63%**). Fine-tune each from its own completed detector and image-classifier checkpoints. This phase is joint fine-tuning, not training from initialization. Its settings and results are in [Part 4 of docs/HISTORY.md](../../docs/HISTORY.md#part-4-phase-3-joint-fine-tuning-of-e5-oct-17); run it with `sbatch sbatch/phase3_joint_finetuning.sbatch`. The YOLO26m run `phase3-yolo26m-joint-20261017-1516` scored official Δm +23.80.

### Execution gates and outputs

Before expensive training, add configurable shared pipeline support for sequential and joint strategies, then validate a small batch: label alignment, actual P2 feature capture, both-loss gradient flow for joint training, checkpoint save/reload, metric calculation, and ZIP contents. The two requested Slurm bundles run their listed experiments uninterrupted after this smoke check. E4 remains conditional on the stated trend gate. Every completed experiment writes its own metrics, logs, best checkpoint, and course-format submission ZIP before the bundle advances.

The Phase 1 bundle is `sbatch/sar_phase1_yolo26s_p2.sbatch`. Phase 2 reuses E5 and runs E6 YOLO26n-P2, E7 RT-DETR-L-P2, and E9 YOLO11n-P2. E8's training code remains available, while E8 is removed from the standard run sequence because its 50 epochs exceed the job time limit at the VRAM-limited batch size. The interrupted E5-E9 job was stopped during E8; E9 ran separately as job `329376` from `sbatch/phase2_e9_yolo11n.sbatch`. Phase 2 depends on E5, not on Phase 1 results.
