# Ultralytics YOLO SAR comparisons

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

The results include COCO mAP, AP50/AP75, AP small, AR small, per-class AP for every box-class policy, and each policy's change from native classes. The comparison page displays these values alongside the classifier head's image-level macro-F1. To use a classifier in a course-format ZIP after choosing a policy, pass `--classifier-checkpoint` and one of `--box-class-policy classifier_class_all_boxes` or `classifier_class_score_weighted` to `make_submission.py`.

## Matched 100-epoch YOLO26 P2 comparison

`sbatch/yolo26_all_p2_100.sbatch` trains YOLO26s, YOLO26m, and YOLO26l with a stride-4 P2 level, up to two scales concurrently. Every run uses the same 8,453/939 split, 512 × 512 inputs, batch size 16, seed 0, 100 detector epochs, and classifier/evaluation stages as YOLO11s-P2. The shared pretrained backbone, P3 neck, P4/P5 neck, and P3-P5 detection heads transfer from the matching official YOLO26 scale; the new P2 branch starts with random weights. Training code is in `train_yolo26s_p2.py`; each run saves under `runs/yolo26<scale>-100-p2-<job-id>/` and `results/yolo26<scale>-100-p2-<job-id>/seed_0/`.
