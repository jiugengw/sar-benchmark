# Phase 3: joint fine-tuning plan

**Status:** plan prepared; neither run has been submitted or queued. These are local holdout experiments, not official benchmark submissions.

## What is already complete

| Reference | Existing training | Local holdout macro-F1 | Local holdout mAP | Local Δm | Checkpoints |
|---|---|---:|---:|---:|---|
| E5 YOLO26m-P2 | 100-epoch detector, then image classifier on frozen detector features | 91.08% | 40.67% | +26.63% | `runs/yolo26m-100-p2-328339/train/weights/best.pt`; `runs/yolo26m-100-p2-328339/image_classifier.pt` |
| E7 RT-DETR-L-P2 | 100-epoch detector, then image classifier on frozen detector features | 91.34% | 40.80% | +27.02% | `runs/E7-rtdetr-l-p2-328598/train/weights/best.pt`; `runs/E7-rtdetr-l-p2-328598/image_classifier.pt` |
| YOLO26s E2 | 15-epoch joint fine-tuning from its own sequential YOLO26s-P2 detector and classifier | 93.46% | 38.09% | +22.96% | `runs/yolo26s-p2-E2-328622/`; metrics in `submissions/sar-experiments/E2-328622/` |

The completed sequential YOLO26m-P2 run is higher than the sequential YOLO26s-P2 run on both macro-F1 (91.08% vs 90.10%) and mAP (40.67% vs 37.47%). YOLO26s E2's 93.46% macro-F1 came from joint fine-tuning; it is a different training strategy, and its mAP was 38.09%. This comparison is evidence that joint training can shift the balance between tasks; it does not show that the larger model has worse classification.

The source records for E5, E7, and E2 are in `submissions/sar-experiments/`. E7's baseline comparison found that using the image classifier to relabel or score-weight every detection reduced mAP, so Phase 3 should keep the detector's native object classes for the primary result.

## Shared method

Run two separate 15-epoch joint fine-tunes, each initialized from that architecture's own detector and image-classifier checkpoints. Keep the saved 8,453/939 image split, seed 0, 512 × 512 inputs, nine classes, and disabled mosaic/mixup/copy-paste augmentations. Train the detection loss and image-classification cross-entropy loss together, starting with classification loss weight 0.5. Validate both tasks after each epoch and save the checkpoint with the best local combined `Δm`; report macro-F1, mAP, `Δm`, per-class F1/AP, GPU hours, and peak VRAM.

For this experiment, **do not freeze the detector**. The previous sequential method froze the whole detector while fitting the extra image classifier. Joint fine-tuning starts from both trained heads, then updates the shared feature extractor, the detector's class/box prediction modules, and the separate image classifier. The classification gradient and detection gradient both pass through shared features. RT-DETR's detector already has class and box outputs; the added image classifier predicts one class for the whole image.

Keep batch sizes at the completed baseline settings to start: 16 for YOLO26m and 8 for RT-DETR. E7 ran on a 24 GB MIG slice with 8.75 GiB peak reserved at batch 8. E5 completed at batch 16, but its run did not record allocator peak VRAM. Record actual allocated GPU/MIG memory for both fine-tunes; reduce batch only if joint training exceeds the assigned slice. Do not assume the full 96 GB physical GPU is available when Slurm has assigned a MIG slice.

## Run A: YOLO26m-P2 joint fine-tuning

The existing joint YOLO trainer supports the `yolo26` model family. A dedicated config is provided at `yolo_baseline/experiment_configs/phase3_yolo26m.yaml`; it initializes from the E5 detector and image-classifier checkpoints, uses the E5 architecture, and sets 15 epochs.

After the GPU allocation and shared dataset/checkpoints are available, run:

```bash
python -m yolo_baseline.train_joint \
  --config yolo_baseline/experiment_configs/phase3_yolo26m.yaml \
  --experiment E2 \
  --output runs/phase3-yolo26m-joint-finetune
```

The config uses AdamW, batch 16, `lr0=0.001`, a 0.1 learning-rate multiplier for shared backbone/neck layers, and classification loss weight 0.5. The existing trainer applies the base learning rate to the detector and image-classification heads and the lower rate to shared layers.

## Run B: RT-DETR-L-P2 joint fine-tuning

There is not yet an RT-DETR joint-training runner. Add an architecture-specific trainer before launching this run; the YOLO trainer cannot train RT-DETR. Start from the E7 detector and classifier checkpoints listed above. Reuse the same pooled multi-scale features that E7's frozen-feature classifier used, and make sure the feature hook remains connected to autograd. Align each detection batch to its image-level label, compute both losses, and verify each loss sends gradients through shared detector features.

Use the E7 detection settings as the starting point: AdamW, batch 8, 512 × 512, base detector learning rate `1e-4`, cosine schedule, and 15 epochs. Use a lower learning rate for the shared backbone/neck than the detector prediction modules. Keep a separate image-classifier learning rate and log the component losses. Save the best combined-`Δm` detector and classifier state together so inference uses the matched pair.

Before the full run, verify one training batch for label alignment, multi-scale feature capture, both-loss gradient flow, checkpoint save/reload, and one validation pass. This is a code/preflight requirement, not a separate training experiment.

## Outputs and comparison

Use separate output directories, for example `runs/phase3-yolo26m-joint-finetune/` and `runs/phase3-rtdetr-joint-finetune/`. Save the config, logs, best detector checkpoint, matched image-classifier checkpoint, per-epoch metrics, final holdout metrics, peak VRAM, and GPU time. Package a course-format ZIP for each final checkpoint under `submissions/sar-experiments/`; packaging does not submit it.

Compare each joint result first with its own sequential baseline (E5 or E7), then compare both joint models by local `Δm`. The 939-image holdout scores are for model selection and local comparison; label them as such and do not describe them as official benchmark scores.

The detector and image-classifier weights under `runs/` are local and are not included in Git. The GPU environment must be able to read those checkpoint files, the course dataset, and the saved split before starting.
