# Phase 3: joint fine-tuning plan

**Status:** ready to submit; neither fine-tune has been submitted or queued. These are local holdout experiments, not official benchmark submissions.

## Existing runs and results

| Reference | Existing training | Local holdout macro-F1 | Local holdout mAP | Local Δm | Checkpoint handoff |
|---|---|---:|---:|---:|---|
| E5 YOLO26m-P2 | 100-epoch detector, then classifier fitted on frozen detector features | 91.08% | 40.67% | +26.63% | `yolo_baseline/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt` and `E5_yolo26m_image_classifier.pt` |
| E7 RT-DETR-L-P2 | 100-epoch detector, then classifier fitted on frozen detector features | 91.34% | 40.80% | +27.02% | `yolo_baseline/phase3_assets/checkpoints/E7_rtdetr_l_p2_best.pt` and `E7_rtdetr_l_image_classifier.pt` |
| YOLO26s E2 | 15-epoch joint fine-tuning from its sequential detector and classifier | 93.46% | 38.09% | +22.96% | `runs/yolo26s-p2-E2-328622/`; metrics in `submissions/sar-experiments/E2-328622/` |

The completed sequential YOLO26m-P2 run is higher than the sequential YOLO26s-P2 run on both macro-F1 (91.08% vs 90.10%) and mAP (40.67% vs 37.47%). YOLO26s E2's 93.46% macro-F1 came from joint fine-tuning and had 38.09% mAP, so it is evidence that joint training can shift the balance between tasks; it does not show that the larger model has worse classification.

The source records for E5, E7, and E2 are under `submissions/sar-experiments/`. E7's baseline comparison found that using the image classifier to relabel or score-weight every detection reduced mAP. The Phase 3 outputs therefore retain the detector's native object classes.

## Shared method

Run two separate 15-epoch joint fine-tunes, each initialized from that architecture's own trained detector and image-classifier checkpoints. Keep the saved 8,453/939 image split, seed 0, 512 × 512 inputs, nine classes, and disabled mosaic/mixup/copy-paste augmentations. Train detection and image-classification cross-entropy together with classification loss weight 0.5. Validate both tasks after each epoch and save the checkpoint with the best local combined Δm. Record macro-F1, mAP, Δm, per-class F1/AP, GPU hours, and peak VRAM.

Do not freeze the detector. Joint fine-tuning updates the shared feature extractor, detector prediction modules, and separate image classifier. Detection and image-classification gradients both pass through shared features. RT-DETR already has object class and box outputs; the added image classifier predicts one class for the whole image.

Start with the completed baseline batch sizes: 16 for YOLO26m and 8 for RT-DETR. E7 ran on a 24 GB MIG slice with 8.75 GiB peak reserved at batch 8. E5 completed at batch 16, but its run did not record allocator peak VRAM. The job records actual allocated and reserved peaks. Do not assume the full 96 GB physical GPU is available when Slurm assigns a MIG slice.

## Packaged handoff

The source dataset archives are already Git LFS files under `data/`. The 8,453/939 split is tracked at `results/yolo26m-100-p2-328339/seed_0/split.json`. The converter is `yolo_baseline/prepare_dataset.py`. The four E5/E7 model and classifier checkpoints are now staged under `yolo_baseline/phase3_assets/checkpoints/` and tracked with Git LFS, so the new GPU does not need this machine's ignored `runs/` directory.

The batch job creates a dedicated `.phase3-venv` from `requirements-phase3.txt` if needed, fetches the LFS files, extracts course data, regenerates `data/yolo11s-sar-512`, runs the RT-DETR joint preflight, and then trains both models in one allocation. From a checkout of the pushed repository, submit it with:

```bash
sbatch sbatch/phase3_joint_finetuning.sbatch
```

This one job requests the student partition's Blackwell GPU allocation, batch size 16/8, and a maximum of 24 hours. Outputs are written to `runs/phase3-yolo26m-joint-<jobid>/` and `runs/phase3-rtdetr-l-joint-<jobid>/`. Each includes `train/weights/best.pt`, the matched `image_classifier.pt`, the training config, per-epoch `train/results.csv`, best per-class classifier F1 and detector AP metrics, and compute/VRAM records. The RT-DETR directory also includes `preflight.json`, which records label alignment, live multi-scale feature capture, gradients from both losses into the shared backbone, validation output, and custom-checkpoint save/reload.

The job does not submit to the leaderboard or generate official validation predictions. After it finishes, compare each result first against its own sequential baseline (E5 or E7), then compare the two Phase 3 models by local Δm. The 939-image holdout is for local selection; it is not an official benchmark score.
