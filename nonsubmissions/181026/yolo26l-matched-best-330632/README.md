# yolo26l-matched-best-330632: conservative box-edge correction

Submission: [yolo26l-matched-best-330632.zip](yolo26l-matched-best-330632.zip). Not uploaded; no official score.

Uses the detector predictions in `/common/scratch/users/j/jgwong.2023/sar-benchmark/results/yolo26l-matched-best-330632/one2many`, with classification predictions copied from
`/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/yolo26l-matched-best-330632-package/yolo26l-matched-best-330632_base.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **0.831, 1.028, 1.097, 0.879**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.

## Matched large result

Job 330632 completed successfully in 3:07:30. Both detector stages matched
medium's training arguments, with model size and output paths as the differences.
The initial 100-epoch models effectively trained from random initialization;
the SAR-trained weights were retained in subsequent stages.

| Score | Full holdout | Clean holdout |
|---|---:|---:|
| Uncalibrated mAP (%) | 45.1587 | 42.2190 |
| Refitted macro-F1 (%) | 95.1337 | 93.8595 |
| Uncalibrated Delta m | +37.3889 | +31.0808 |
| Calibrated mAP (%) | 46.0427 | 43.4033 |
| Calibrated Delta m | +39.0804 | +33.3471 |

`holdout.metrics.json` is the uncalibrated export. Calibrated metrics and
predictions are in [`results/yolo26l-matched-best-330632/seed_0/`](../../../results/yolo26l-matched-best-330632/seed_0/).
`architecture_comparison.json` compares both detectors before box calibration.
Calibrated scores include fit images. The full candidate uses the accepted correction.

[Completed training curves](../../../reports/yolo26-matched-size/comparison.png)
and [final result graph](../../../reports/yolo26-matched-size/final-results.png).
The detector is preserved in [Git LFS](../../../checkpoints/yolo26l-matched-best-330632/model.pt);
`candidate.json` records its hash. The classifier is `image_classifier.pt` here.
