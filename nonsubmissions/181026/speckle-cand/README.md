# YOLO26m-P2 with conservative box-edge correction

Submission: [speckle-cand.zip](speckle-cand.zip). Official score pending.

Uses the detector from the 40-epoch speckle continuation (job 330447), with a
refitted 1024-neuron image classifier and classification predictions copied from
`/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/speckle-cand-package/speckle-cand_base.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **0.571, 0.825, 0.913, 0.979**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records the detector, classifier, and inference settings.

Local calibrated scores: full holdout mAP **46.36%**, macro-F1 **95.48%**, Delta m
**+39.88**; clean holdout mAP **43.23%**, macro-F1 **94.47%**, Delta m **+33.34**.
Metrics and predictions are in [`results/speckle-cand/seed_0/`](../../../results/speckle-cand/seed_0/).
