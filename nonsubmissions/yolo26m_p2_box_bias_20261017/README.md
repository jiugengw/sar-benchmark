# YOLO26m-P2 with conservative box-edge correction

Submission: [yolo26m_p2_box_bias_20261017.zip](yolo26m_p2_box_bias_20261017.zip). Official score pending.

Uses the first continuation's frozen detector and classifier (original official
Delta m 34.5697), followed by one deterministic box-edge correction. No ensemble,
extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **0.604, 0.814, 0.870, 0.841**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.
