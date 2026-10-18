# chain-dota-l-331166: conservative box-edge correction

Submission: [chain-dota-l-331166.zip](chain-dota-l-331166.zip). Official score pending.

Uses the detector predictions in `/common/scratch/users/j/jgwong.2023/sar-benchmark/results/chain-dota-l-331166/one2many`, with classification predictions copied from
`/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/chain-dota-l-331166-package/chain-dota-l-331166_base.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **0.880, 0.997, 1.420, 1.180**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.
