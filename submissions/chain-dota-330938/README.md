# chain-dota-330938: DOTA-pretrained YOLO26m-P2 with the speckle-cand chain

Submission: [chain-dota-330938.zip](chain-dota-330938.zip). Official **Δm +40.73** (mAP 47.25%, macro-F1 93.90%).

Detector: YOLO26m-P2 started from Ultralytics' DOTA weights (`yolo26m-obb.pt`, optical aerial photos; job 330705),
100 epochs, then 40 plain and 40 speckle epochs (job 330938, `sbatch/continue_chain.sbatch`). Holdout with the
nudge: mAP 48.78% full / 45.22% clean, macro-F1 95.96% / 94.85%, Δm +44.76 / +37.36.

Uses the detector predictions in `/common/scratch/users/j/jgwong.2023/sar-benchmark/results/chain-dota-330938/one2many`, with classification predictions copied from
`/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/chain-dota-330938-package/chain-dota-330938_base.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **1.156, 0.867, 1.231, 1.383**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.
