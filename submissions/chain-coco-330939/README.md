# chain-coco-330939: COCO-pretrained YOLO26m-P2 with the speckle-cand chain

Submission: [chain-coco-330939.zip](chain-coco-330939.zip). Official **Δm +40.45** (mAP 47.04%, macro-F1 94.12%).

Same recipe as [chain-dota-330938](../chain-dota-330938/) (+40.73) but started from Ultralytics' COCO weights
(`yolo26m.pt`; job 330705 task 1, chain job 330939). Holdout with the nudge: mAP 49.05% full / 45.93% clean,
macro-F1 94.93% / 93.71%, Δm +44.72 / +38.11. Officially the two starts tie: COCO 0.21 lower mAP, 0.22 higher F1.

Uses the detector predictions in `/common/scratch/users/j/jgwong.2023/sar-benchmark/results/chain-coco-330939/one2many`, with classification predictions copied from
`/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/chain-coco-330939-package/chain-coco-330939_base.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **0.944, 1.100, 1.057, 0.885**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.
