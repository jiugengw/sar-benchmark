# Matched YOLO26l-P2 detector

`model.pt` is the Ultralytics best checkpoint from job 330632's speckle stage
(best epoch 35). It uses Git LFS. Fetch it with `git lfs pull` after cloning.
The original baseline effectively trained from random initialization, followed
by 40 plain and 40 speckle epochs on SAR. Both continuation stages retained the
learned weights. This file is a full Ultralytics checkpoint, not a bare state dictionary.

The matching classifier, calibrated validation ZIP and audit are in
[`nonsubmissions/181026/yolo26l-matched-best-330632/`](../../nonsubmissions/181026/yolo26l-matched-best-330632/).
`metrics.json` contains calibrated full and clean holdout scores, which include
calibration fit images. No official score is available.
