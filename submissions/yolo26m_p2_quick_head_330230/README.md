# YOLO26m-P2 quick classifier ablation

GPU job 330230 completed in 9 minutes 3 seconds. The detector is frozen at the
first continuation checkpoint (original official Delta m 34.5697). Cached shared
features train two heads: a warm-started 512-unit head and a fresh 1024-unit head.
The wider head at epoch 234 is selected by clean-holdout macro-F1, requiring
full-holdout F1 to be at least the original. No holdout images receive gradients.

The exact submission generator reproduces macro-F1 96.0566% (939 full images)
and 95.0974% (720 clean images). Detection mAP is unchanged at 44.7547% full and
41.7335% clean; local Delta m is 37.1103 full and 30.8152 clean. Selection uses
these images, so these are not independent estimates. Official scoring is pending.

This ZIP has the new classifier only. The recommended combined classifier and
box-correction ZIP is in `../yolo26m_p2_final_330230/`, which also preserves the
selected `image_classifier.pt` checkpoint. Code: `yolo_baseline/quick_classifier.py`;
full head histories and baseline comparisons are in `candidate.json` and matching
`results/yolo26m_p2_quick_head_330230/`. No ensemble or test-time augmentation.
