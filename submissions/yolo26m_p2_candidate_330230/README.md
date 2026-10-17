# YOLO26m-P2 candidate with wider classifier and box correction

Submission: [yolo26m_p2_candidate_330230.zip](yolo26m_p2_candidate_330230.zip). Official score pending.

Uses the frozen detector from the first continuation (that original model scored
official Delta m 34.5697), with classification predictions copied from
`submissions/yolo26m_p2_quick_head_330230/yolo26m_p2_quick_head_330230.zip`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

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

## Candidate and local results

This is the candidate from GPU job 330230: the original first-continuation
YOLO26m-P2 detector, a new 1024-hidden-unit classifier on its frozen shared
features, and the conservative box-edge correction. Total parameters: 29,748,913.
The classifier was trained on the same 8,453 images, with training flips only;
checkpoint selection uses clean macro-F1 without a full-holdout F1 regression.
The selected wider head is epoch 234 of 240. Its checkpoint is `image_classifier.pt`.

| Local metric | Full holdout (939) | Clean subset (720) |
|---|---:|---:|
| Macro-F1 | 96.0566% | 95.0974% |
| Detection mAP | 45.5838% | 42.7081% |
| Delta m | 38.6969 | 32.6802 |

These scores include calibration fit images. They are local selection scores,
not independent estimates or official scores. The classifier-only exact submission
path reproduces F1 96.0566% / 95.0974% and unchanged original detection mAP.
The separate grouped cross-validation checks above support the box correction.
`candidate.json` records classifier provenance and parameter count. The candidate ZIP
was checked for all 1,426 image labels, valid boxes, archive integrity, and hashes;
its boxes match the separately verified box-only candidate, with updated labels.

The newly pulled `yolo26m-map-only-groups-20261017` candidate reports local Delta m
30.1695, F1 93.6225%, and mAP 41.8091%; it is not the recommended replacement.
Official scoring is pending for this candidate. DenoDet continues separately.
