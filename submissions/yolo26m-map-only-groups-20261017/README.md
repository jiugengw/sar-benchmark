# YOLO26m-P2 map-only groups submission

## Package

`yolo26m_map_only_groups_20261017.zip` is the generated submission archive. It contains the project's `classification.csv` and `detection.json` files for 1,426 images, with 106,970 detections. The adjacent inference JSON records the archive format and checksums.

## Model and classifier

The detector is the best checkpoint from a 40-epoch detection-only fine-tuning run. Checkpoints were selected by standard validation fitness (mAP). The run continued from `runs/phase3-yolo26m-continue-20261017/train/weights/best.pt` and used the existing 939-image validation split.

The ZIP's image-classification predictions use the classification head carried forward unchanged from that joint source checkpoint. The map-only run trained the detector using detection loss only; it did not train a new classification head.

## Local validation score

Evaluation used the same 939-image validation split used for checkpoint selection, so these are local validation results rather than an independent test or an official leaderboard score.

- Combined Δm: **30.17**
- Detection mAP: **41.81%** (AP50 **68.29%**)
- Image-classification macro-F1: **93.62%**
- Image-classification accuracy: **93.08%**

Full evaluation details are in `validation_metrics.json`. No official score has been recorded for this package yet.
