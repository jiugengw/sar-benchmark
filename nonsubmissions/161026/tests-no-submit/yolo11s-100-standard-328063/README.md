# YOLO11s 100-epoch standard

This package uses the YOLO11s standard detector from Slurm job 328063 and its supervised image-classification head. The head supplies image labels in `classification.csv`; detector-native classes supply box labels in `detection.json`. Predictions cover all 1,426 official course validation images.

Local holdout metrics are recorded in `yolo11s_100_standard_328063.metrics.json` after package generation. The official score is not known. The ZIP is for repository review and is not uploaded to the course leaderboard.
