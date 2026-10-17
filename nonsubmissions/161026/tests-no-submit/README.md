# 16 October 2026 training comparison

Use the [interactive comparison report](../../../reports/training-comparison/index.html) to compare the completed 326894 and 326397 baselines with the 327063_0 targeted run and the completed YOLO baselines. Filter by job, recipe or status, then select which runs to compare. The 327063_1 detail-branch run and 327100 fixed-weight ablation were cancelled.

Job 326894 completed successfully. Its calibrated submission ZIP is available here:
[terramind_48epoch_baseline_calibrated.zip](tm-48baseline-326894/terramind_48epoch_baseline_calibrated.zip).

The 64-epoch calibrated baseline is also available here:
[terramind_64epoch_baseline_calibrated.zip](../../../submissions/terramind_64epoch_baseline_calibrated/terramind_64epoch_baseline_calibrated.zip).

## YOLO prediction ZIPs — generated, not uploaded

The YOLO comparison will create one separate predictions ZIP for each completed model (YOLO11s/m/n/l and YOLO26s/m/l). Each archive contains the project's required `classification.csv` and `detection.json` for all 1,426 images in `data/course-data/val/`. ZIPs and inference sidecars are stored under a model-specific `yolo*-baseline-*` folder here. The highest-confidence detection supplies the image class; when a model returns no boxes, the most frequent image class from the course training labels is used as the required fallback. No ZIP is uploaded automatically.

To add future runs, edit reports/training-comparison/runs.json and rebuild the report with the script described in that folder.
