# Phase 3 baseline checkpoints

These files are copies of the completed sequential E5 and E7 detector/classifier pairs. They are Git LFS tracked so a clean checkout can run Phase 3 without access to this machine's ignored `runs/` directory.

| Pair | Packaged detector | Packaged image classifier | Original run |
|---|---|---|---|
| E5 YOLO26m-P2 | `checkpoints/E5_yolo26m_p2_best.pt` | `checkpoints/E5_yolo26m_image_classifier.pt` | `runs/yolo26m-100-p2-328339/` |
| E7 RT-DETR-L-P2 | `checkpoints/E7_rtdetr_l_p2_best.pt` | `checkpoints/E7_rtdetr_l_image_classifier.pt` | `runs/E7-rtdetr-l-p2-328598/` |

The model files were copied byte for byte from the original runs. Their SHA-256 values are checked in `checkpoints/manifest.json`.
