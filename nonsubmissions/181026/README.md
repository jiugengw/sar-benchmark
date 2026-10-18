# Completed experiments — 18 October 2026

All four jobs completed successfully. Scores below are local holdout scores using the one2many head; no official score is claimed. All packages remain unsubmitted.

| Experiment | Full mAP (%) | Full macro-F1 (%) | Full Δm | Clean Δm | Course-validation ZIP |
|---|---:|---:|---:|---:|---|
| [yolo26m-200-p2-330445](yolo26m-200-p2-330445/) | 42.2757 | 96.0526 | +32.3646 | +27.9911 | Saved |
| [ft40-nwd-330446](ft40-nwd-330446/) | 44.7466 | 93.8246 | +35.8988 | +28.9021 | Not generated |
| [ft40-speckle-330447](ft40-speckle-330447/) | 45.6923 | 92.7424 | +37.1285 | +29.2964 | Not generated |
| [ft40-p1-330448](ft40-p1-330448/p1/) | 44.4205 | 94.4202 | +35.5939 | +29.9037 | Saved |

The 200-epoch run starts from COCO weights. The other three extend `continue_1` of job 329644 for 40 epochs. NWD and speckle reuse the source image classifier; the stride-2 run trains its own. These classification differences affect Δm.

Raw holdout predictions and annotations remain in `results/<experiment>/`; per-epoch histories remain in `results/training-curves/<experiment>/`. Each experiment directory here contains the full and clean scores, Slurm completion summary, available run metadata, and links to those artifacts.

[Completed DenoDet experiment](denodet-v2-329477/) is also archived for this date.
