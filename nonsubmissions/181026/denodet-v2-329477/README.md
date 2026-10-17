# DenoDet V2, 48 epochs

`denodet-v2-329477.zip` covers all 1,426 course validation images. It has not been
submitted to Codabench. Best checkpoint: epoch 48; resume job 330220 completed
successfully. Initialization is ImageNet-only, all 32,939,931 parameters trained,
one shared backbone, one image pass at 512px, no ensemble or test-time augmentation.

| Local score | Full holdout (939) | Clean subset (720) |
|---|---:|---:|
| Macro-F1 | 93.7500% | 91.9947% |
| Detection mAP | 37.4147% | 32.7103% |
| Delta m | 21.8292 | 11.8866 |

The epoch-32 snapshot scored Delta m 14.2341 / 6.2251. The completed schedule
improves both subsets but stays below YOLO26m. These are local scores only.

The adjacent `run.json` links all 48 epochs, shown in `curves.png`.
Per-epoch mAP is rounded by MMEngine and uses 512px annotations; exported scores
rescale detections to original image pixels. Training loss averages logged
moving windows. Validation loss, per-image holdout classification predictions,
and inference timing were not saved by the original exporter; they are not
reconstructed or invented. `inference.json` records ZIP/checkpoint hashes.
Full settings, split, predictions, compute and history provenance are under
`results/denodet-v2-329477/seed_0/`. The saved compute record covers only the last
resume allocation; earlier failed/recovery jobs also count toward total cost.
