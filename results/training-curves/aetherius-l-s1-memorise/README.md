# YOLO26l-P2 at stride 1: full chain (25 October 2026)

The 100-epoch stage ran on the cluster (L40S, batch 8; `yolo26l-p2-dota-s1-336693`). The four fine-tune stages
(plain -> speckle -> third (x4) -> memorise) ran on a borrowed RTX 3090 (24 GB, batch 8, the most that fits), with the
same recipes as the m stride-1 chain. Holdout mAP50-95 during training (best epoch of each stage):

| Stage | m stride 1 (L40S, batch 16) | L stride 1 (batch 8) |
|---|---|---|
| 100-epoch | 48.7 | 49.5 |
| plain | 50.6 | 51.4 |
| speckle | 52.2 | 52.1 |
| third (x4) | 52.7 | 52.7 |
| memorise | **57.9** | 57.1 |

L led by about +0.8 through the plain stage, was level in speckle and third, and finished about 0.8 behind after
memorise, so the m stride-1 model is the one carried to the all-data runs. Batch 8 (noisier batch-norm statistics) may
account for part of the gap. The memorise weights stay on the 3090 (`~/sar-live/runs/ft40-a-ls1-memorise/train/
weights/best.pt`, about 156 MB, over GitHub's file limit).
