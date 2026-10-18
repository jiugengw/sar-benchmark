# Dome-DETR

Dome-DETR-M (ACM MM 2025, [code](https://github.com/RicePasteM/Dome-DETR), Apache-2.0), a
D-FINE detector for tiny objects that places more queries where a predicted density map is high. 24.4M parameters.

**Pretraining (no SAR):** HGNetV2-B2 backbone on ImageNet, then the whole detector on AI-TOD v2 (optical aerial
photos). We fine-tune the authors' AI-TOD checkpoint on our training split.

| File | Purpose |
|---|---|
| `scripts/setup_dome.sh` | Login node, once: clone the pinned commit into `third_party/`, make `.dome-venv`, download and verify the checkpoint |
| `prepare_data.py` | COCO files for our 8453 / 939 split; our classes go into the matching AI-TOD class slots (`common.py`) |
| `transforms.py` | 512 × 512 canvas **without upscaling**: larger images shrink, 256 px images keep their size and are padded |
| `train.py` | Writes the config on top of the authors' Dome-M AI-TOD config and runs their `train.py` in fine-tune mode |
| `export_predictions.py` | One pass per holdout image, boxes back to original pixels, full / clean scores |
| `sbatch/dome_detr.sbatch` | Train (48 epochs, batch 8 by default), then export and score |

```bash
bash scripts/setup_dome.sh
sbatch sbatch/dome_detr.sbatch
```

Dome has no image classifier. The exporter gives each image the class of its top box for a rough F1, so judge the run
on detection mAP. The YOLO control is 44.98 full / 41.89 clean.
