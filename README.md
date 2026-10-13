# CS701 SAR Multi-Task Benchmark

Course-data training code adapted from the TA's CS701 reference implementation. One ViT-B/16 backbone, two tasks
on SAR images: **classify** the image (9 classes) and **detect** its objects. The current trainer uses a labeled
training holdout; the TA's original benchmark scores are retained below as reference results.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Train Baseline](#22-train-baseline)
   - [2.3 Use Course Data](#23-use-course-data)
   - [2.4 Make Submission](#24-make-submission)
   - [2.5 Saved Files and Detection Report](#25-saved-files-and-detection-report)
3. [TA Reference Results](#3-ta-reference-results)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 106-107 M) and at most 40 M
  trainable (heads included: LoRA 21.0 M; full fine-tuning and from scratch, 106 M, are over); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs the TA's ViT + LoRA reference, averaged over 3 seeds:
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound in the TA's test results).
- The current trainer uses fixed validation reference scores: **macro-F1 93.31%, mAP 26.13%**. Your local holdout
  differs from the TA's labeled validation split, so local Δm is a comparison aid rather than an official score.
  A new ViT + LoRA run is not forced to score Δm = 0.

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Train Baseline

After fetching and extracting the [course data](#23-use-course-data):

```bash
python train.py --data data/course-data --backbone vit --init pretrained --adapt lora \
  --seed 0 --eval-every 1

# All eight reference configurations, including the over-budget full/scratch runs
bash run_all.sh --data data/course-data
```

- Options: `--backbone vit|terramind`, `--init pretrained|scratch`, `--adapt full|lora|moelora`
  (`python train.py -h`).
- Each validation line reports Δm against the validation ViT + LoRA reference (macro-F1 93.31%, mAP 26.13%).
- Default training: 24 epochs, batch size 16, learning rate 1e-4, seed 0. `--eval-every 1` validates every epoch;
  the default interval is 4 epochs. The saved checkpoint is the final-epoch model.
- `--adapt full` and `--init scratch` train 106 M parameters: over the 40 M budget, for reference only.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Optional `--unfreeze-last N` trains the last N original backbone blocks alongside pretrained LoRA/MoE-LoRA;
  `--unfrozen-lr` sets their learning rate. See [partial fine-tuning examples](docs/WORKFLOW.md#partial-fine-tuning-and-short-screening-runs).
- Training time depends on GPU, adaptation, and validation frequency. Hardware-specific measurements are in
  [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.3 Use Course Data

Fetch the course archives already stored under `data/` with Git LFS, then extract them into `data/course-data/`.

```bash
git lfs install
git lfs pull --include="data/*.tar.gz"
mkdir -p data/course-data
tar -xzf data/train.tar.gz -C data/course-data
tar -xzf data/val.tar.gz -C data/course-data
```

`--data` points to the folder containing `train/labels.csv`, `train/instances.json`, and the referenced images.
Pass it explicitly; the script's legacy default points to a different dataset location.

The trainer holds out a seeded random 10% of the labeled training set. For the 9,392 course training images,
that means **8,453 training images and 939 local validation images**. The public course validation images are
unlabeled and used for submission prediction. `train.py` does not evaluate the official validation or test split.

Select experiments with CLI options rather than editing the training loop. Full data layout and command examples
are in [docs/WORKFLOW.md](docs/WORKFLOW.md).

### 2.4 Make Submission

`make_submission.py` loads a saved model, predicts the course validation images, maps boxes back to original image
pixels, and writes a ZIP containing `classification.csv` and `detection.json`:

```bash
python make_submission.py --run checkpoints/terramind_pretrained_moelora/seed_0 \
  --data data/course-data --split val --out submissions/my-model.zip
```

`--run` must contain `model.pt` and `metrics.json`. Choose an unused ZIP filename. Generating a ZIP does not upload
it to Codabench or consume a submission slot. `--split test` works when the course test images and `images.json`
are available locally.

### 2.5 Saved Files and Detection Report

After a successful run, `train.py` saves the following. Here, `<config>` means `<backbone>_<init>_<adapt>`.
Partial fine-tuning adds `_unfreezeN` to that configuration name.

| Location | Contents / purpose |
|---|---|
| `runs/<config>/` | Training log, final model weights, local validation history, predictions, and holdout annotations. |
| `results/<config>/seed_<seed>/` | Copies of `metrics.json`, `predictions_val.json`, and `instances_val_holdout.json` for saved results and reproducible reports. |
| `checkpoints/<config>/seed_<seed>/` | Copies of `model.pt` and `metrics.json` for running the trained model again; weights use Git LFS. |
| `reports/detection-review/` | Generated HTML and annotated images, recreated from saved predictions and training images. |

`--out`, `--results`, and `--checkpoints` change the respective parent directories. Reusing the same run path
overwrites its outputs. For another seed or hyperparameter experiment, choose distinct output parents.
`model.pt` stores learned weights, without optimizer or scheduler state for exact training resumption.

To compare saved detections:

```bash
python inspect_detections.py \
  --runs results/terramind_pretrained_moelora results/vit_pretrained_moelora \
  --data data/course-data --out reports/detection-review
```

Use result directories containing all three JSON files. Some saved examples place them directly under the
configuration directory; new runs use the seed subdirectory. The report shows a ground-truth-only panel, model
predictions, class AP scores, and error filters. Diagnostic matches use confidence ≥ 0.3 and IoU ≥ 0.5; these
counts are separate from COCO mAP.

Open `reports/detection-review/index.html` locally, or follow the
[Mac / SSH viewing instructions](docs/WORKFLOW.md#view-the-report). Generated reports, run files, and extracted
data are ignored by Git. Saved result JSONs and Git LFS checkpoints can be committed. Details:
[results/README.md](results/README.md), [checkpoints/README.md](checkpoints/README.md).

## 3. TA Reference Results

These are the TA's original benchmark measurements using labeled validation and test splits, averaged over three
seeds. They provide historical comparison values. The current course-data trainer uses a different local holdout
and does not reproduce these evaluation splits. Current experiment scores are stored in `results/`.

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
train.py              entry: train + evaluate one configuration
train.py.old          archived previous version of train.py; not used by training
run_all.sh            the eight reference runs (calls train.py)
make_submission.py    write_submission(): predictions -> submission.zip
inspect_detections.py saved predictions + correct boxes -> HTML detection report
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
├── metrics.py        accuracy, macro-F1, COCO box AP
└── optim.py          learning-rate groups for adapters, heads, and unfrozen backbone weights
tests/                CPU partial-fine-tuning checks and original tests; dataset-dependent tests need adaptation
docs/DETAILS.md       model, training protocol, design notes, time and memory
results/              saved metrics, holdout predictions, and correct annotations
checkpoints/          saved model.pt weights (Git LFS) and model configuration
reports/detection-review/  generated HTML and image overlays (ignored by Git)
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
