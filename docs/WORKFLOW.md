# Data and experiment workflow

Back to the [README](../README.md).

## Get the course data

The training code expects `--data` to point to a directory whose `train/` folder contains `labels.csv`,
`instances.json`, and the referenced images. For example:

```
data/course-data/
└── train/
    ├── labels.csv
    ├── instances.json
    └── ... image files ...
```

Local validation holds out a seeded random 10% of the labeled training rows: 939 holdout images and 8,453 training
images for this dataset. The public course `val/` images are a separate, unlabeled submission split. The trainer
does not evaluate official validation or test scores locally. Always pass `--data`; its legacy default points
outside the extracted `data/course-data/` folder.

### Fetch and extract the repository archives

The repository already contains the archive entries `data/train.tar.gz` and `data/val.tar.gz`; Git LFS stores their
large file contents. From the repository root, fetch those contents and unpack them under `data/course-data/`:

```bash
git lfs install
git lfs pull --include="data/*.tar.gz"
mkdir -p data/course-data
tar -xzf data/train.tar.gz -C data/course-data
tar -xzf data/val.tar.gz -C data/course-data
```

If Git LFS has already downloaded the archive contents, skip the first two commands and extract them. The folder can
live elsewhere; pass its path with `--data /path/to/course-data`.

## Run experiments

Use the CLI to select the configuration. For the supported experiment choices, there is no need to edit `models/terramind/train.py`:
choose `--backbone vit|terramind`, `--init pretrained|scratch`, and `--adapt full|lora|moelora`. Add code only when
you want to change training behavior or add a new method.

```bash
# One configuration, using the data downloaded above
python models/terramind/train.py --data data/course-data --backbone vit --init pretrained --adapt lora

# Change the model mode and keep this run's files under runs/
python models/terramind/train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
  --seed 1 --out runs/seed-1

# Short run for checking the setup
python models/terramind/train.py --data data/course-data --backbone vit --init pretrained --adapt lora \
  --limit 64 --epochs 1 --eval-every 1 --out runs/smoke \
  --results results/smoke --checkpoints checkpoints/smoke
```

The tiny holdout can omit classes and produce non-finite classification scores; use it only to check pipeline
execution. Full-score comparisons need the normal labeled holdout.

### Partial fine-tuning and short screening runs

Frozen backbone weights stay fixed; LoRA/MoE-LoRA adapters and task heads still train. `--unfreeze-last N` also
allows the original weights in the last N encoder blocks to train. It is supported with pretrained LoRA or
MoE-LoRA. Those original weights use `--unfrozen-lr` (default 2e-5), while adapters use `--backbone-lr` and heads
use `--lr`. The trainer checks the total/trainable parameter budget for this mode.

For example, TerraMind + MoE-LoRA with two unfrozen blocks has 35.30M trainable parameters. A 12-epoch screen:

```bash
python models/terramind/train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
  --unfreeze-last 2 --unfrozen-lr 2e-5 --epochs 12 --seed 0 --eval-every 1 \
  --out runs/partial-screen --results results/partial-screen --checkpoints checkpoints/partial-screen
```

The configuration directory gets an `_unfreeze2` suffix. Short runs help screen ideas, but `--epochs 12` changes
the cosine schedule: it is not equivalent to epoch 12 of a 24-epoch run. Compare promising methods with a matched
epoch count and schedule before interpreting a difference as an improvement.

`--out` chooses a parent output folder. The run is written to a configuration-named directory inside it. Each run
contains `log.txt`, `metrics.json` (including validation history and Δm), `predictions_val.json`, and `model.pt`.
The default parent folder is `runs/`.

At the end of a successful run, `models/terramind/train.py` also copies `metrics.json`, `predictions_val.json`, and
`instances_val_holdout.json` into `results/<backbone>_<init>_<adapt>/seed_<seed>/`. Different seeds have separate
result folders. Use `--results results/my-experiment` to choose a different parent folder for a learning-rate or
epoch-count variant with the same configuration and seed; reusing the same result path replaces its JSON files.

`model.pt` contains the learned model weights. It is saved in the run folder and also copied with `metrics.json`
to `checkpoints/<backbone>_<init>_<adapt>/seed_<seed>/`. Use `--checkpoints` to choose another checkpoint parent.
The weights are required for new predictions and submission generation. The JSON results are sufficient to
regenerate the detection report when the training images are also available.

## Inspect detection errors

Compare saved predictions on the same local holdout without another GPU inference pass. Each result directory
must contain `metrics.json`, `predictions_val.json`, and `instances_val_holdout.json`:

```bash
python tools/inspect_detections.py \
  --runs results/terramind_pretrained_moelora results/vit_pretrained_moelora \
  --data data/course-data --out reports/detection-review
```

Open `reports/detection-review/index.html` in a browser. It shows the correct boxes in a separate ground-truth panel,
model predictions on the same images, class AP scores, and filters for missed objects, incorrect classes,
duplicates, and inaccurate boxes. Green/orange boxes in the model panels are ground truth; blue/yellow/purple/red
boxes are predictions, as explained in the legend. Click an image to view it larger.

The diagnostic matcher uses confidence at least 0.3 and IoU at least 0.5; change these with `--score-threshold` and
`--iou`. Its counts are separate from COCO mAP. `summary.json` contains counts for all holdout images; the gallery
shows selected examples with errors and coverage across classes. Add more completed result directories to `--runs`
to compare more models. File descriptions are in [results/README.md](../results/README.md).

### View the report

If the files are on your own computer, open `reports/detection-review/index.html` directly in a browser.

If they are on a GPU machine you access over SSH, start a server from the repository folder on that machine:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory reports/detection-review
```

In a separate terminal on your Mac, forward that port using the same `user@host` you use to connect to the GPU:

```bash
ssh -N -L 8000:127.0.0.1:8000 YOUR_GPU_HOST
```

Replace `YOUR_GPU_HOST` with your SSH destination, then open **http://localhost:8000** on the Mac. Leave both
terminals running while viewing the report. The generated HTML and overlay images are ignored by Git; regenerate
them from the saved JSON files whenever needed.

## Save files with Git

`runs/`, the extracted `data/course-data/` folder, and generated `reports/detection-review/` files are ignored by
Git. Training automatically preserves the three small JSON files under `results/`, where they are visible to Git.

The checkpoint copies under `checkpoints/` and the two source archives under `data/` use Git LFS, so commits store
pointers while LFS stores the large files. Detailed logs remain in the ignored run directory. Files in `results/`
and `checkpoints/` become part of repository history after adding and committing them:

```bash
git status --short
git add results/ checkpoints/ .gitattributes
git commit -m "Save experiment results"
```

## Five-fold offline evaluation

Prepare deterministic, stratified folds from `train/labels.csv`:

```bash
bash models/terramind/run_anchor_cv5.sh
```

Inspect `runs/anchor48-cv5/fold_plan.json` for row assignments, class counts,
and exact training commands. Each image appears in validation exactly once;
all boxes belonging to that image stay together. The course validation images
are not used. The seed for partitions is independent of the training seed.

Run all five folds sequentially on one CUDA GPU:

```bash
bash models/terramind/run_anchor_cv5.sh --run
```

Defaults use 48-epoch TerraMind MoE-LoRA with normal anchors,
photometric augmentation and box loss weight 2. Override `--anchor-profile`
to compare another supported anchor profile on identical folds. Each fold starts a fresh
model from pretrained weights. This is five training runs, with roughly 80/20
train/validation splits. Completed folds are skipped on restart; an interrupted
fold starts again. Use a different `--out` when changing settings.

Training logs are `runs/anchor48-cv5/fold_<n>.console.log`. Each fold saves its
own model, split manifest, predictions, ground truth, and metrics. After all
folds complete, `cv_summary.json` reports individual scores and mean/sample
standard deviation for classification F1, detection mAP, and other metrics.
To regenerate the summary, repeat the same arguments with `--summarize-only`.

Final-epoch scoring is the default. Enabling `--select-best` uses the reported
validation fold to select the checkpoint, so its scores are selection-biased;
use a separate inner holdout for checkpoint or calibration tuning when an
unbiased outer-fold estimate is required. Local Delta m still uses the fixed
reference scores, not a baseline retrained on matching folds. Compare candidate
models on identical folds. Five models here provide an evaluation estimate;
submission inference still uses a single model.
