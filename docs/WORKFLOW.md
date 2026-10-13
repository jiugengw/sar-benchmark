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

Use the CLI to select the configuration. For the supported experiment choices, there is no need to edit `train.py`:
choose `--backbone vit|terramind`, `--init pretrained|scratch`, and `--adapt full|lora|moelora`. Add code only when
you want to change training behavior or add a new method.

```bash
# One configuration, using the data downloaded above
python train.py --data data/course-data --backbone vit --init pretrained --adapt lora

# Change the model mode and keep this run's files under runs/
python train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
  --seed 1 --out runs/seed-1

# Short run for checking the setup
python train.py --data data/course-data --backbone vit --init pretrained --adapt lora \
  --limit 64 --epochs 1 --eval-every 1 --out runs/smoke \
  --results results/smoke --checkpoints checkpoints/smoke
```

The tiny holdout can omit classes and produce non-finite classification scores; use it only to check pipeline
execution. Full-score comparisons need the normal labeled holdout.

`--out` chooses a parent output folder. The run is written to a configuration-named directory inside it. Each run
contains `log.txt`, `metrics.json` (including validation history and Δm), `predictions_val.json`, and `model.pt`.
The default parent folder is `runs/`.

At the end of a successful run, `train.py` also copies `metrics.json`, `predictions_val.json`, and
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
python inspect_detections.py \
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
