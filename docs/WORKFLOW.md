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

The validation set is made by holding out 10% of the labeled training rows. The loader does not need labels for the
separate course validation or test splits.

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
  --limit 64 --epochs 1 --eval-every 1 --out runs/smoke
```

`--out` chooses a parent output folder. The run is written to a configuration-named directory inside it. Each run
contains `log.txt`, `metrics.json` (including validation history and Δm), `predictions_val.json`, and `model.pt`.
The default parent folder is `runs/`.

## Keep Git focused on code

`runs/` and the extracted `data/course-data/` folder are ignored by Git. Keep generated checkpoints and predictions
in `runs/`. The saved ViT + LoRA metrics file is in `results/vit_pretrained_lora/metrics.json`, outside the ignored
folder. Use `--out results/experiment-1` when you want a new run's files visible to Git; otherwise the default keeps
them under ignored `runs/`. Git only tracks files after you add them to a commit. The two source archives under
`data/` remain tracked with Git LFS, so commits store pointers while LFS stores the large files. Commit code and
documentation changes normally:

```bash
git status --short
git add .gitignore train.py README.md docs/WORKFLOW.md
git commit -m "Report validation delta m and document workflow"
```
