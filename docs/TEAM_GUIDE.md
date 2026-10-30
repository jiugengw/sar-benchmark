# Team guide: set up, rebuild and extend our best model

**Best result so far: Δm +65.92** (mAP 60.04, macro-F1 95.24),
[`submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/`](../submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/).

- **What the model is and why:** [BEST_MODEL_RECIPE.md](BEST_MODEL_RECIPE.md) (the stage-by-stage chain, options,
  times and scores) and [HISTORY.md](HISTORY.md) (every experiment and its lesson).
- **Job scripts:** [`sbatch/finetune_detector.sbatch`](../sbatch/finetune_detector.sbatch) runs any one stage;
  [`sbatch/staging/`](../sbatch/staging/) has every job and driver script used from 21 to 29 Oct.
- **Trained weights are not in the repository** (`runs/` is git-ignored). Ask Jiugeng for the checkpoints listed at
  the end of the recipe if you want to continue from them instead of retraining.

## 1. One-time setup on the SMU cluster (about an hour)

Run on the login node. The job scripts expect the repository at `~/scratchDirectory/sar-benchmark`.

```bash
mkdir -p ~/scratchDirectory && cd ~/scratchDirectory
git clone git@github.com:jiugengw/sar-benchmark.git && cd sar-benchmark
git lfs install && git lfs pull --include="data/*.tar.gz"
mkdir -p data/course-data && tar -xzf data/train.tar.gz -C data/course-data && tar -xzf data/val.tar.gz -C data/course-data
source /opt/apps/lmod/lmod/init/bash && module load Python/3.12.8-GCCcore-13.3.0
python -m venv .venv && source .venv/bin/activate && pip install -r models/yolo/requirements.txt
python models/yolo/prepare_dataset.py            # YOLO-format copy at 512 px with the fixed 8,453 / 939 split
mkdir -p runs/tm-48baseline-326894/terramind_pretrained_moelora
cp results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/split.json runs/tm-48baseline-326894/terramind_pretrained_moelora/
mkdir -p results/memorise-lr2-332326 && cp results/memorise-332085/box_bias_crossfit.json results/memorise-lr2-332326/
mkdir -p pretrained && (cd pretrained && python -c "from ultralytics import YOLO; YOLO('yolo26m-obb.pt')")
```

- The two `cp` lines put files where the scripts look for them: the classifier heads read the holdout split from
  `runs/`, and packaging reads a box-correction audit. Both audits rejected the correction, so packaging applies none
  either way.
- The cluster allows 100,000 files per user. The data, the YOLO copy and `.venv` together use about 50,000.
- Limits per user: 2 running and 4 submitted jobs, 1 GPU, 4 CPUs and 32 GB per job, 24 h per job.

## 2. Rebuild the best model (about 4-5 GPU-days)

Run every command from `~/scratchDirectory/sar-benchmark`. Each `sbatch` prints a job id; that stage's weights land
in `runs/ft40-<name>-<job>/train/weights/` (stage 1: `runs/yolo26m-p2-dota-s1-<job>/train/weights/`). Start the next
stage when the previous one has finished (`squeue -u $USER` shows nothing for it and its `last.pt` exists).

```bash
G='--gres=gpu:1 --constraint=l40s|a40|rtx_pro_6000_blackwell_max --prefer=l40s'   # any 48 GB-class GPU, L40S first
CLS=submissions/chain-dota-330938/image_classifier.pt   # only checked to exist: all-data stages skip evaluation
FLIPS='--dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine'
```

| Stage | Command (replace `<...>` with the run folders the previous stages made) |
|---|---|
| 1. Base, 100 epochs | `sbatch $G sbatch/staging/s1_stage1.sbatch` |
| 2. Plain | `sbatch $G --time=12:00:00 sbatch/finetune_detector.sbatch s1ad-plain --source <stage 1>/train/weights/best.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.002 --stem-stride 1 --batch 16` |
| 3. Speckle | `sbatch $G --time=12:00:00 sbatch/finetune_detector.sbatch s1ad-speckle --source <stage 2>/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.001 --speckle-prob 0.5 --stem-stride 1 --batch 16` |
| 4. Third | `sbatch $G --time=18:00:00 sbatch/finetune_detector.sbatch s1ad-third --source <stage 3>/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.001 --stem-stride 1 --batch 16 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512` |
| 5. Flip soup: six memorise runs | see [The soup, step by step](#the-soup-step-by-step) below |
| 6. Average them into one model | see [The soup, step by step](#the-soup-step-by-step) below |
| 7. Refit 1, 10 epochs | `sbatch $G --time=08:00:00 sbatch/finetune_detector.sbatch s1adfsr1 --source runs/soup-s1fs/alldata-s1fs-six.pt --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 $FLIPS` |
| 8. Refit 2, 10 more | the same with `s1adfsr2 --source <stage 7>/train/weights/last.pt` |
| 9. Package (quick classifier, rounding, hedging, low-score tail) | `sbatch $G --export=ALL,TAG=alldata-s1fsr2,DET=<stage 8>/train/weights/last.pt,RUN=<stage 8> sbatch/staging/fliprefit_package.sbatch`; its log (`slurm-<job>.out`) prints `S1FSR_ZIP_READY <zip>` and `S1FSR_TAIL_READY <zip>` |
| 10. Two-head classifier | `sbatch $G --export=ALL,DET=<stage 8>/train/weights/last.pt,Z=<the S1FSR_ZIP_READY zip> sbatch/staging/fliprefit_heads.sbatch`; it prints `S1FSR_TWOHEAD_ZIP <zip>` |
| 11. Tail on the final ZIP | `python models/yolo/lowconf_tail.py <two-head zip> <same folder>/alldata-s1fsr2-lowconf.zip <two-head zip without .zip>-tail.zip` |

The ZIP from step 11 is the one to upload. Run times: stage 1 about 30 h (resume it once past 24 h, see the script's
header), stages 2-4 12-18 h each, each soup run about 10 h, each refit about 5 h, packaging 1.5 h, the heads about
4 h. In all about 140 GPU-hours: roughly 4 days of wall-clock with six GPUs for the soup, 5 days with two.

### The soup, step by step

**What it is.** Six copies of the same model are trained from the same stage-4 checkpoint, each with a slightly
different setting (the assigner's `--tal-beta`: 8, 12, 10, 7, 9, 11). Because they start from one point, their weights
stay close, and the plain average of the six weights is one network that is better than each of them. It is still
**one model** (one set of weights, one forward pass), so it is allowed; it is not an ensemble, which would run several
models and merge their outputs.

**What it buys.** Measured on Codabench (stride 1, all data): single model mAP 55.17, soup of six 55.93, so about
+0.75 mAP (+1.4 Δm). It costs about 60 GPU-hours, the biggest single item in the chain. Fewer runs (three) would
probably keep most of the gain, but that has not been tested.

**Each run** is stage 4 plus 20 epochs of memorise (no augmentation) with the images of the six flip classes shown in
all four orientations, so one epoch takes about 32 minutes and a run about 10 hours. Every run must start from the
**same** stage-4 `last.pt`.

**1. Submit the six runs.** You can have 4 jobs submitted and 2 running at a time, so submit four, then the last two
when two have finished:

```bash
T=<stage 4 folder>/train/weights/last.pt
for B in 8 12 10 7; do
  sbatch $G --time=12:00:00 sbatch/finetune_detector.sbatch s1adfs-b$B --source $T --classifier $CLS --all-data \
      --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 16 --tal-beta $B $FLIPS
done
# later, when squeue shows fewer than 4 of your jobs: the same command with B in 9 11
```

**2. Check all six finished.** Each `runs/ft40-s1adfs-b<B>-<job>/` needs `train/weights/last.pt` and `compute.json`
(written at the end of training):

```bash
ls runs/ft40-s1adfs-b*-*/compute.json | wc -l      # must print 6
```

**3. Average them** (a minute on the login node). `swa_study.average` averages every weight and batch-norm statistic
and saves one checkpoint:

```bash
mkdir -p runs/soup-s1fs
python -c "import sys; sys.path.insert(0, 'models/yolo'); from pathlib import Path; from swa_study import average; average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1fs/alldata-s1fs-six.pt')); print('averaged', len(sys.argv) - 1)" runs/ft40-s1adfs-b*-*/train/weights/last.pt
```

It should print `averaged 6`. `runs/soup-s1fs/alldata-s1fs-six.pt` is the input of stage 7 (refit 1).

**What did not work:** averaging checkpoints of one run, or a run with its own continuation (no gain), and mixing
different recipes in one soup (flip runs averaged with strong-speckle runs lost the flip runs' gain). Average siblings
of one parent trained with the same recipe, changing only a small setting such as beta.

## 3. Rules to keep

- One model: no ensembles, no test-time augmentation. Averaging the weights of runs from one start point (a soup) is
  one model.
- At most 130 M parameters in total and 40 M trainable; input at most 512 x 512, each image once.
- Pretraining only on optical or generic data (COCO, DOTA, ImageNet; TerraMind is allowed). No other SAR datasets and
  no models trained on them. Never train on val or test images.

## 4. Working habits

- Commit new job scripts to `sbatch/` and record every upload in `submissions/` (see
  [submissions/README.md](../submissions/README.md)); score changes on the holdout first (`group_runs.py` style,
  official-mix plus the twin / flipped / new-scene groups) and decide with a rule set before the run.
- Every job file should carry `#SBATCH --prefer=l40s` next to its GPU constraint.
