# How the best model was built (official Δm +65.92)

The submission `submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/` (mAP 60.04, macro-F1 95.24) came from the chain
below. Every stage is one `models/yolo/finetune_detector.py` (or `train_yolo26s_p2.py`) run started from the previous
stage's weights; nothing else is needed but the course data and Ultralytics' public DOTA weights. The job scripts that
ran it are in [`sbatch/staging/`](../sbatch/staging/) (rebuilt from logs on 29 Oct, see its README). Weights are not in
the repository (`runs/` is git-ignored).

Common to every stage: YOLO26m-P2 with the first convolution at stride 1 (`--stem-stride 1`), 512 px input, batch 16,
MuSGD, one GPU (L40S or RTX PRO 6000 Blackwell). Times are the ones the runs recorded; about 85 GPU-hours in all. `--all-data` trains on the training split and the holdout
together and keeps the last epoch. CLS below is `runs/memorise-lr2-332326-package/image_classifier.pt` (only used for the
skipped evaluation).

## The chain

| # | Stage | Starts from | Options (besides `--stem-stride 1 --classifier $CLS --skip-evaluation`) | Time |
|---|---|---|---|---|
| 1 | Base, 100 epochs (training split) | `pretrained/yolo26m-obb.pt` (Ultralytics, DOTA) | `train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1` | 12.7 h |
| 2 | Plain, 40 epochs | 1, best.pt | `--all-data --lr0 0.002` | 2.8 h |
| 3 | Speckle, 40 epochs | 2, last.pt | `--all-data --lr0 0.001 --speckle-prob 0.5` | 2.8 h |
| 4 | Third, 40 epochs | 3, last.pt | `--all-data --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512` | 5.5 h |
| 5 | Flip soup: 6 memorise runs, 20 epochs each over the 4x flip list | 4, last.pt | `--all-data --no-augment --lr0 0.002 --epochs 20 --batch 16 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine` with `--tal-beta` 8, 12, 10, 7, 9, 11 (one run each) | 5-11 h each (45 h in all) |
| 6 | Average the six | 5 | `swa_study.average([...last.pt]) -> runs/soup-s1fs/alldata-s1fs-six.pt` (one model, not an ensemble) | minutes |
| 7 | Refit 1, 9 epochs (stopped at 9 of 10) | 6 | `--all-data --no-augment --lr0 0.001 --epochs 10 --batch 16 --tal-beta 8 --dataset-flips --dataset-flips-classes <same six>` (job 340515) | ~5 h |
| 8 | **Refit 2**, 10 more epochs | 7, last.pt | same as 7 (job 340754): `runs/ft40-s1adfsr2-340754/train/weights/last.pt` | ~5 h |
| 9 | Package: quick classifier, whole-pixel rounding, two-tier hedging, low-score tail | 8 | `sbatch/staging/fliprefit_package.sbatch` (`package_candidate.py --classifier-all-data`, `round_boxes.py zip`, `hedge_study.py zip`, `make_submission.py --conf 1e-5 --max-det 1000` + `lowconf_tail.py`) | ~1.5 h |
| 10 | Two-head image classifier, then the tail again | 8 + the ZIP of 9 | `sbatch/staging/fliprefit_heads.sbatch` (`branch_head_study.py --epochs 20 --all-data`, `conv_head_study.py --epochs 30 --all-data`, both predictors, 0.5/0.5 average + size rule), then `lowconf_tail.py` | ~4 h |

Stages 9 and 10 run unattended with `sbatch/staging/refit_ship.sh <refit job> <N>`.

## Official scores along the way

| After | Official mAP | Official F1 | Official Δm |
|---|---:|---:|---:|
| 4 + memorise (single all-data stride-1 model) | 55.17 | 94.45 | +56.18 |
| Soup of six plain memorise runs + two-head classifier | 55.93 | 95.13 | +58.00 |
| 6, the flip soup, + two-head classifier | 59.80 | 95.40 | +65.55 |
| 8, refit 2, quick classifier | 59.93 | 92.34 | +64.16 |
| 8 + two-head classifier (10 without the tail) | 59.93 | 95.24 | +65.71 |
| **8 + two-head classifier + low-score tail** | **60.04** | **95.24** | **+65.92** |

## The training-split twin (for holdout tests)

Holdout tests use the same chain on the training split only (`sbatch/staging/reset_s1.sh`: `s1r-plain`, `s1r-speckle`,
`s1r-third`, then memorise), the flip memorise test 338727 (all classes flipped, 20 epochs) and its 10-epoch refit 340514
(`runs/ft40-s1r-fliprefit-340514/train/weights/best.pt`, all classes flipped, lr 0.001, beta 8). Scores by holdout group
come from `group_runs.py`; official-mix is the clean holdout weighted like the test set.

## Checkpoints worth keeping (all on the cluster under `~/scratchDirectory/sar-benchmark/runs/`)

- `ft40-s1adfsr2-340754/train/weights/last.pt`: refit 2, the detector behind the best ZIP
- `soup-s1fs/fliprefit-heads-341248/{branch_head,conv_head}.pt`: its two classifier heads
- `soup-s1fs/alldata-s1fs-six.pt`: the flip soup (start of refits)
- `ft40-s1ad-third-*/train/weights/last.pt`: the all-data third stage (start of new soups or reruns)
- `ft40-s1r-fliprefit-340514/train/weights/best.pt` and `results/ft40-s1r-fliprefit-340514/`: the holdout twin
