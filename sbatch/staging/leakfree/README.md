# Leak-free retrain (19 Nov 2026)

The whole final pipeline ([docs/BEST_MODEL_RECIPE.md](../../../docs/BEST_MODEL_RECIPE.md), stages 1-6 and 9-11, every
setting unchanged) and the TA's reference model, retrained on the leak-free split of
[docs/LEAKFREE_SPLIT.md](../../../docs/LEAKFREE_SPLIT.md) and scored on its new test (and val).

Copy this folder, `tools/leakfree/make_dataset.py` and `results/leakfree/split.csv` to `~/sbatch-staging/leakfree/`,
then on the login node, from the repository: `nohup bash ~/sbatch-staging/leakfree/chain.sh >/dev/null 2>&1 &`.

| script | what | GPU |
|---|---|---|
| `job_aj` | new-split data (`runs/leakfree/course-data.tar`), holdout split.json, smoke test | any |
| `job_ao` | TA reference (ViT-B/16 + LoRA), seeds 0 1 2, scored on new test / val | Blackwell |
| `job_ak` | stage 1, 100 epochs | L40S / A40 |
| `job_al` | stages 2-4 (plain, speckle, third), all data | L40S / A40 |
| `job_am` x6 | stage 5, flip-soup ingredients, TAL beta 8 12 10 7 9 11 | L40S / A40 |
| `job_an` | stage 6 soup, two-head classifier, `make_submission.py --final`, scores (packaged and plain) | L40S / A40 |

`env.sh` gives each job a private copy of the code in `/tmp` with the new-split data at `data/course-data`, so the
scripts' built-in paths read the new split; outputs go to `runs/leakfree/{runs,results}` on scratch.
