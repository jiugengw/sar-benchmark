# Cluster job scripts, 21-29 October (rebuilt)

The job and driver scripts that ran on the SMU cluster from `~/sbatch-staging/` (outside the repository) from 21 to
29 October: the stride-1 chain, the soups, the flip soup, the refits, packaging, the two-head classifier, the low-score
tail and the holdout tests. The order that produced the best model is in
[docs/BEST_MODEL_RECIPE.md](../../docs/BEST_MODEL_RECIPE.md).

**Exact copies** of `~/sbatch-staging` on the cluster, taken on 30 Oct (94 files). A few more were rebuilt from the run
logs because they no longer exist there (they were written to `/tmp` by drivers, or removed); `PROVENANCE.json` says
which is which. `s1_stage1.sbatch` is stage 1 of the best model, written out of `reset_s1.sh` as a standalone job, and
`flip_gap.py` + `flip_gap.sbatch` are the 30 Oct flipped-copy diagnosis.

They assume the layout on the cluster: the repository at `~/scratchDirectory/sar-benchmark`, these scripts in
`~/sbatch-staging/`, the `student` partition, and the Python 3.12.8 module with the repository's `.venv`. Many are
one-off experiments; see docs/HISTORY.md for what each tested and how it scored.

To rebuild or extend the best model, follow [docs/TEAM_GUIDE.md](../../docs/TEAM_GUIDE.md).
