# Cluster job scripts, 21-29 October (rebuilt)

The job and driver scripts that ran on the SMU cluster from `~/sbatch-staging/` (outside the repository) from 21 to
29 October: the stride-1 chain, the soups, the flip soup, the refits, packaging, the two-head classifier, the low-score
tail and the holdout tests. The order that produced the best model is in
[docs/BEST_MODEL_RECIPE.md](../../docs/BEST_MODEL_RECIPE.md).

**Rebuilt on 29 Oct from the Claude session logs** while the cluster refused logins, so not byte-for-byte:
- `fliprefit_package.sbatch`, `fliprefit_heads.sbatch`, `refit_ship.sh`, `fliprefit3.sbatch` are exact copies printed
  from the cluster that day.
- Every other file is its last full version from the logs. `PROVENANCE.json` says how each was written and lists the
  later one-line `sed` edits made on the cluster that are not applied here (mostly memory and GPU lines).
- Some job files were written by drivers at run time (e.g. `s1_stage1.sbatch` inside `reset_s1.sh`,
  `s1soup_package.sbatch` inside `alldata_s1_soup.sh`); their text is inside those drivers.
- Eight files were written through an unquoted heredoc (`convhead`, `polviews`, `polviews_full`, `rawcand`, `soup`,
  `soup_cls`, `s1_memorise_submit`, `scmosaic_submit`): the shell filled in variables such as job ids as it wrote them;
  here they keep the variable names.
- `flip_gap.py` + `flip_gap.sbatch` (29 Oct, not run yet): why flipped copies score ~91 against ~98 for the same
  scenes as-is.

They assume the layout on the cluster: the repository at `~/scratchDirectory/sar-benchmark`, these scripts in
`~/sbatch-staging/`, the `student` partition, and the Python 3.12.8 module with the repository's `.venv`. Many are
one-off experiments; see docs/HISTORY.md for what each tested and how it scored.
