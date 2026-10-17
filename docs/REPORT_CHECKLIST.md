Back to the [README](../README.md).

# Report checklist and compute records

Parameter counts, configurations and results of every attempt are summarised in [docs/HISTORY.md](HISTORY.md).
Learned task weights and component losses are recorded in each run's `training_history.json`.

Use our saved experiment results for the report. The TA's reference scores are their measurements, not
repeated runs by our team.

| Required item | Evidence / remaining work |
|---|---|
| Total and trainable parameters, including heads | Each completed run's `metrics.json`: `parameters`, `trainable_parameters`. |
| GPU type/count and final training time | New `compute.json` records GPU model, model GPU count, timestamps and process hours. `metrics.json` also retains training-loop hours including validation. Existing hardware information comes from job scripts and experiment notes. |
| Whole-project GPU hours, tuning included | Export Slurm allocation accounting with the command below. Include failed/cancelled jobs and add runs from other machines. Do not sum copied metrics or overlapping accounting sources. Historical whole-project total is still pending. |
| Peak GPU memory | Existing metrics store peak allocated GiB. New compute records also save peak reserved GiB. |
| Inference time per image | New submission generation writes `<name>.inference.json`: GPU, batch size, precision, prediction wall time/image and synchronized model time/image. Model timing excludes the first batch; it is batch-amortized throughput, not batch-1 latency. The final model still needs a recorded measurement. |
| Input and preprocessing | New records save grayscale 512×512 input, geometry, scale, normalization and training augmentation flags. Full implementation: `sarbench/data.py`, `models/terramind/make_submission.py`. |
| Every pretrained model, including training-only models | New records save model identifiers, source URLs and local weight paths. Verify checkpoint provenance and document any additional teacher or auxiliary model if introduced. |
| External code versus team contributions | `requirements.txt` pins dependencies; new records capture actual installed versions and Git revision/dirty status. Still write the attribution distinguishing TA/reference code, external libraries and team modifications. |
| Validation and Phase 2 selection | Saved configuration, holdout annotations and newer `split.json` record the split. Recent runs select highest local Δm and calibrate locally. Record the exact final checkpoint, ZIP and decision before Phase 2. |
| Seeds and uncertainty | Current saved experiments use seed 0. Repeat final comparisons where a gain below about 2 points is claimed; report mean ± sample std and number of runs. Keep holdout fixed: current 90/10 splitting uses `--seed`, so separate split/training seeds before these repeats. |
| Ablations | Assemble local comparisons from `results/`, keeping the split fixed. State all settings changed; longer schedules change both epoch count and LR decay. Planned changes are not trained results. |

Future `models/terramind/train.py` runs write `compute.json` at startup, every 30 seconds, after
training epochs, and on normal completion or Python exceptions. Completed records
are copied beside results/checkpoints. Reusing an output preserves the previous
compute record under a unique filename. Calibration has its own
`calibration_compute.json`; submission inference writes a separate record beside
its ZIP. These include software/model/input provenance. They do not automatically
produce missing experiments, seed repetitions or the written contribution analysis.

Process GPU hours estimate process wall hours × GPUs used by the model, including
CPU setup. Slurm allocation GPU hours include all time GPUs were allocated,
including setup, calibration and export. Use the latter for cluster project totals;
do not add process records on top. A hard kill can leave a heartbeat marked
`running`; Slurm accounting determines its actual final state and duration.

From the repository root, export the cluster accounting (choose the date covering
the project's first cluster run):

```bash
python tools/account_compute.py --start 2026-10-01
```

This saves `results/compute/slurm_accounting.json` and the raw accounting output.
It matches this working directory and job IDs found in `runs/*.out`; use
`--job-ids ID ...` for additional jobs. Refresh the export after jobs finish.
Snapshots replace previous totals; do not add snapshots together. Review missing
GPU counts and add usage from other hosts. Older jobs outside Slurm's retention
window may require saved logs. Already-running training processes do not acquire
the new logging code; their Slurm allocations can still be counted.
