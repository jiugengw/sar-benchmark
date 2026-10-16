# Training run comparison

Open index.html to compare the selected runs. Use the search and status filters to narrow the run list, then select the runs to compare. The metric table includes per-class AP. The visual review shows the same labeled images side by side for every selected run with saved predictions, with filters for class and error type. Completed runs use selected-checkpoint validation metrics; running runs show the latest validation recorded in their Slurm log.

The manifest lists the 326894 baseline and the targeted 327063/327100 jobs. Other completed runs with `metrics.json`, `predictions_val.json`, and `instances_val_holdout.json` under `results/` are discovered automatically. All discovered runs must use the same holdout for the visual comparison. The gallery shows up to 48 representative error examples, selected from the shared labeled holdout at confidence 0.30 and IoU 0.50. The 326894 calibrated submission ZIP is linked from the report.

To refresh the snapshot, run:

    python reports/training-comparison/build_comparison.py

To add a running job, add an entry to runs.json with its `results_root`, `log_path`, and `expected_epochs`. Completed result folders are picked up automatically. Set `compare_by_default` to true for runs that should be selected when the report opens. Rebuild the HTML after updating the manifest.
