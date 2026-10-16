# Training run comparison

Open index.html to compare the selected runs. Use the search and status filters to narrow the run list, then select the runs to compare. Completed runs use selected-checkpoint validation metrics; running runs show the latest validation recorded in their Slurm log.

The report currently compares the 326894 48-epoch baseline with running variants 327063_0 and 327063_1. The queued 327100 run is listed without metrics until it starts and produces a validation result. The 326894 calibrated submission ZIP is linked from the report.

To refresh the snapshot, run:

    python reports/training-comparison/build_comparison.py

To add later experiments, add an entry to runs.json. Point completed runs at metrics_path, or set results_root to a directory containing metrics.json. For running jobs, set log_path and expected_epochs. Set compare_by_default to true for runs that should be selected when the report opens. Rebuild the HTML after updating the manifest.
