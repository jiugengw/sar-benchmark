# Training run comparison

Open [index.html](index.html) to compare the selected runs. The default selection is the 100-epoch E5 YOLO26m-P2 baseline and its 15-epoch joint fine-tune. The top section displays official scores separately from local holdout scores, followed by an interactive macro-F1/mAP calculator for a target Δm of +30. The local metric table includes per-class AP. The visual review shows the same labeled images side by side, with filters for class and error type.

The official results use 1,426 course validation images. The local gallery and diagnostics use the separate 939-image labeled training holdout. Official percentage values are normalized when building the report; they must not be combined with local metrics to produce a score.

Shareable artifacts: [official-score comparison](phase3-official-comparison.png), [same-image detection contact sheet](phase3-detection-comparison.jpg), and [machine-readable score audit](score-audit.json).

Archived training curves, raw per-epoch CSV logs, run arguments, and separate classifier histories are indexed in [curves/README.md](curves/README.md). This includes E5 YOLO26m-P2 and the other logged YOLO / RT-DETR runs.

Older completed runs with `metrics.json`, `predictions_val.json`, and `instances_val_holdout.json` under `results/` remain selectable and are discovered automatically. All runs must use the same holdout for the visual comparison. The gallery shows up to 48 error examples at confidence 0.30 and IoU 0.50. The standalone contact sheet instead selects median-object-count examples within six classes, without selecting for which model wins. Both YOLO26m submission ZIPs are linked in the report.

To refresh the snapshot, run:

    python reports/training-comparison/build_comparison.py
    python reports/training-comparison/render_phase3_summary.py

To add a running job, add an entry to runs.json with its `results_root`, `log_path`, and `expected_epochs`. Completed result folders are picked up automatically. Set `compare_by_default` to true for runs that should be selected when the report opens. Rebuild the HTML after updating the manifest.

Use `official_scores_path` to associate a run with its saved official scoring JSON. Use `training_history_path` and `checkpoint_metrics_path` to trace differences between training validation, exported inference, and official scoring. Official result files that are empty or invalid are omitted from the scoreboard.

## Phase 3 score gap

The best training-epoch record was macro-F1 93.7171%, mAP 40.9880%. Running the **exact submission generator** on the same labeled holdout gives macro-F1 93.7171%, mAP 40.6013%, AP50 67.1401%. Its classifier confusion matrix matches the training record exactly. With predicted boxes held fixed, changing between Ultralytics and COCO AP calculators changes mAP by only about 0.03 percentage points; the larger 0.39-point training-to-export difference therefore involves the wider inference/validation path, not just the AP calculator. Its precise source has not been isolated.

The official result is macro-F1 92.5600%, mAP 38.7776%, Δm +23.7968: gaps of 1.1571 F1 points and 1.8237 mAP points from the exact local submission pipeline. The original E5 also has local-to-official gaps (1.3313 F1 points, 1.4315 mAP points). Fine-tuning improves official F1 by 2.8127 points but reduces official mAP by 0.4633 points, for a net Δm gain of 0.6207. These measurements do not prove overfitting; official per-class labels/errors are unavailable, so split composition and difficulty cannot be separated from model generalization effects.

To reproduce the exact submission-path diagnostic, use the CUDA-enabled environment and run:

```bash
PYTHONPATH=. .venv/bin/python reports/training-comparison/audit_submission_pipeline.py \
  --run runs/phase3-yolo26m-joint-20261017-1516 \
  --ground-truth results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json \
  --official-scores submissions/phase3_yolo26m_joint_20261017_1516/scoring_result/scores.json \
  --output results/phase3-yolo26m-joint-20261017-1516/submission-pipeline
```

This builds a temporary view containing only the labeled holdout, passes it through `make_submission.py` with batch size 16, evaluates its ZIP predictions, and saves the predictions and diagnostic metrics. It performs no training or leaderboard submission. The metrics also include `clean_holdout` and `clean_minus_official_percentage_points`: the same scores on the 720 holdout images with no near-duplicate twin in training (see `sarbench/twins.py`). Graded this way, Phase 3 is 0.41 mAP points below official instead of 1.82 above.
