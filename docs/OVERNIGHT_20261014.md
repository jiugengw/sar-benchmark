# Overnight detection experiments, 14 October 2026

Active Slurm array: **324156**, submitted using `sbatch sbatch/overnight_improve.sbatch`.
Previous array 324152 was cancelled at the user's request; it produced no completed model.
The user then approved restarting on Blackwell. Each task uses one RTX PRO 6000 Blackwell Max GPU;
at most two tasks run concurrently. Account access is the student partition.

| Task | Experiment |
|---|---|
| 0 | TerraMind pretrained MoE-LoRA, 36 epochs, baseline box loss |
| 1 | Same, RoI box regression loss multiplied by 2 |
| 2 | Same as task 1, training-only brightness/contrast jitter of 0.15 |
| 3 | Locally evaluate classifier confidence calibration on existing baseline and box-loss checkpoints |

All new training runs use learning rate 1e-4, batch size 16, seed 0, the existing 90/10 labeled-training
split, uniform sampling, default anchors, stretched 512-pixel input, and frozen original backbone weights.
Validation runs every two epochs. The checkpoint with highest measured local Delta m is retained;
metrics distinguish its selected epoch from the final epoch. Prior runs retained final-epoch checkpoints.
Checkpoint selection and calibration use the same local holdout, so the resulting local scores are
selection estimates, not independent evidence of generalization.

Calibration tests powers 0, 0.5, and 1 in
`detection_score * classification_probability[detection_category] ** power`.
Power 0 is unchanged scoring. Classes and boxes are unchanged. The winning local setting is used to
generate each submission ZIP. This is one model, one inference pass per image, no ensemble or TTA.

Outputs:

- Logs: `runs/overnight-improve-324156_<task>.out`
- Runs: `runs/overnight-improve-324156/`
- Metrics and holdout predictions: `results/overnight-improve-324156/`
- New checkpoints: `checkpoints/overnight-improve-324156/`
- ZIPs and selected local scoring settings: `submissions/overnight-improve-324156/`

Target is official Delta m above 5. No official improvement is guaranteed or measured by local evaluation.
ZIP generation does not upload to Codabench; official scores require separate upload.
Three training tasks are expected to take roughly 10–11 hours in two waves based on prior Blackwell
throughput; each task has a seven-hour allocation limit.

Validation of changes: 15 relevant tests passed, syntax checks passed. A legacy detection metric test
requires an absent old dataset and was excluded after its missing-file failure.
