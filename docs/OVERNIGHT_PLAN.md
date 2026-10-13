# Overnight detection experiments

Submit **overnight.sbatch**, not job.sbatch, for this plan. No jobs have been submitted automatically.

## What we are trying to improve

Your strongest completed official result is TerraMind + MoE-LoRA: macro-F1 91.87%, mAP 28.04%, Delta m +2.89.
The target is Delta m above +11. At macro-F1 92%, that requires approximately 32.25% mAP using the repository's
reference formula. Local holdout scores are useful for comparing methods, but do not guarantee official scores.

The best model's local bridge AP is 2.14% and playground AP is 1.64%. Local AP50 is 57.79%, while AP75 is 22.02%.
These motivate experiments with object coverage and box accuracy. They do not prove which change will help.

## Settings shared by every run

- Backbone: pretrained TerraMind; adaptation: MoE-LoRA; no original backbone blocks unfrozen.
- 24 epochs; batch size 16; seed 0; existing 90/10 labeled-training split; validation every epoch.
- Adapter learning rate and head learning rate both stay at 1e-4; weight decay stays at 0.05.
- One Blackwell GPU; one training process at a time; four data-loader workers.
- Separate output folders for every experiment and job. The completed baseline is not rerun.

Only the factors in the table below change. The script runs in this order.

## The twelve new runs

| Run | Exactly what changes | Why try it? | Compare against |
|---|---|---|---|
| 1: wide anchors | Add aspect ratios 0.2 and 5 to the existing 0.5, 1, 2; sizes unchanged. | Anchors are candidate box shapes. Long, narrow objects may be poorly covered by the current shapes. | Completed baseline |
| 2: large anchors | Change the coarsest anchor sizes from 128/181 to 181/256 pixels; ratios unchanged. | Large objects may need larger candidate boxes. This tests scale separately from shape. | Completed baseline |
| 3: preserve geometry | Resize proportionally, then pad the right/bottom to 512x512, instead of stretching to a square. | Preserve original object proportions in rectangular images. If images are all square, expect little difference. | Completed baseline |
| 4: class balance | Sample training images with probability proportional to 1/sqrt(number of training images in their class). | Give rarer classes more exposure without using aggressive full equalization. Same number of sampled images per epoch. | Completed baseline |
| 5: box localization | Double the weight of the detector's RoI box-regression loss. All other loss weights stay at 1. | Encourage accurate box boundaries; may trade off classification or other detection losses. | Completed baseline |
| 6: wide anchors + localization | Combine run 1's anchors and run 5's box-loss weight. | Test whether better candidate shapes and stronger box training help together. This is an interaction experiment, not a previously established winning combination. | Runs 1 and 5 |
| 7: wide anchors + balance | Combine runs 1 and 4. | Does class exposure help when elongated objects have more candidate shapes? | Runs 1 and 4 |
| 8: preserve geometry + balance | Combine runs 3 and 4. | Does preserving proportions change the benefit of class sampling? | Runs 3 and 4 |
| 9: preserve geometry + localization | Combine runs 3 and 5. | Do undistorted shapes help stronger box regression? | Runs 3 and 5 |
| 10: balance + localization | Combine runs 4 and 5. | Does more rare-class exposure help stronger box training? | Runs 4 and 5 |
| 11: wide anchors + preserve geometry | Combine runs 1 and 3. | Does avoiding distortion change the benefit of elongated anchors? | Runs 1 and 3 |
| 12: large anchors + localization | Combine runs 2 and 5. | Do larger candidate boxes benefit from stronger regression? | Runs 2 and 5 |

Runs 1-5 change one factor each. Runs 6-12 check specific two-factor interactions. These are hypotheses, not promised improvements.
Class sampling uses only the training subset. Official unlabeled validation images and local held-out labels are never
used for training. Padding is saved in the experiment configuration and used automatically during submission generation.

## How to submit

One-time environment setup (the batch loads the same module and activates .venv automatically):

```bash
module load Python/3.12.8-GCCcore-13.3.0
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests/test_detection_experiments.py
```

If not already extracted:

```bash
mkdir -p data/course-data
tar -xzf data/train.tar.gz -C data/course-data
tar -xzf data/val.tar.gz -C data/course-data
```

From the repository root:

```bash
sbatch overnight.sbatch
squeue -u "$USER"
```

You may disconnect after submission. The job starts when Slurm allocates a GPU.

## Timing and checks

The Slurm limit is **20 hours after the job starts**, excluding queue wait. Each run has a **five-hour cap**,
including downloads, training, evaluation, final export, and submission prediction. Before each run, the script reads the actual Slurm
allocation end time and starts only if the full cap plus two minutes fits. There is no fixed 17h40m deadline.
The observed allocation is a 24 GB MIG slice of Blackwell, with early training throughput around 17 images/second.
That suggests about 3.3 hours of training per 24-epoch run, plus validation and submission prediction. Expect roughly
four completed runs in twenty hours, rather than all twelve; actual speed can change. The script will not start
another run unless its full five-hour cap plus two minutes fits. Wider/large anchors, geometry, and balancing run first;
localization and combinations may remain unstarted. All twelve finishing is not guaranteed.

If a run exceeds its cap, it is stopped and the script moves on. The trainer saves weights only after finishing all
24 epochs, so a timed-out run has logs but no final checkpoint. Ordinary training errors stop the batch to avoid
repeating a broken setup. No submissions are uploaded to Codabench.

Before training, the script runs small CPU tests. These are code checks, not extra training experiments:

- Resize/pad a synthetic image and box, then confirm the original box coordinates are recovered.
- Confirm training and submission preprocessing agree.
- Confirm anchor counts match the proposal network's outputs.
- Confirm class sampling is restricted to the training subset.

The CPU tests do not measure model quality. Training still uses the Blackwell GPU. If the checks fail, training
does not start. The ten CPU checks passed in the installed project .venv. Python and shell syntax checks passed
as well. Full training and prediction on Blackwell still require a GPU job; these checks do not guarantee scores
or runtime.

## Where to find results tomorrow

For job ID 12345:

- Overall job output: slurm-12345.out.
- Completion table: results/overnight-12345/status.tsv (exit 0 = success; 124 = time cap; 137 can be forced termination).
- Data report: results/overnight-12345/data_diagnostics.json (class counts, image aspect ratios, anchor shape coverage).
- Per-run console: runs/overnight-12345/<experiment>/console.log.
- Per-run metrics: results/overnight-12345/<experiment>/terramind_pretrained_moelora/seed_0/metrics.json.
- Per-run model: checkpoints/overnight-12345/<experiment>/terramind_pretrained_moelora/seed_0/model.pt.
- Per-run submission: submissions/overnight-12345/<experiment>/submission.zip (classification.csv and detection.json inside).

Compare final-epoch local Delta m, mAP, macro-F1, AP75, and per-class AP to the existing local baseline
(Delta m +0.66, mAP 27.22%, macro-F1 90.65%). Local and official baseline values differ because they evaluate
different images. Look for improvement in the weak classes without damaging overall performance.

The anchor diagnostic assumes an anchor centered perfectly on each box and ignores the grid stride, so it is
an idealized shape check, not measured proposal recall. The twelve runs are predeclared hypotheses; the script
does not automatically select or combine configurations based on this report.

After every successful training run, the batch calls make_submission.py to create the validation submission ZIP.
It reads the saved anchor and geometry settings; older checkpoints default to the original settings. Generating
a ZIP does not upload it or consume a leaderboard slot. No scoring_result folder is created.
If the time cap is reached during submission prediction, training metrics and weights remain available but the ZIP
may not be finished; regenerate it later with make_submission.py.
