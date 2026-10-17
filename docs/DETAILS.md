# Model and training details

Back to the [README](../README.md).

The architecture below follows the TA's reference implementation. The training and file descriptions document
the current course-data code. The final score tables preserve the TA's original labeled-split measurements.

## Model

```
image [B,1,512,512] -> backbone -> tokens [B,1024,768]   (32 x 32 patches of 16 px)
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens as a [B,768,32,32] map -> SimpleFeaturePyramid (strides 4 to 64, 256 channels)
                  -> RegionProposalNetwork + RoIHeads (torchvision Faster R-CNN, 9 classes + background)
loss = cross-entropy + the four Faster R-CNN losses
```

![The reference model](reference_model.png)

**Backbones** (`--backbone`), both ViT-B/16:

- `vit`: timm's `vit_base_patch16_224.augreg_in21k` (ImageNet-21k).
- `terramind`: TerraMind-1.0-base (IBM/ESA), an Earth-observation model, through its Sentinel-1 GRD input.

**Adaptation** (`--adapt`):

- `full`: every backbone weight trains.
- `lora`: the backbone is frozen; the `qkv` and `proj` layers of every attention block get
  `y = W x + (alpha / r) B A x`, with r = 16 and alpha = 32 (0.88 M trainable backbone parameters).
- `moelora`: as `lora`, but the rank-16 update is split into 4 experts of rank 4 (alpha = 8, so alpha / r = 2
  as in `lora`), mixed per token by a softmax router. Each task has its own router (1.03 M trainable backbone
  parameters).

Every run also trains the 20.1 M parameters of the heads (neck, RPN, RoI heads, classifier), so
`trainable_parameters` in `metrics.json` is the backbone count + 20.1 M.

**Partial fine-tuning** (`--unfreeze-last N`) optionally makes the original weights in the last N encoder blocks
trainable alongside pretrained LoRA/MoE-LoRA. Remaining original backbone weights stay frozen. The unfrozen
weights have their own learning rate (`--unfrozen-lr`, default 2e-5). TerraMind + MoE-LoRA with two unfrozen blocks
has 35,304,217 trainable parameters, including heads, within the 40M limit. This mode adds `_unfreezeN` to the
configuration directory name and rejects settings that exceed the parameter budget.

## Current training protocol

- Read labels and COCO annotations from `train/labels.csv` and `train/instances.json` under `--data`.
- Hold out a seeded random 10% of the labeled training rows. With 9,392 images this gives 8,453 training images
  and 939 local validation images. This split is not stratified; a different `--seed` changes it.
- AdamW, weight decay 0.05, batch size 16, bf16 autocast, 24 epochs.
- Learning rate 1e-4. `--backbone-lr` defaults to that rate; `run_all.sh` sets it to 2e-5 for the two pretrained
  `full` runs, so their pretrained backbones are fine-tuned gently.
- One epoch of linear warmup, then cosine decay to 0, updated every iteration.
- Validate every `--eval-every` epochs (default 4) and after the last epoch. Use `--eval-every 1` for every epoch.
- Report accuracy, macro-F1, balanced accuracy, mAP, AP50, and Delta m on the labeled local holdout. Delta m uses
  fixed validation reference scores 0.9331 / 0.2613. Classification and AP values in JSON are fractions; Delta m
  is stored as a percentage value.
- Report the same scores on the clean holdout subset under `clean`: holdout images with no near-duplicate twin
  among the training images. A quarter of the labeled images have a twin (same scene and boxes, different
  speckle), listed in `sarbench/train_twins.json`; with seed 0, 720 of the 939 holdout images are clean. Twins
  inflate the full-holdout score, so compare runs on the clean scores. `python -m sarbench.twins score
  results/<run>/seed_0` re-grades a saved run.
- Save the final-epoch model. There is no best-checkpoint selection, and the weights do not contain optimizer or
  scheduler state for exact training resumption.
- The trainer does not score the separate course validation or test splits. Use `make_submission.py` to predict
  their images; Codabench provides the official scores for the submitted predictions.

## Design notes

- **Both labels on every image**, so every training batch trains both heads.
- **512 px input.** Every image is resized to 512 x 512 (non-square images are stretched): 1,024 tokens.
  Objects are small (median box 15.9 px at 512 px), so the smallest anchors are 8 px, on the stride-4 level.
  Predicted boxes are mapped back to original pixels before COCO scoring.
- **Normalization.** Images are read as one gray channel. Both backbones get the same mean and std
  (0.2526, 0.2133), computed over the 9,392 train images at 512 px.
- **Precision rule.** Backbone, neck and classification head run in bf16; the RPN and RoI heads run in fp32.
  torchvision's box coder casts anchors to the dtype of the regression output, and bf16 numbers between 256
  and 512 are 2 px apart: too coarse for objects of 6-15 px. Importing terratorch (`sarbench/backbones.py` does, for
  every run) switches fp32 matrix multiplications on the GPU to TF32; box coordinates are decoded
  elementwise in fp32 and are unaffected.
- **MoE-LoRA runs the backbone twice.** Its routers are task-specific, so the two tasks see different
  features and every forward pass runs the backbone once per task.
- **TerraMind domain gap.** TerraMind was pretrained on Sentinel-1 GRD backscatter in dB (polarizations VV
  and VH, 10 m pixels). Our images are 8-bit gray display images, mostly at a higher resolution; the gray
  channel is fed as both VV and VH. The gap is part of the benchmark and is not corrected in code.

## Time and memory

The TA's original measurements on one RTX PRO 6000 Blackwell GPU were about 50 minutes per run (LoRA about
47 minutes, MoE-LoRA about 1 hour) and about 7 hours for all eight configurations. Reported allocated memory was
about 11–17 GiB; the TerraMind + MoE-LoRA configuration did not fit a 16 GB card under that setup.

Recorded RTX 3090 (24 GB) measurements for the current local holdout, seed 0, batch size 16, 24 epochs, and
validation every epoch:

| Configuration | Training loop including validation | Peak allocated GPU memory |
|---|---:|---:|
| ViT pretrained + MoE-LoRA | 2.72 h | 11.98 GiB |
| TerraMind pretrained + MoE-LoRA | 2.82 h | 14.17 GiB |

These are single-run measurements, not estimates for every configuration. `train_hours` excludes initial
downloads, data setup, and final file export. GPU utilization, other processes, and validation frequency affect
elapsed time. `nvidia-smi` can report more memory than PyTorch's allocated-memory figure because it also includes
reserved memory and other GPU allocations. A smaller `--batch-size` saves memory but changes the protocol.

## What a run writes

`<out>/<backbone>_<init>_<adapt>/`, with `--out runs` by default:

- `log.txt`: training progress and local validation scores, including Delta m.
- `metrics.json`: arguments, parameter counts, `train_hours` (wall time of the epoch loop, including the
  periodic validation; it grows if the GPU is shared), `peak_gpu_memory_gb` (`torch.cuda.max_memory_allocated`
  in GiB), every local validation result, and the final validation metrics. `test` is `null`, since the trainer
  does not perform test evaluation.
- `predictions_val.json`: final-model detections on the local holdout, in COCO result format and original pixels.
- `instances_val_holdout.json`: correct annotations and image metadata for that same holdout.
- `model.pt`: the final weights (`state_dict`).

The current course-data trainer also copies `metrics.json`, `predictions_val.json`, and
`instances_val_holdout.json` to `results/<backbone>_<init>_<adapt>/seed_<seed>/` after a successful run. The
`--results` option changes the parent folder. Model weights and configuration are also archived under
`checkpoints/<backbone>_<init>_<adapt>/seed_<seed>/`; `--checkpoints` changes that parent folder. Logs stay in the
run directory. Detection reports are regenerated under `reports/detection-review/`;
[viewing instructions](WORKFLOW.md#view-the-report) explain local browsing and SSH port forwarding. Submission
generation is a separate command and writes prediction ZIPs rather than local validation metrics.

## Checks

```bash
# Short training-pipeline check; this is not a useful score comparison
python train.py --data data/course-data --backbone vit --init pretrained --adapt lora \
  --limit 64 --epochs 1 --out runs/smoke --results results/smoke --checkpoints checkpoints/smoke

# Original tests that do not read the dataset (require a GPU and may download weights)
python -m pytest tests/test_backbones.py tests/test_adapters.py
```

The small `--limit` is applied before making the 90/10 holdout. A 64-image limit leaves only 6 validation images,
which cannot represent all 9 classes; classification metrics can be non-finite. Use it to inspect pipeline
execution rather than rank models.

`test_data.py`, the dataset-dependent detection test in `test_metrics.py`, and `test_model.py` still expect the
TA's original dataset layout; the data/model tests also use the original loader API. They need adaptation before
the full suite is usable with course data. The hand-computed classification metric test does not need a dataset:

```bash
python -m pytest tests/test_metrics.py::test_classification_metrics_by_hand
python -m pytest tests/test_partial_finetuning.py
```

## Results in full

These are the TA's original measurements on labeled validation and test splits. The current local holdout has
939 images, so its metrics should not be read as a reproduction of these tables. Saved current scores live in
`results/`; the model architecture is shared, but the evaluation protocol differs.

Scores in %, mean ± sample standard deviation over seeds 0, 1 and 2, last-epoch model. Accuracy and macro-F1
for classification; COCO mAP (AP@[.50:.95]) and AP50 for detection. The seeds differ by a few tenths of a
point, so smaller differences are noise. Δm is computed per seed against the reference, the mean of the three
ViT-B/16 + LoRA runs (test 93.99 / 25.41, val 93.31 / 26.13), then averaged, so the reference configuration shows 0.
The current trainer's local Delta m uses the validation reference values. Full fine-tuning and training from scratch train 106 M
parameters with the heads, over the 40 M trainable budget of the assignment: they are references, full fine-tuning
the upper bound.

Test (2,065 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 95.9 ± 0.3 | 95.8 ± 0.3 | 31.8 ± 0.2 | 62.4 ± 0.1 | +13.5 ± 0.4 |
| vit | pretrained | lora | 0.88 | 94.2 ± 0.2 | 94.0 ± 0.2 | 25.4 ± 0.5 | 53.9 ± 0.4 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 93.7 ± 0.5 | 93.4 ± 0.5 | 26.8 ± 0.3 | 55.9 ± 0.5 | +2.5 ± 0.8 |
| terramind | pretrained | full | 85.3 | 95.8 ± 0.2 | 95.6 ± 0.2 | 30.8 ± 0.4 | 61.8 ± 0.4 | +11.5 ± 0.7 |
| terramind | pretrained | lora | 0.88 | 93.1 ± 0.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | 55.4 ± 1.2 | +1.4 ± 1.0 |
| terramind | pretrained | moelora | 1.03 | 92.9 ± 0.1 | 92.4 ± 0.1 | 27.8 ± 0.3 | 57.7 ± 0.4 | +3.8 ± 0.7 |
| vit | scratch | full | 86.0 | 83.9 ± 0.6 | 82.8 ± 0.7 | 17.0 ± 0.3 | 36.7 ± 0.4 | −22.5 ± 0.9 |
| terramind | scratch | full | 85.3 | 86.7 ± 0.2 | 86.1 ± 0.2 | 17.8 ± 0.5 | 38.9 ± 0.4 | −19.3 ± 0.8 |

Val (1,426 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 94.7 ± 0.3 | 94.4 ± 0.3 | 32.0 ± 0.4 | 62.7 ± 0.4 | +11.8 ± 0.7 |
| vit | pretrained | lora | 0.88 | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 92.6 ± 0.8 | 92.2 ± 0.8 | 27.1 ± 0.2 | 56.4 ± 0.3 | +1.3 ± 0.9 |
| terramind | pretrained | full | 85.3 | 94.5 ± 0.2 | 94.2 ± 0.3 | 31.5 ± 0.1 | 62.4 ± 0.0 | +10.7 ± 0.3 |
| terramind | pretrained | lora | 0.88 | 92.6 ± 0.4 | 92.2 ± 0.4 | 27.0 ± 0.5 | 55.9 ± 0.8 | +1.0 ± 1.3 |
| terramind | pretrained | moelora | 1.03 | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | +4.4 ± 0.3 |
| vit | scratch | full | 86.0 | 83.8 ± 0.8 | 83.5 ± 0.9 | 18.2 ± 0.5 | 39.0 ± 0.7 | −20.3 ± 1.3 |
| terramind | scratch | full | 85.3 | 86.0 ± 1.0 | 86.0 ± 0.9 | 19.2 ± 0.4 | 41.4 ± 0.4 | −17.3 ± 1.2 |
