# DenoDet V2 on the course benchmark

Uses the authors' [DenoDet V2 implementation](https://github.com/GrokCV/GrokSAR)
at commit `d41fb0e5cbaaf9121174357c7a48547b278fe458`: FFTResNet-50, phase/amplitude
cross denoising, FPN, and the GFL detection head. A small image classification
head pools the same five FPN outputs; both tasks share one image forward pass.

Initialization is **ImageNet-1K ResNet-50 V1 only**, downloaded from torchvision
and checked against its SHA256 prefix. No published SAR checkpoint is loaded.
The FFT attention, FPN, detector head, and image head start with fresh weights.
All **32,939,931 parameters** are trained, within the benchmark's 40M trainable /
130M total parameter limits, which the trainer checks before training.

## Setup and run

From the repository root:

```bash
bash scripts/setup_denodet.sh
sbatch sbatch/denodet_v2.sbatch
```

The job requests one RTX Pro 6000 Blackwell GPU and uses a separate
`.denodet-venv`, reusing the existing PyTorch 2.9.0 CUDA 12.8 installation.
MMCV 2.0.1 is built from source for compute capability 12.0. Compatibility
changes are confined to the generated dependencies: C++17, the current tensor
dtype API, and duplicate Adafactor registration in MMEngine. PyTorch is told
to load the locally generated MMEngine checkpoints with their metadata.

Defaults: **48 epochs**, batch 8, seed 0, FP32, validation each epoch,
DAdaptAdam, 500 warmup iterations, and learning-rate drops at epochs 32 and 44.
This scales the authors' 12-epoch schedule to a longer budget for the smaller
course dataset; it does not assume that a fixed epoch count ensures convergence.
Change the duration or batch:

```bash
sbatch --export=ALL,EPOCHS=24,BATCH=16 sbatch/denodet_v2.sbatch
```

The dataset adapter reuses the existing 512x512 grayscale PNGs and the exact
8,453/939 training/holdout split used by the YOLO comparisons. Training-only
pixel statistics supply normalization. Detection and classification use all
nine course classes. Single-GPU BatchNorm replaces distributed SyncBatchNorm.

Before training, the job checks GPU NMS, a real-image forward/backward pass,
finite losses/gradients, and parameter counts. A nonfinite training loss stops
the job. Checkpoints are saved every epoch (two latest retained, plus the best
holdout detection mAP checkpoint). Final results and the submission use that
same best checkpoint for both tasks.

## Outputs and tracking

For Slurm job `<id>`:

| Path | Contents |
|---|---|
| `runs/sar-denodet-v2-<id>.out` | Main Slurm log |
| `runs/denodet-v2-<id>/mmcv-build.log` | First-run CUDA build log |
| `runs/denodet-v2-<id>/` | Resolved config, weights, per-epoch MMEngine metrics, initialization provenance, parameter counts, compute |
| `results/denodet-v2-<id>/seed_0/` | Final original-pixel COCO metrics, classification metrics, predictions, split |
| `nonsubmissions/<DDMMYY>/denodet-v2-<id>/denodet-v2-<id>.zip` | `classification.csv` and `detection.json` |

```bash
squeue -u "$USER"
tail -f runs/sar-denodet-v2-<id>.out
tail -f runs/denodet-v2-<id>/mmcv-build.log
```

MMEngine's per-epoch COCO evaluation uses the 512px annotations. Final exported
metrics rescale predictions to original image pixels, matching the existing
benchmark's small-object area definitions. No ensemble or test-time augmentation
is used; creating the ZIP does not upload it to Codabench.

Job `329456` was submitted while the default was 12 epochs. Before its CUDA build
finished, `runs/denodet-v2-329456/run_settings.json` was set to `{"epochs": 48}`.
The trainer reads this optional per-run file before preparing data and logs the
resolved duration; it takes precedence over the submitted CLI epoch count.
This changes the upcoming training duration without interrupting the CUDA build.
Changing this file after the trainer starts does not alter a running schedule.

## Recovery and continuation

Run `329477` failed while saving epoch 20 after a disk-quota error. Recovery job
`329870` deliberately stopped at epoch 32 with `--epochs 48 --stop-after-epoch 32`,
preserving the original learning-rate drops at epochs 32 and 44. Its completed
[results](../../results/denodet-v2-329477-epoch32/) and
[submission](../../nonsubmissions/161026/denodet-v2-329477-epoch32/) are archived.

Job `330220` resumed the saved epoch-32 optimizer/model state toward epoch 48
using `sbatch/denodet_v2_resume.sbatch` (four-hour allocation, 32 GB host memory).
No epoch-48 evaluation has been saved to the repository, so the epoch-32 snapshot
(local Δm +14.23 full, +6.23 clean) is the latest DenoDet result. It was never
submitted officially and stays well below the YOLO26m-P2 line. Local weights and
logs remain in `runs/denodet-v2-329477/` on the cluster.
