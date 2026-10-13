# Run this repo as a batch job on Origami

This submits the repo's existing `train.py` command as a Slurm job. It uses the account settings you shared: the
`student` partition/account, `studentqos`, one GPU, four CPU cores, 32 GB RAM, and a one-day time limit. It does not
modify the training code.

## Before submitting

1. Put a copy of this repository on Origami, including `train.py`, `sarbench/`, `requirements.txt`, and the course
   data. The cluster cannot read files that exist only on your Mac. Your account's scratch path is
   `/common/scratch/users/j/jgwong.2023/`; using a working copy there keeps training outputs together.
2. Set up Python 3.12 and install this repo's `requirements.txt` in your cluster environment, following the
   cluster's software/network guidance. The pretrained ViT weights are downloaded on first use, so that first run
   needs Hugging Face access or a cached copy of the weights.
3. Extract the course archives so these files exist:
   `data/course-data/train/labels.csv` and `data/course-data/train/instances.json`.
4. From the repository root, create the log directory and submit:

   ```bash
   mkdir -p logs
   sbatch slurm/train_vit_lora.sbatch
   ```

The script runs ViT with pretrained weights and LoRA for 24 epochs on the labeled training data, validating each
epoch. Change the command-line options in a personal copy of the Slurm script if you want another experiment.
Training outputs go to the repo's normal `runs/`, `results/`, and `checkpoints/` directories. Job output and errors
are written to `logs/`.

To check the queue and completed job status, use `squeue -u "$USER"` and `sacct -j JOB_ID`. Replace `JOB_ID` with
the number printed by `sbatch`.
