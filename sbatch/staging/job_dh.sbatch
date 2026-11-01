#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/common/home/users/j/jgwong.2023/sbatch-staging/job1-%j.out
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
# ~/sbatch-staging/job_d.py was a copy of sbatch/staging/job_dh.py; final.zip the +65.80 ZIP
python -u ~/sbatch-staging/job_d.py ~/sbatch-staging/final.zip results/d-out 600
