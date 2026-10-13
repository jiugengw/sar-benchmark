#!/bin/bash
#SBATCH --job-name=sar-overnight
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load Python/3.12.8-GCCcore-13.3.0
if [[ ! -f .venv/bin/activate ]]; then
    echo "Create the project environment first: python -m venv .venv" >&2
    echo "Then: .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi
source .venv/bin/activate
# Availability: sinfo -p student -N -o '%N %t %G'
DATA_DIR="${DATA_DIR:-data/course-data}"
[[ -f "$DATA_DIR/train/labels.csv" && -f "$DATA_DIR/train/instances.json" ]] || {
    echo "Extract training data into $DATA_DIR before submission." >&2
    exit 1
}
[[ -f "$DATA_DIR/val/images.json" ]] || {
    echo "Extract val.tar.gz into $DATA_DIR before submission; submission ZIPs need val/images.json." >&2
    exit 1
}
command -v timeout >/dev/null
python -u -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
python -m pytest -q tests/test_detection_experiments.py

# See docs/OVERNIGHT_PLAN.md for hypotheses and comparisons.
# Twelve NEW detection configurations, fixed learning rates (both 1e-4).
# All use 24 epochs, seed 0, batch size 16, adapters only, evaluation every epoch.
# One process at a time. Each has a five-hour cap INCLUDING training and submission prediction.
# Timeout ends an unfinished run; only completed runs have final checkpoints.
CAP_SECONDS=18000
# Read Slurm's actual remaining allocation time before each run; no fixed early deadline.
ROOT="overnight-${SLURM_JOB_ID}"
STATUS="results/$ROOT/status.tsv"
mkdir -p "results/$ROOT"
python inspect_training_data.py --data "$DATA_DIR" --out "results/$ROOT/data_diagnostics.json"
printf 'experiment\tanchors\tgeometry\tsampling\tbox_weight\texit_code\tseconds\n' > "$STATUS"

run_experiment() {
    local name=$1 anchors=$2 geometry=$3 sampling=$4 box_weight=$5
    local end_time now remaining
    end_time=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
    if [[ -z "$end_time" || "$end_time" == "Unknown" || "$end_time" == "N/A" ]]; then
        echo "Cannot determine allocation end time; stopping before starting another run."
        return 1
    fi
    now=$(date +%s)
    remaining=$(( $(date -d "$end_time" +%s) - now ))
    if (( remaining < CAP_SECONDS + 120 )); then
        echo "Only ${remaining}s remain; another full capped run cannot fit. Stopping."
        return 1
    fi
    local root="$ROOT/$name" started=$SECONDS code=0
    mkdir -p "runs/$root"
    echo "Starting $name: anchors=$anchors geometry=$geometry sampling=$sampling box_weight=$box_weight"
    timeout --signal=TERM --kill-after=60s "${CAP_SECONDS}s" \
        python -u train_and_package.py \
        --data "$DATA_DIR" \
        --checkpoint-dir "checkpoints/$root/terramind_pretrained_moelora/seed_0" \
        --submission "submissions/$root/submission.zip" -- \
        --data "$DATA_DIR" --backbone terramind --init pretrained --adapt moelora \
        --epochs 24 --batch-size 16 --seed 0 --unfreeze-last 0 \
        --backbone-lr 1e-4 --lr 1e-4 \
        --anchor-profile "$anchors" --geometry "$geometry" \
        --class-sampling "$sampling" --box-loss-weight "$box_weight" \
        --eval-every 1 --workers 4 \
        --out "runs/$root" --results "results/$root" --checkpoints "checkpoints/$root" \
        > "runs/$root/console.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$anchors" "$geometry" "$sampling" "$box_weight" "$code" "$((SECONDS - started))" >> "$STATUS"
    echo "Finished $name with exit code $code (124 means time cap reached)."
    # Abort on errors rather than spending the overnight allocation repeating a broken setup.
    if (( code != 0 && code != 124 && code != 137 )); then
        tail -n 40 "runs/$root/console.log"
        exit "$code"
    fi
}

while read -r name anchors geometry sampling box_weight; do
    run_experiment "$name" "$anchors" "$geometry" "$sampling" "$box_weight" || break
done <<'EXPERIMENTS'
01_wide_anchors wide stretch uniform 1
02_large_anchors large stretch uniform 1
03_preserve_geometry default pad uniform 1
04_class_balance default stretch sqrt-balanced 1
05_box_localization default stretch uniform 2
06_wide_with_localization wide stretch uniform 2
07_wide_with_balance wide stretch sqrt-balanced 1
08_padding_with_balance default pad sqrt-balanced 1
09_padding_with_localization default pad uniform 2
10_balance_with_localization default stretch sqrt-balanced 2
11_wide_with_padding wide pad uniform 1
12_large_with_localization large stretch uniform 2
EXPERIMENTS

echo "Overnight batch finished. Status: $STATUS"
cat "$STATUS"
