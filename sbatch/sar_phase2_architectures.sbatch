#!/bin/bash
# One uninterrupted architecture bundle: reuse completed E5, then run E6, E7, and E9.
# Submit with --dependency=afterok:<phase1_job_id> after E5's existing task ends.
# The student partition has a 24-hour maximum allocation.
#SBATCH --job-name=sar-phase2-architectures
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-phase2-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

DATA="data/course-data"
DATASET="data/yolo11s-sar-512"
SPLIT="runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"
BASELINE_RESULTS="results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0"
BASELINE_CHECKPOINT="checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt"
MANIFEST="$DATASET/conversion_manifest.json"
E5_ARRAY_TASK="328337_1"

record_legacy_compute() {
    local run_root="$1"
    local array_task="$2"
    local log_prefix="$3"
    python - "$run_root" "$array_task" "$log_prefix" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

run_root, job_id, log_prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
elapsed = None
allocated_gpu = None
try:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=ElapsedRaw,AllocTRES"],
        check=True, capture_output=True, text=True,
    )
    for row in result.stdout.splitlines():
        fields = row.split("|")
        if len(fields) >= 2 and fields[0].isdigit():
            elapsed = int(fields[0])
            match = re.search(r"gres/gpu:([^,=]+)=\d+", fields[1])
            allocated_gpu = match.group(1) if match else None
            break
except (OSError, subprocess.CalledProcessError):
    pass

gpu_mem = []
log = Path("runs") / f"{log_prefix}-{job_id}.out"
if log.is_file():
    gpu_mem = [float(value) for value in re.findall(
        r"GPU_mem\s+([0-9.]+)G", log.read_text(errors="replace")
    )]
settings = {}
settings_path = run_root / "benchmark_settings.json"
if settings_path.is_file():
    settings = json.loads(settings_path.read_text())
record = {
    "source": "Slurm accounting plus Ultralytics training log for the completed array task",
    "slurm_array_task": job_id,
    "gpu_name": allocated_gpu,
    "wall_seconds": elapsed,
    "gpu_hours": elapsed / 3600.0 if elapsed is not None else None,
    "peak_vram_gib_observed_console": max(gpu_mem) if gpu_mem else None,
    "peak_vram_note": "Maximum Ultralytics GPU_mem value in this task's array log; the original run did not capture the allocator peak.",
    "total_parameters": settings.get("trainable_parameters"),
    "trainable_parameters": settings.get("trainable_parameters"),
    "best_weights": str(run_root / "train/weights/best.pt"),
}
(run_root / "compute.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
PY
}

copy_experiment_artifacts() {
    local experiment="$1"
    local run_root="$2"
    local result_root="$3"
    local package_dir="$4"
    mkdir -p "$package_dir"
    cp "$result_root/metrics.json" "$package_dir/${experiment}_${SLURM_JOB_ID}.metrics.json"
    if [[ -f "$run_root/compute.json" ]]; then
        cp "$run_root/compute.json" "$package_dir/${experiment}_${SLURM_JOB_ID}.compute.json"
    fi
    if [[ -f "$run_root/experiment_config.json" ]]; then
        cp "$run_root/experiment_config.json" "$package_dir/${experiment}_${SLURM_JOB_ID}.config.json"
    elif [[ -f "$run_root/benchmark_settings.json" ]]; then
        cp "$run_root/benchmark_settings.json" "$package_dir/${experiment}_${SLURM_JOB_ID}.config.json"
    fi
    if [[ -f "$run_root/image_classifier.metrics.json" ]]; then
        cp "$run_root/image_classifier.metrics.json" "$package_dir/${experiment}_${SLURM_JOB_ID}.classifier.metrics.json"
    fi
    for report in standard_preflight.json p2_preflight.json validation_history.json; do
        if [[ -f "$run_root/$report" ]]; then
            cp "$run_root/$report" "$package_dir/$report"
        fi
    done
}

package_completed_e5() {
    local run_root="$1"
    local result_root="$2"
    local pretrained="$3"
    local run_name="${run_root##*/}"
    local run_suffix="${run_name##*-}"
    local source_package="nonsubmissions/161026/${run_name}"
    local source_zip="$source_package/yolo26m_100_p2_${run_suffix}.zip"
    local package_dir="nonsubmissions/sar-experiments/E5-${SLURM_JOB_ID}"
    local best="$run_root/train/weights/best.pt"

    if [[ ! -f "$best" || ! -f "$run_root/image_classifier.pt" ]]; then
        echo "E5 artifacts are incomplete in $run_root" >&2
        exit 1
    fi
    mkdir -p "$result_root" "$package_dir/scoring_result"
    touch "$package_dir/scoring_result/.gitkeep"

    # Re-evaluate only if the existing metrics do not identify the trained best
    # checkpoint and include the current combined metric.
    if ! python - "$result_root/metrics.json" "$best" <<'PY'
import json
import sys
from pathlib import Path
metrics_path, best_path = map(Path, sys.argv[1:])
try:
    payload = json.loads(metrics_path.read_text())
    training = payload.get("training", {})
    scores = payload.get("val", {})
    matches = Path(training.get("best_weights", "")).resolve() == best_path.resolve()
    raise SystemExit(0 if matches and "combined_delta_m" in scores else 1)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
PY
    then
        python -u models/yolo/run_experiment.py \
            --data "$DATA" --dataset-yaml "$DATASET/dataset.yaml" \
            --dataset-root "$DATASET" --split "$SPLIT" \
            --baseline-results "$BASELINE_RESULTS" --baseline-checkpoint "$BASELINE_CHECKPOINT" \
            --weights "$pretrained" --train-output "$run_root" \
            --results-output "$result_root" --epochs 100 --batch 16 --workers 4 \
            --classifier-checkpoint "$run_root/image_classifier.pt" \
            --model-label "YOLO26m-P2 E5" --eval-only
    fi

    if [[ ! -s "$source_zip" ]]; then
        mkdir -p "$source_package/scoring_result"
        touch "$source_package/scoring_result/.gitkeep"
        python -u models/yolo/make_submission.py \
            --weights "$best" --classifier-checkpoint "$run_root/image_classifier.pt" \
            --box-class-policy native_detector_classes --data "$DATA" --classes "$MANIFEST" \
            --out "$source_zip" --batch 16 --workers 4
    fi
    cp "$source_zip" "$package_dir/E5_yolo26m_p2_${run_suffix}.zip"
    copy_experiment_artifacts E5 "$run_root" "$result_root" "$package_dir"

    # The pre-existing array job did not write compute.json. Record its Slurm
    # allocation duration and logged GPU-memory high-water mark beside E5.
    if [[ ! -f "$run_root/compute.json" ]]; then
        record_legacy_compute "$run_root" "$E5_ARRAY_TASK" "sar-yolo26-p2-100"
        cp "$run_root/compute.json" "$package_dir/E5_${SLURM_JOB_ID}.compute.json"
    fi
}

package_prior_yolo26s() {
    local run_root="runs/yolo26s-100-p2-328338"
    local result_root="results/yolo26s-100-p2-328338/seed_0"
    local array_task="328337_0"
    local package_dir="nonsubmissions/sar-experiments/YOLO26s-prior-${SLURM_JOB_ID}"
    local zip_path="$package_dir/YOLO26s_P2_328338.zip"
    local best="$run_root/train/weights/best.pt"
    if [[ ! -f "$best" || ! -f "$run_root/image_classifier.pt" || ! -f "$result_root/metrics.json" ]]; then
        echo "Completed YOLO26s-P2 artifacts are incomplete; refusing to create a partial package" >&2
        exit 1
    fi
    mkdir -p "$package_dir/scoring_result"
    touch "$package_dir/scoring_result/.gitkeep"
    if [[ ! -s "$zip_path" ]]; then
        python -u models/yolo/make_submission.py \
            --weights "$best" --classifier-checkpoint "$run_root/image_classifier.pt" \
            --box-class-policy native_detector_classes --data "$DATA" --classes "$MANIFEST" \
            --out "$zip_path" --batch 16 --workers 4
    fi
    cp "$result_root/metrics.json" "$package_dir/YOLO26s_P2_328338.metrics.json"
    if [[ ! -f "$run_root/compute.json" ]]; then
        record_legacy_compute "$run_root" "$array_task" "sar-yolo26-p2-100"
    fi
    cp "$run_root/compute.json" "$package_dir/YOLO26s_P2_328338.compute.json"
    cp "$run_root/benchmark_settings.json" "$package_dir/YOLO26s_P2_328338.config.json"
    cp "$run_root/image_classifier.metrics.json" "$package_dir/YOLO26s_P2_328338.classifier.metrics.json"
}

train_evaluate_package() {
    local experiment="$1"
    local model_label="$2"
    local architecture="$3"
    local pretrained="$4"
    local run_root="$5"
    local epochs="$6"
    local batch="$7"
    local classifier_checkpoint="$8"
    local result_root="results/${run_root##*/}/seed_0"
    local best_checkpoint="$9"
    local package_dir="nonsubmissions/sar-experiments/${experiment}-${SLURM_JOB_ID}"

    mkdir -p "$result_root"
    python -u models/yolo/train_image_head.py \
        --architecture "$architecture" --detector-checkpoint "$best_checkpoint" \
        --data "$DATA" --dataset-root "$DATASET" --split "$SPLIT" \
        --output "$classifier_checkpoint" --epochs 100 --batch 32 --workers 4
    python -u models/yolo/run_experiment.py \
        --data "$DATA" --dataset-yaml "$DATASET/dataset.yaml" \
        --dataset-root "$DATASET" --split "$SPLIT" \
        --baseline-results "$BASELINE_RESULTS" --baseline-checkpoint "$BASELINE_CHECKPOINT" \
        --weights "$pretrained" --architecture "$architecture" \
        --train-output "$run_root" --results-output "$result_root" \
        --epochs "$epochs" --batch "$batch" --workers 4 \
        --classifier-checkpoint "$classifier_checkpoint" \
        --model-label "$model_label" --eval-only
    python reports/training-comparison/build_comparison.py

    mkdir -p "$package_dir/scoring_result"
    touch "$package_dir/scoring_result/.gitkeep"
    python -u models/yolo/make_submission.py \
        --weights "$best_checkpoint" --architecture "$architecture" \
        --classifier-checkpoint "$classifier_checkpoint" \
        --box-class-policy native_detector_classes --data "$DATA" --classes "$MANIFEST" \
        --out "$package_dir/${experiment}_${SLURM_JOB_ID}.zip" --batch 16 --workers 4
    copy_experiment_artifacts "$experiment" "$run_root" "$result_root" "$package_dir"
}

mapfile -t E5_CANDIDATES < <(find runs -maxdepth 1 -type d -name 'yolo26m-100-p2-*' -printf '%T@ %p\n' | sort -nr | sed 's/^[^ ]* //')
E5_RUN=""
for candidate in "${E5_CANDIDATES[@]}"; do
    if [[ -f "$candidate/train/weights/best.pt" && -f "$candidate/image_classifier.pt" ]]; then
        E5_RUN="$candidate"
        break
    fi
done
if [[ -z "$E5_RUN" ]]; then
    echo "Could not locate the completed E5 YOLO26m-P2 run" >&2
    exit 1
fi
package_prior_yolo26s
package_completed_e5 "$E5_RUN" "results/${E5_RUN##*/}/seed_0" pretrained/yolo26m.pt
python reports/training-comparison/build_comparison.py

E6_RUN="runs/yolo26n-100-p2-${SLURM_JOB_ID}"
python -u models/yolo/train_yolo26s_p2.py \
    --scale n --weights pretrained/yolo26n.pt --train-output "$E6_RUN" \
    --epochs 100 --batch 16 --workers 4
train_evaluate_package E6 "YOLO26n-P2 E6" yolo "$E6_RUN/pretrained_yolo26n.pt" \
    "$E6_RUN" 100 16 "$E6_RUN/image_classifier.pt" "$E6_RUN/train/weights/best.pt"

E7_RUN="runs/E7-rtdetr-l-p2-${SLURM_JOB_ID}"
python -u models/yolo/train_rtdetr_p2.py \
    --weights pretrained/rtdetr-l.pt --output "$E7_RUN" --epochs 100 --batch 8 --workers 4
train_evaluate_package E7 "RT-DETR-L-P2 E7" rtdetr pretrained/rtdetr-l.pt \
    "$E7_RUN" 100 8 "$E7_RUN/image_classifier.pt" "$E7_RUN/train/weights/best.pt"

E9_RUN="runs/yolo11n-100-p2-${SLURM_JOB_ID}"
python -u models/yolo/train_100_epoch.py \
    --scale n --variant p2 --weights pretrained/yolo11n.pt \
    --dataset-yaml "$DATASET/dataset.yaml" --classes "$MANIFEST" \
    --train-output "$E9_RUN" --epochs 100 --batch 16 --workers 4
train_evaluate_package E9 "YOLO11n-P2 E9" yolo pretrained/yolo11n.pt \
    "$E9_RUN" 100 16 "$E9_RUN/image_classifier.pt" "$E9_RUN/train/weights/best.pt"

python reports/training-comparison/build_comparison.py
python - "$SLURM_JOB_ID" <<'PY'
import csv
import json
import sys
from pathlib import Path

job_id = sys.argv[1]
rows = []
for experiment in ("E5", "E6", "E7", "E9"):
    package = Path("nonsubmissions/sar-experiments") / f"{experiment}-{job_id}"
    metrics_files = sorted(package.glob(f"{experiment}_*.metrics.json"))
    compute_files = sorted(package.glob(f"{experiment}_*.compute.json"))
    if not metrics_files:
        raise FileNotFoundError(f"Missing {experiment} metrics in {package}")
    report = json.loads(metrics_files[0].read_text())
    score = report.get("val", {})
    compute = json.loads(compute_files[0].read_text()) if compute_files else {}
    training = report.get("training", {})
    config = report.get("report_metadata", {}).get("configuration", {})
    rows.append({
        "experiment": experiment,
        "architecture": config.get("backbone", experiment),
        "epochs": training.get("completed_epochs"),
        "macro_f1": score.get("macro_f1"),
        "mAP": score.get("mAP"),
        "combined_delta_m": score.get("combined_delta_m", score.get("delta_m")),
        "gpu_hours": compute.get("gpu_hours"),
        "peak_vram_allocated_gib": compute.get("peak_vram_allocated_gib",
                                                   compute.get("peak_vram_gib_observed_console")),
        "submission_zip": next((str(path) for path in package.glob("*.zip")), ""),
    })
prior_package = Path("nonsubmissions/sar-experiments") / f"YOLO26s-prior-{job_id}"
prior_metrics_path = prior_package / "YOLO26s_P2_328338.metrics.json"
if prior_metrics_path.is_file():
    report = json.loads(prior_metrics_path.read_text())
    score = report.get("val", {})
    delta = score.get("combined_delta_m", score.get("delta_m"))
    if delta is None and score.get("macro_f1") is not None and score.get("mAP") is not None:
        delta = 100.0 * 0.5 * (
            (float(score["macro_f1"]) - 0.9331) / 0.9331
            + (float(score["mAP"]) - 0.2613) / 0.2613
        )
    compute_path = prior_package / "YOLO26s_P2_328338.compute.json"
    compute = json.loads(compute_path.read_text()) if compute_path.is_file() else {}
    rows.append({
        "experiment": "extra completed run",
        "architecture": "YOLO26s-P2",
        "epochs": report.get("training", {}).get("completed_epochs"),
        "macro_f1": score.get("macro_f1"),
        "mAP": score.get("mAP"),
        "combined_delta_m": delta,
        "gpu_hours": compute.get("gpu_hours"),
        "peak_vram_allocated_gib": compute.get("peak_vram_allocated_gib",
                                                   compute.get("peak_vram_gib_observed_console")),
        "submission_zip": next((str(path) for path in prior_package.glob("*.zip")), ""),
    })
summary = Path("nonsubmissions/sar-experiments") / f"phase2_comparison_{job_id}.csv"
with summary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(summary)
PY
echo "Phase 2 bundle completed: E5 reused; E6, E7, and E9 trained, evaluated, and packaged."
