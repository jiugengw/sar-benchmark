"""Archive completed candidate curves, audit paired gains, and export DenoDet by 23:00.

This monitor does not train detector backbones, upload to Codabench, cancel jobs,
or allocate another GPU. Snapshot packaging can fit the fixed image-head recipe
and uses a small overlapping step in the model's current GPU job.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def archive_curves(tag):
    run = ROOT / 'runs' / tag
    for ready in run.glob('*_ready.json'):
        record = json.loads(ready.read_text())
        stage = record['stage']
        destination = ROOT / 'results/training-curves' / tag / stage
        destination.mkdir(parents=True, exist_ok=True)
        stage_run = run / stage
        # P1's detector logs are at its root train/; its classifier lives at p1/.
        detector_logs = run / 'train' if stage == 'p1' else stage_run
        for name in ('results.csv', 'results.png', 'args.yaml'):
            source = detector_logs / name
            if source.exists():
                shutil.copy2(source, destination / name)
        classifier = stage_run / 'image_classifier.metrics.json'
        if classifier.exists():
            shutil.copy2(classifier, destination / 'classifier.metrics.json')
            metrics = json.loads(classifier.read_text())
            with (destination / 'classifier_training.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'macro_f1'])
                writer.writeheader()
                writer.writerows(metrics['validation_history'])
            plot = destination / 'classifier_training.png'
            if not plot.exists():
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                history = metrics['validation_history']
                figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].plot([row['epoch'] for row in history], [row['train_loss'] for row in history])
                axes[0].set_ylabel('Training cross-entropy')
                axes[1].plot([row['epoch'] for row in history], [100 * row['macro_f1'] for row in history])
                axes[1].set_ylabel('939-image holdout macro-F1 (%)')
                for axis in axes:
                    axis.set_xlabel('Classifier training epoch')
                    axis.grid(alpha=.2)
                figure.tight_layout()
                figure.savefig(plot, dpi=150)
                plt.close(figure)
        audit = ROOT / 'results' / tag / stage / 'paired_audit.json'
        if not audit.exists():
            candidate = ROOT / 'results' / tag / stage / record['branch']
            command = [str(ROOT / '.venv/bin/python'), '-u',
                       str(ROOT / 'models/yolo/paired_holdout_audit.py'),
                       '--baseline', str(ROOT / 'results/yolo26m-100-p2-328339/seed_0'),
                       '--candidate', str(candidate), '--output', str(audit), '--replicates', '200']
            print('AUDIT_START', tag, stage, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            print('AUDIT_READY', audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags', nargs='+', required=True)
    parser.add_argument('--denodet-job', required=True)
    parser.add_argument('--denodet-source', type=Path, required=True)
    parser.add_argument('--export-at', required=True)
    parser.add_argument('--until', required=True)
    parser.add_argument('--p1-job')
    parser.add_argument('--p1-source', type=Path)
    parser.add_argument('--p1-export-at')
    args = parser.parse_args()
    export_at = datetime.fromisoformat(args.export_at)
    until = datetime.fromisoformat(args.until)
    exported = False
    p1_exported = False
    failed_audits = set()
    next_log = 0
    while datetime.now(timezone.utc) < until:
        for tag in args.tags:
            if tag in failed_audits:
                continue
            try:
                archive_curves(tag)
            except Exception as error:
                print('AUDIT_ERROR', tag, repr(error), flush=True)
                failed_audits.add(tag)
        now = datetime.now(timezone.utc)
        if not exported and now >= export_at:
            exported = True
            status = subprocess.run(['squeue', '-h', '-j', args.denodet_job, '-o', '%T'],
                                    capture_output=True, text=True, check=False)
            if 'RUNNING' in status.stdout:
                output = ROOT / 'runs' / f'{args.denodet_source.name}-deadline-2300'
                command = ['srun', '--jobid', args.denodet_job, '--overlap', '--ntasks=1',
                           '--cpus-per-task=1', str(ROOT / '.denodet-venv/bin/python'), '-u',
                           str(ROOT / 'models/denodet/export_checkpoint.py'),
                           '--source', str(args.denodet_source.resolve()), '--output', str(output),
                           '--batch', '2', '--workers', '1']
                print('DENODET_EXPORT_START', ' '.join(command), flush=True)
                result = subprocess.run(command, cwd=ROOT, check=False)
                print('DENODET_EXPORT_EXIT', result.returncode, flush=True)
            else:
                print('DENODET_EXPORT_SKIPPED', status.stdout.strip(), status.stderr.strip(), flush=True)
        if args.p1_export_at and not p1_exported and now >= datetime.fromisoformat(args.p1_export_at):
            p1_exported = True
            status = subprocess.run(['squeue', '-h', '-j', args.p1_job, '-o', '%T'],
                                    capture_output=True, text=True, check=False)
            if ('RUNNING' in status.stdout and not (args.p1_source / 'complete.json').exists()
                    and not (args.p1_source / 'compute.json').exists()):
                output = ROOT / 'runs' / f'{args.p1_source.name}-deadline-snapshot'
                command = ['srun', '--jobid', args.p1_job, '--overlap', '--ntasks=1',
                           '--cpus-per-task=1', str(ROOT / '.venv/bin/python'), '-u',
                           str(ROOT / 'models/yolo/export_training_snapshot.py'),
                           '--weights', str(args.p1_source.resolve() / 'train/weights/best.pt'),
                           '--output', str(output), '--feature-batch', '4']
                print('P1_SNAPSHOT_EXPORT_START', ' '.join(command), flush=True)
                result = subprocess.run(command, cwd=ROOT, check=False)
                print('P1_SNAPSHOT_EXPORT_EXIT', result.returncode, flush=True)
                if result.returncode == 0:
                    archive_curves(output.name)
            else:
                print('P1_SNAPSHOT_EXPORT_SKIPPED', status.stdout.strip(), flush=True)
        if time.monotonic() >= next_log:
            print('MONITOR_ALIVE', now.isoformat(), flush=True)
            next_log = time.monotonic() + 300
        time.sleep(30)
    print('DEADLINE_MONITOR_FINISHED', flush=True)


if __name__ == '__main__':
    main()
