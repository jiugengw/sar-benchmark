"""Snapshot matched medium/large training curves; rebuild in the project Python environment."""
import csv
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sarbench-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
    'plain': {
        'Medium': 'results/training-curves/yolo26m_p2_finetune_329644/continue_1/results.csv',
        'Large': 'results/training-curves/yolo26l-matched-best-330632-plain/training.csv',
    },
    'speckle': {
        'Medium': 'results/training-curves/ft40-speckle-330447/training.csv',
        'Large': 'results/training-curves/yolo26l-matched-best-330632-speckle/training.csv',
    },
}
colors = {'Medium': '#2563eb', 'Large': '#ea580c'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.7))
checked = datetime.datetime.now().astimezone()
summary = {'checked_at': checked.isoformat(), 'sources': SOURCES, 'stages': {}}
snapshot = {}
for col, (stage, paths) in enumerate(SOURCES.items()):
    histories = {}
    for label, path in paths.items():
        with (ROOT / path).open() as stream:
            histories[label] = list(csv.DictReader(stream))
    snapshot[stage] = histories
    last_large = int(histories['Large'][-1]['epoch'])
    summary['stages'][stage] = {'large_completed_epochs': last_large}
    for label, rows in histories.items():
        epochs = [int(r['epoch']) for r in rows]
        maps = [100 * float(r['metrics/mAP50-95(B)']) for r in rows]
        observed = [i for i, e in enumerate(epochs) if e <= last_large]
        axes[0, col].plot([epochs[i] for i in observed], [maps[i] for i in observed],
                          lw=2.2, color=colors[label], label=f'YOLO26{label[0].lower()}')
        if label == 'Medium' and last_large < epochs[-1]:
            future = [i for i, e in enumerate(epochs) if e >= last_large]
            axes[0, col].plot([epochs[i] for i in future], [maps[i] for i in future],
                              ls='--', lw=1.7, alpha=.55, color=colors[label], label='Medium later epochs')
        if label == 'Large':
            axes[0, col].scatter([epochs[-1]], [maps[-1]], color=colors[label], s=28, zorder=5)
        for split, linestyle in [('train', '-'), ('val', '--')]:
            axes[1, col].plot(epochs, [float(r[f'{split}/box_loss']) for r in rows], color=colors[label],
                              lw=1.6, ls=linestyle, label=f'{label} {split}')
        paired = [r for r in rows if int(r['epoch']) <= last_large]
        summary['stages'][stage][label] = {
            'latest_epoch': int(rows[-1]['epoch']), 'latest_mAP_percent': maps[-1],
            'best_mAP_percent': max(maps),
            'same_epoch_mAP_percent': float(paired[-1]['metrics/mAP50-95(B)']) * 100,
            'best_through_large_epoch_percent': max(float(r['metrics/mAP50-95(B)']) * 100 for r in paired),
        }
    title = 'Plain continuation: 40 epochs, lr 0.002' if stage == 'plain' else 'Speckle continuation: 40 epochs, lr 0.001'
    axes[0, col].set(title=title, xlabel=f'{stage.capitalize()} stage epoch', ylabel='Training-validation mAP (%)',
                     xlim=(1, 41), ylim=(39, 48))
    axes[0, col].legend(loc='lower right', fontsize=9)
    axes[1, col].set(title='Box loss: solid = train; dashed = validation', xlabel=f'{stage.capitalize()} stage epoch',
                     ylabel='Box loss', xlim=(1, 41))
    axes[1, col].legend(ncol=2, fontsize=9, loc='upper right')
    if last_large < 40:
        for row in range(2):
            axes[row, col].axvline(last_large, color='#64748b', ls=':', lw=1)
        axes[0, col].text(last_large + .8, 39.6, f'Large: {last_large}/40 complete', color='#475569', fontsize=9)
    for row in range(2):
        axes[row, col].grid(alpha=.2)
finished = all(int(histories['Large'][-1]['epoch']) == 40 for histories in snapshot.values())
status = 'completed training' if finished else 'training progress'
fig.suptitle(f'Matched YOLO26m vs YOLO26l: {status}', fontsize=16, fontweight='bold', y=.98)
fig.text(.5, .925, 'Same baseline procedure, split, seed, augmentations and continuation schedule; size changes.',
         ha='center', color='#475569', fontsize=10)
note = 'Both continuation stages completed.' if finished else 'Large speckle stage is still running.'
fig.text(.5, .025, f'Snapshot: {checked:%d %b %Y, %H:%M} SGT. {note}\n'
         'Original 100-epoch baselines were effectively random-initialized. These are training-validation curves; exported scores are shown separately.',
         ha='center', fontsize=8.5, color='#475569')
fig.tight_layout(rect=(0, .08, 1, .91), h_pad=2.3)
for suffix in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'comparison.{suffix}', dpi=160)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
print(json.dumps(summary['stages'], indent=2))

comparison_path = ROOT / 'results/yolo26l-matched-best-330632/architecture_comparison.json'
if comparison_path.exists():
    data = json.loads(comparison_path.read_text())['before_box_calibration']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ['Full holdout', 'Clean holdout']
    for offset, scale, color in [(-.18, 'medium', colors['Medium']), (.18, 'large', colors['Large'])]:
        values = [100 * data[scale][subset]['mAP'] for subset in ['full', 'clean']]
        bars = axes[0].bar([i + offset for i in range(2)], values, width=.34, color=color, label=scale.capitalize())
        axes[0].bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=3)
        tag = 'speckle-cand' if scale == 'medium' else 'yolo26l-matched-best-330632'
        calibrated = json.loads((ROOT / 'results' / tag / 'seed_0/metrics.json').read_text())['val']
        values = [calibrated['combined_delta_m'], calibrated['clean_holdout']['combined_delta_m']]
        bars = axes[1].bar([i + offset for i in range(2)], values, width=.34, color=color, label=scale.capitalize())
        axes[1].bar_label(bars, labels=[f'+{v:.2f}' for v in values], padding=3)
    axes[0].set(title='Exported detection mAP, before calibration', ylabel='mAP (%)', ylim=(0, 60))
    axes[1].set(title='Calibrated Delta m, includes fit images', ylabel='Delta m', ylim=(0, 55))
    for ax in axes:
        ax.set_xticks(range(2), labels)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Final matched-size results: medium leads full; clean scores are close', fontsize=13, fontweight='bold')
    fig.text(.5, .02, 'Single seed. Same classifier recipe refitted to each detector. Local holdouts only; no official score for large.',
             ha='center', fontsize=8.5, color='#475569')
    fig.tight_layout(rect=(0, .07, 1, .9))
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'final-results.{suffix}', dpi=160)

# Matplotlib's SVG paths contain trailing spaces; normalize them for repository checks.
for path in OUT.glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
