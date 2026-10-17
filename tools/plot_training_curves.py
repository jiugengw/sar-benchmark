"""Draw per-epoch training curves for every packaged ZIP and list the ZIPs whose history is missing.

Each ZIP under submissions/ or nonsubmissions/ has a `<zip stem>.run.json` beside it that says where its training
history lives (see docs/RUN_TEMPLATE.md). This script draws `<zip stem>.curves.png` from it: training vs validation
loss (to spot overfitting) and validation mAP, macro-F1 and delta m per epoch, with the selected epoch marked.

    python tools/plot_training_curves.py            # redraw every ZIP's curves and docs/TRAINING_CURVES.md
    python tools/plot_training_curves.py --check    # exit 1 if a ZIP lacks run.json or curves (documented gaps only warn)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F1_REF, MAP_REF = 0.9331, 0.2613


def delta_m(f1, map_score):
    return 50 * (f1 / F1_REF + map_score / MAP_REF - 2)


def read_csv(path):
    with open(path, newline='') as stream:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    return [{k: float(v) for k, v in row.items() if v not in ('', None)} for row in rows]


def load(source):
    """One history as {'title', 'epochs', 'losses': {name: (train, val)}, 'metrics': {name: values}, 'note'}."""
    path = ROOT / source['path']
    kind = source['type']
    if kind == 'ultralytics':  # Ultralytics results.csv, optionally with the joint image-classification columns
        rows = read_csv(path)
        epochs = [r['epoch'] for r in rows]
        losses = {}
        for name in ('box_loss', 'cls_loss', 'image_cls_loss'):
            if f'train/{name}' in rows[0]:
                losses[name] = ([r[f'train/{name}'] for r in rows], [r.get(f'val/{name}') for r in rows])
        metrics = {'mAP50-95': [r['metrics/mAP50-95(B)'] for r in rows], 'mAP50': [r['metrics/mAP50(B)'] for r in rows]}
        if 'metrics/image_macro_f1' in rows[0]:
            metrics['macro-F1'] = [r['metrics/image_macro_f1'] for r in rows]
            metrics['Δm'] = [r['metrics/combined_delta_m'] for r in rows]
        return dict(epochs=epochs, losses=losses, metrics=metrics)
    if kind == 'terramind':  # train.py: metrics.json val_history (+ training_history.json losses for newer runs)
        summary = json.loads(path.read_text())
        history = summary['val_history']
        metrics = {'mAP50-95': [r['mAP'] for r in history], 'mAP50': [r['AP50'] for r in history],
                   'macro-F1': [r['macro_f1'] for r in history],
                   'Δm': [r.get('delta_m', delta_m(r['macro_f1'], r['mAP'])) for r in history]}
        losses, note = {}, 'validation loss was not recorded by train.py; judge overfitting by the validation metrics'
        training = summary.get('training_history') or []
        if not training and source.get('training'):
            training = json.loads((ROOT / source['training']).read_text())
        if training:
            losses['training loss (sum of components)'] = (
                [sum(r['mean_losses'].values()) for r in training], [None] * len(training))
            loss_epochs = [r['epoch'] for r in training]
        else:
            loss_epochs, note = [], 'per-epoch training loss was not saved by this older run'
        return dict(epochs=[r['epoch'] for r in history], losses=losses, loss_epochs=loss_epochs,
                    metrics=metrics, note=note)
    if kind == 'classifier_csv':  # image head fitted on frozen detector features
        rows = read_csv(path)
        return dict(epochs=[r['epoch'] for r in rows],
                    losses={'image-classifier loss': ([r['train_loss'] for r in rows], [None] * len(rows))},
                    metrics={'macro-F1': [r['macro_f1'] for r in rows]},
                    note='detector frozen; only the image classifier trains')
    if kind == 'quick_head':  # tools comparing two image-head recipes
        rows = json.loads(path.read_text())['history']
        recipe = source.get('recipe') or max({r['recipe'] for r in rows}, key=lambda name: name)
        rows = [r for r in rows if r['recipe'] == recipe]
        return dict(epochs=[r['epoch'] for r in rows], losses={},
                    metrics={'macro-F1 (full holdout)': [r['full_f1'] for r in rows],
                             'macro-F1 (clean holdout)': [r['clean_f1'] for r in rows]},
                    note=f'detector frozen; image-head recipe "{recipe}"; no loss saved')
    raise ValueError(f'unknown history type {kind}')


def plot(run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    histories = [(source.get('label', source['type']), load(source)) for source in run['histories']]
    fig, axes = plt.subplots(len(histories), 2, figsize=(12, 3.6 * len(histories)), squeeze=False)
    for (label, h), (loss_ax, metric_ax) in zip(histories, axes):
        loss_epochs = h.get('loss_epochs') or h['epochs']
        for i, (name, (train, val)) in enumerate(h['losses'].items()):
            color = f'C{i}'
            loss_ax.plot(loss_epochs, train, color=color, label=f'train {name}')
            if any(v is not None for v in val):
                loss_ax.plot(loss_epochs, val, color=color, linestyle='--', label=f'val {name}')
        loss_ax.set_title(f'{label}: loss (solid = train, dashed = validation)', fontsize=10)
        if not h['losses']:
            loss_ax.text(0.5, 0.5, 'no loss history saved', ha='center', va='center', transform=loss_ax.transAxes)
        dm = h['metrics'].get('Δm')
        for i, (name, values) in enumerate(h['metrics'].items()):
            if name == 'Δm':
                continue
            metric_ax.plot(h['epochs'], values, color=f'C{i}', label=name)
        if dm:
            twin = metric_ax.twinx()
            twin.plot(h['epochs'], dm, color='black', linewidth=1, label='Δm (right axis)')
            twin.set_ylabel('Δm')
            best = max(range(len(dm)), key=dm.__getitem__)
            metric_ax.axvline(h['epochs'][best], color='grey', linestyle=':', label=f'best Δm, epoch {h["epochs"][best]:g}')
            twin.legend(loc='lower right', fontsize=8)
        metric_ax.set_title(f'{label}: validation metrics', fontsize=10)
        for ax in (loss_ax, metric_ax):
            ax.set_xlabel('epoch')
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc='best')
        if h.get('note'):
            loss_ax.annotate(h['note'], (0, -0.28), xycoords='axes fraction', fontsize=8, color='dimgray')
    fig.suptitle(run['zip'] + (f'  ({run["note"]})' if run.get('note') else ''), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def resolve(run_path):
    """The run plus its parent's histories (post-processed or classifier-only packages inherit the detector's)."""
    run = json.loads(run_path.read_text())
    histories = list(run.get('histories', []))
    if run.get('parent'):
        parent = resolve(ROOT / run['parent'])
        histories = parent['histories'] + histories
    return dict(run, histories=histories)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='only verify; do not draw')
    args = parser.parse_args()
    zips = sorted(p for folder in ('submissions', 'nonsubmissions') for p in (ROOT / folder).rglob('*.zip'))
    rows, problems, warnings = [], [], []
    for zip_path in zips:
        stem = zip_path.with_suffix('')
        run_path = stem.with_name(stem.name + '.run.json')
        rel = zip_path.relative_to(ROOT)
        if not run_path.is_file():
            problems.append(f'{rel}: missing {run_path.name}')
            rows.append((rel, None, 'no run.json'))
            continue
        run = resolve(run_path)
        if not run['histories']:
            if run.get('missing'):  # a documented gap is reported but does not fail --check
                warnings.append(f'{rel}: no drawable history ({run["missing"]})')
            else:
                problems.append(f'{rel}: no history and no "missing" reason in {run_path.name}')
            rows.append((rel, None, run.get('missing', 'no history recorded')))
            continue
        png = stem.with_name(stem.name + '.curves.png')
        if not args.check:
            plot(run, png)
        elif not png.is_file():
            problems.append(f'{rel}: {png.name} not drawn yet')
        rows.append((rel, png.relative_to(ROOT), run.get('note', '')))
    if not args.check:
        lines = ['# Training curves', '', 'Back to the [README](../README.md). One row per packaged ZIP. Regenerate with',
                 '`python tools/plot_training_curves.py`; what each run must save is in [RUN_TEMPLATE.md](RUN_TEMPLATE.md).', '',
                 'How to read them:', '',
                 '- **Still learning:** validation mAP / F1 / Δm still rising at the last epoch; more epochs may help.',
                 '- **Overfitting:** training loss keeps falling while validation loss flattens or rises, and validation',
                 '  metrics stop improving or drop. Train shorter, regularize, or pick an earlier checkpoint.',
                 '- **Plateau:** both losses flat and metrics flat; change the recipe rather than training longer.',
                 '- The dotted line marks the epoch with the best validation Δm, when F1 and mAP were both recorded.', '',
                 '| ZIP | Curves | Note |', '|---|---|---|']
        for rel, png, note in rows:
            link = f'[curves](../{png})' if png else '**missing**'
            lines.append(f'| `{rel}` | {link} | {note} |')
        (ROOT / 'docs/TRAINING_CURVES.md').write_text('\n'.join(lines) + '\n')
    drawn = sum(png is not None for _, png, _ in rows)
    print(f'{drawn} of {len(rows)} ZIPs have curves')
    for warning in warnings:
        print('  warning: ' + warning)
    for problem in problems:
        print('  PROBLEM: ' + problem)
    if args.check and problems:
        sys.exit(1)


if __name__ == '__main__':
    main()
