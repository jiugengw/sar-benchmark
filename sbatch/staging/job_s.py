# job_s: does each step of the final pipeline still help on NEW scenes? (31 Oct.)
# No training and no inference: rescores holdout predictions that already exist, one step
# added at a time, by image group (job_g.py's groups: twins, flipped copies, near, mid and NEW scenes < 0.6).
#  Boxes (steps 1-4 of make_final_zip.py): raw boxes >= 0.001 -> + rounding to whole pixels -> + hedging -> + low-score
#    tail, on the plain stride-1 model 337666, the class-selective flip model 344678 (the final recipe) and refit 340514
#    (the one holdout run predicted down to score 1e-5, so the only one with a tail).
#  Image labels (step 5): third-stage classifier, branch head, conv head and their 50/50 average (training-split heads
#    runs/s1-heads-337761), accuracy and macro-F1 per group.
#   python job_s.py
import contextlib, csv, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT, GROUPS, RELIABLE, holdout_groups

RUNS = ['ft40-s1r-memorise-337666', 'ft40-s1r-flipsel-344678']
TAIL_RUN = Path('results/tail-lowconf-340514')
HEADS = Path('runs/s1-heads-337761')


def boxes_table():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from official_mix_score import NAMES, official_shares
    from resolution_probe import coco
    from hedge_study import CHOSEN_TIERS, hedge_tiers
    from round_boxes import round_predictions
    from sarbench.twins import clean_image_ids
    gt = json.load(open(GT)); images = {im['id']: im for im in gt['images']}
    widths = {i: im['width'] for i, im in images.items()}
    clean = set(clean_image_ids(gt['images'])); groups = holdout_groups(gt)
    small = {i for i in clean if images[i]['width'] == 256}
    shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(GT)
    names = {c['id']: c['name'] for c in G.dataset['categories']}
    nan = float('nan')

    def row(label, b):
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = coco(GT, b, small); _, y = coco(GT, b, clean - small)
            D = G.loadRes([dict(p) for p in b if p['image_id'] in images])
        mix = np.mean([100 * (y[n] if x.get(n, nan) != x.get(n, nan) else x[n] if y.get(n, nan) != y.get(n, nan)
                              else shares[n] * x[n] + (1 - shares[n]) * y[n]) for n in NAMES])
        per = {}
        for g in GROUPS:
            ids = sorted(i for i in images if groups[i] == g)
            with contextlib.redirect_stdout(io.StringIO()):
                ev = COCOeval(G, D, 'bbox'); ev.params.imgIds = ids; ev.evaluate(); ev.accumulate()
            prec = ev.eval['precision'][:, :, :, 0, -1]
            per[g] = {names[c]: (100 * prec[:, :, k][prec[:, :, k] > -1].mean() if (prec[:, :, k] > -1).any() else nan)
                      for k, c in enumerate(ev.params.catIds)}
        new = per['new <0.6']; mean = lambda d: np.nanmean(list(d.values()))
        print(f"{label:40s} {mix:7.2f} {np.mean([new[n] for n in RELIABLE]):6.2f} {mean(new):6.2f} {mean(per['flipped']):7.2f} "
              f"{mean(per['near 0.7-0.9']):6.2f} {mean(per['mid 0.6-0.7']):6.2f} {mean(per['twins']):6.2f} | "
              + ' '.join(f'{new[names[c]]:8.1f}' for c in sorted(names)), flush=True)

    def tail(hedged, low):
        """The sub-0.001 boxes appended below every hedged box (scores score/0.001 * 1e-12, under tier 2's >= 9e-10),
        each image and class filled up to the scorer's 100 boxes, as lowconf_tail.py does."""
        room = defaultdict(lambda: 100)
        for p in hedged:
            room[p['image_id'], p['category_id']] -= 1
        out = list(hedged)
        for p in sorted(low, key=lambda p: -p['score']):
            key = (p['image_id'], p['category_id'])
            if room[key] > 0:
                room[key] -= 1
                out.append({**p, 'score': p['score'] / 0.001 * 1e-12})
        return out

    print(f"\n{'run / steps':40s} {'off-mix':>7s} {'new6':>6s} {'new9':>6s} {'flipped':>7s} {'near':>6s} {'mid':>6s} {'twins':>6s} | "
          + ' '.join(f'{names[c][:8]:>8s}' for c in sorted(names)) + '   (per class: new scenes)')
    sources = [(r, json.load(open(f'results/{r}/one2many/predictions_val.json')), []) for r in RUNS]
    lowconf = json.load(open(TAIL_RUN / 'predictions_lowconf.json'))
    sources.append(('refit-340514', [p for p in lowconf if p['score'] >= 0.001], [p for p in lowconf if p['score'] < 0.001]))
    for run, raw, low in sources:
        short = run.replace('ft40-s1r-', '')
        rounded = round_predictions(raw, widths, 'whole-pixel-classes')
        hedged = hedge_tiers(rounded, CHOSEN_TIERS)
        row(f'{short}: 1 raw boxes', raw)
        row(f'{short}: 2 + rounding', rounded)
        row(f'{short}: 3 + hedging', hedged)
        if low:
            row(f'{short}: 4 + low-score tail', tail(hedged, round_predictions(low, widths, 'whole-pixel-classes')))
        print()


def labels_table():
    from sarbench.twins import clean_image_ids
    gt = json.load(open(GT)); groups = holdout_groups(gt)
    with open('data/course-data/train/labels.csv', newline='') as stream:
        truth = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    val_ids = json.load(open(HEADS / 'branch_head.json'))['val_ids']
    branch, conv = np.load(HEADS / 'branch_head.holdout_probs.npy'), np.load(HEADS / 'conv_head.holdout_probs.npy')
    third = json.load(open(f'results/{RUNS[0]}/one2many/image_classification_predictions.json'))
    preds = {'third-stage classifier': {int(k): int(v) for k, v in third.items()},
             'branch head': dict(zip(val_ids, branch.argmax(1).tolist())),
             'conv head': dict(zip(val_ids, conv.argmax(1).tolist())),
             'two heads 50/50 (final)': dict(zip(val_ids, ((branch + conv) / 2).argmax(1).tolist()))}
    clean = set(clean_image_ids(gt['images']))
    ids = [i for i in val_ids if i in groups and i in clean]
    print(f'image labels: {len(ids)} clean holdout images with a group ({len(val_ids)} head ids, {len(groups)} grouped)')

    def f1(pairs):
        s = []
        for c in range(9):
            tp = sum(t == c and p == c for t, p in pairs); fp = sum(t != c and p == c for t, p in pairs)
            fn = sum(t == c and p != c for t, p in pairs)
            if tp + fp + fn:
                s.append(2 * tp / (2 * tp + fp + fn))
        return 100 * np.mean(s)
    cols = ['all'] + GROUPS
    print(f"{'classifier':26s} " + ' '.join(f'{c[:12]:>16s}' for c in cols) + '   (accuracy / macro-F1, n)')
    for name, pred in preds.items():
        cells = []
        for g in cols:
            sel = [i for i in ids if g == 'all' or groups[i] == g]
            pairs = [(truth[i], pred[i]) for i in sel if i in pred]
            cells.append(f'{100 * np.mean([t == p for t, p in pairs]):5.1f}/{f1(pairs):5.1f} {len(pairs):3d}')
        print(f'{name:26s} ' + ' '.join(f'{c:>16s}' for c in cells), flush=True)


if __name__ == '__main__':
    labels_table()
    boxes_table()
    print('JOB_S_DONE', flush=True)
