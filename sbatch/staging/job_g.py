# job_g: which model does best on NEW scenes? (31 Oct, user: "make sure generalise for phase 2, so means must be good
# on new unseen images, thats our goal"; "judge if our current setup is actually good for phase 2 too".)
#  1. Builds weighted averages of training-split checkpoints that share one start (ft40-s1r-third-337119) and exports
#     their holdout predictions with the submission code (deadline_recovery.evaluate_export, one2many branch).
#  2. Scores each listed run rounded + hedged as submitted: official-mix (clean images, as new_scene_runs.py) and AP per
#     image group, the groups by whole-image similarity to the training images (64 px correlation under the four
#     flips, as where_lost.py): twins, flipped copies, near-copies (0.7-0.9), mid (0.6-0.7) and NEW scenes (< 0.6).
#     new6 averages the six classes with at least 95 new-scene labels (aircraft, bridge, car, oil tank, playground,
#     ship); airport (13), harbor (15) and wind turbine (8) have too few to rank runs, so new9 swings with them.
#   python job_g.py SPEC.json
#   SPEC: {"build": {"tag": [["checkpoint", weight], ...], ...}, "score": ["tag", ...]}
import contextlib, copy, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')

CLASSIFIER = Path('runs/third-stage-331789-package/image_classifier.pt')  # scores the holdout labels only
GT = 'results/ft40-s1r-fliprefit-340514/one2many/instances_val_holdout.json'
GROUPS_CACHE = Path.home() / 'sbatch-staging/holdout_groups.json'
GROUPS = ['twins', 'flipped', 'near 0.7-0.9', 'mid 0.6-0.7', 'new <0.6']
RELIABLE = ('aircraft', 'bridge', 'car', 'oil_tank', 'playground', 'ship')
FLIPS = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]


def mix_weights(parts, out):
    """One model whose weights (and batch-norm statistics) are the weighted mean of the checkpoints'."""
    import torch
    from swa_study import checkpoint_model
    template, model = checkpoint_model(Path(parts[0][0]))
    total = sum(w for _, w in parts)
    states = [(model.state_dict(), parts[0][1])] + [(checkpoint_model(Path(p))[1].state_dict(), w) for p, w in parts[1:]]
    merged = {k: sum(s[k] * (w / total) for s, w in states) if v.is_floating_point() else v for k, v in states[0][0].items()}
    model.load_state_dict(merged)
    checkpoint = copy.copy(template)
    checkpoint.update(model=copy.deepcopy(model).half(), ema=None, optimizer=None, averaged_from=[[str(p), w] for p, w in parts])
    torch.save(checkpoint, out)


def load(im, n=64):
    return np.asarray(Image.open(f'data/course-data/train/{im["file_name"]}').convert('L').resize((n, n), Image.BILINEAR),
                      dtype=np.float32)


def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s


def holdout_groups(gt):
    if GROUPS_CACHE.exists():
        return {int(k): v for k, v in json.loads(GROUPS_CACHE.read_text()).items()}
    from sarbench.twins import clean_image_ids
    held = {im['file_name'] for im in gt['images']}
    pool = [im for im in json.load(open('data/course-data/train/instances.json'))['images'] if im['file_name'] not in held]
    best, kind = {}, {}
    for w in sorted({im['width'] for im in gt['images']}):
        tg = [im for im in gt['images'] if im['width'] == w]; pl = [im for im in pool if im['width'] == w]
        if not tg or not pl:
            continue
        P, sp = norm(np.stack([load(im) for im in pl])); X = np.stack([load(im) for im in tg])
        for k, f in enumerate(FLIPS):
            A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0; m = C.max(1)
            for i, im in enumerate(tg):
                if m[i] > best.get(im['id'], -2):
                    best[im['id']], kind[im['id']] = float(m[i]), k
    clean = set(clean_image_ids(gt['images']))

    def group_of(r, k):
        if r >= 0.9:
            return 'twins' if k == 0 else 'flipped'
        return 'near 0.7-0.9' if r >= 0.7 else 'mid 0.6-0.7' if r >= 0.6 else 'new <0.6'
    groups = {im['id']: 'twins' if im['id'] not in clean else group_of(best.get(im['id'], -1), kind.get(im['id'], 0))
              for im in gt['images']}
    GROUPS_CACHE.write_text(json.dumps(groups))
    return groups


def build(spec):
    from models.yolo.deadline_recovery import evaluate_export
    for tag, parts in spec.get('build', {}).items():
        out = Path('results') / tag / 'one2many'
        if (out / 'predictions_val.json').exists():
            print(f'{tag}: predictions already there', flush=True)
            continue
        missing = [p for p, _ in parts if not Path(p).is_file()]
        if missing:
            print(f'{tag}: SKIPPED, missing {missing}', flush=True)
            continue
        if len(parts) == 1:
            model = Path(parts[0][0])
        else:
            model = Path('results') / tag / 'model.pt'; model.parent.mkdir(parents=True, exist_ok=True)
            mix_weights(parts, model)
        evaluate_export(model, CLASSIFIER, 'one2many', out)
        print(f'{tag}: exported', flush=True)


def score(runs):
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
    print('holdout images per group: ' + ', '.join(f'{g} {sum(1 for v in groups.values() if v == g)}' for g in GROUPS))
    print(f"\n{'run':34s} {'off-mix':>7s} {'new6':>6s} {'new9':>6s} {'flipped':>7s} {'near':>6s} {'mid':>6s} {'twins':>6s} | "
          + ' '.join(f'{names[c][:8]:>8s}' for c in sorted(names)) + '   (per class: new scenes)')
    nan = float('nan')
    for run in runs:
        path = Path(f'results/{run}/one2many/predictions_val.json')
        if not path.exists():
            print(f'{run[:34]:34s} no predictions', flush=True)
            continue
        b = hedge_tiers(round_predictions(json.load(open(path)), widths, 'whole-pixel-classes'), CHOSEN_TIERS)
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
        print(f"{run[:34]:34s} {mix:7.2f} {np.mean([new[n] for n in RELIABLE]):6.2f} {mean(new):6.2f} {mean(per['flipped']):7.2f} "
              f"{mean(per['near 0.7-0.9']):6.2f} {mean(per['mid 0.6-0.7']):6.2f} {mean(per['twins']):6.2f} | "
              + ' '.join(f'{new[names[c]]:8.1f}' for c in sorted(names)), flush=True)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_text())
    build(spec)
    score(spec.get('score', []) or list(spec.get('build', {})))
    print('JOB_G_DONE', flush=True)
