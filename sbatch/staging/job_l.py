# job_l: where do the holdout's bridges and playgrounds go missing, and what could the same detector still give? (1 Nov.)
# No training,
# no inference: refit 340514's holdout boxes down to score 1e-5 (results/tail-lowconf-340514), the training-split twin of
# the final recipe.
#  1. Every bridge / playground label on the clean holdout gets one outcome: found (same-class box IoU >= 0.5, score
#     >= 0.001), found low (only a sub-0.001 box), near miss (best same-class IoU 0.1-0.5), other class (another class's
#     box at IoU >= 0.5), no box. Split by image group, size and image width.
#  2. Near misses: what kind of error (centre offset, size, aspect) and which simple box change would reach IoU 0.5.
#  3. Upper bound: snap every near-miss box onto its label (an oracle; nothing a ZIP could do) and rescore the mix.
#  4. Crop sheets of new-scene labels per outcome (green label, red same-class boxes >= 0.01 with score, blue others).
#   python job_l.py OUTDIR
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT, holdout_groups
from job_k import LOW, Scorer, split, submitted

CLASSES = {3: 'bridge', 7: 'playground'}
OUTCOMES = ['found', 'found low', 'near miss', 'other class', 'no box']


def iou(a, b):
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = w * h
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def outcome(g, boxes):
    same = [p for p in boxes if p['category_id'] == g['category_id']]
    best_main = max((iou(g['bbox'], p['bbox']), p['score']) for p in same if p['score'] >= 0.001) if any(
        p['score'] >= 0.001 for p in same) else (0.0, 0.0)
    best_any = max(((iou(g['bbox'], p['bbox']), p['score'], k) for k, p in enumerate(same)), default=(0.0, 0.0, -1))
    other = max((iou(g['bbox'], p['bbox']) for p in boxes if p['category_id'] != g['category_id'] and p['score'] >= 0.001),
                default=0.0)
    if best_main[0] >= 0.5:
        return 'found', best_main, same[best_any[2]] if best_any[2] >= 0 else None
    if best_any[0] >= 0.5:
        return 'found low', best_any[:2], same[best_any[2]]
    if best_any[0] >= 0.1:
        return 'near miss', best_any[:2], same[best_any[2]]
    if other >= 0.5:
        return 'other class', (other, 0.0), None
    return 'no box', best_any[:2], None


def fixes(g, p):
    """Which single change of the near-miss box p reaches IoU 0.5 with label g."""
    x, y, w, h = p['bbox']; cx, cy = x + w / 2, y + h / 2
    out = []
    t = [cx - h / 2, cy - w / 2, h, w]
    if iou(g['bbox'], t) >= 0.5:
        out.append('swap w/h')
    for f in (1.3, 1.6, 0.75, 0.6):
        if iou(g['bbox'], [cx - w * f / 2, cy - h * f / 2, w * f, h * f]) >= 0.5:
            out.append(f'scale x{f}'); break
    for dx, dy in ((0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
        if iou(g['bbox'], [x + dx * w, y + dy * h, w, h]) >= 0.5:
            out.append('shift 25-50%'); break
    return out


def sheets(rows, images, by_image, out):
    from PIL import Image, ImageDraw
    random.seed(0)
    TILE, PER, CROP = 192, 8, 96
    for cls, name in CLASSES.items():
        picks = []
        for oc in ('no box', 'near miss', 'found low', 'found'):
            pool = [r for r in rows if r['cls'] == cls and r['outcome'] == oc and r['group'] == 'new <0.6']
            picks.append((oc, random.sample(pool, min(2 * PER, len(pool)))))
        sheet = Image.new('RGB', (PER * TILE, sum((len(p) + PER - 1) // PER * (TILE + 14) + 18 for _, p in picks)), 'white')
        d = ImageDraw.Draw(sheet); top = 0
        for oc, pool in picks:
            d.text((4, top + 3), f'{name}, new scenes: {oc} ({len(pool)} shown)', fill='black'); top += 18
            for k, r in enumerate(pool):
                im = images[r['image_id']]; s = 512 / im['width']; crop = CROP / s
                gx, gy, gw, gh = r['bbox']; cx, cy = gx + gw / 2, gy + gh / 2
                x0, y0 = min(max(0, cx - crop / 2), im['width'] - crop), min(max(0, cy - crop / 2), im['height'] - crop)
                z = TILE / crop
                tile = Image.open(f"data/course-data/train/{im['file_name']}").convert('RGB').crop(
                    (int(x0), int(y0), int(x0 + crop), int(y0 + crop))).resize((TILE, TILE), Image.NEAREST)
                t = ImageDraw.Draw(tile)
                box = lambda b: [(b[0] - x0) * z, (b[1] - y0) * z, (b[0] + b[2] - x0) * z, (b[1] + b[3] - y0) * z]
                for p in by_image[r['image_id']]:
                    if p['score'] >= 0.01:
                        t.rectangle(box(p['bbox']), outline='red' if p['category_id'] == cls else 'blue', width=1)
                t.rectangle(box(r['bbox']), outline=(0, 255, 0), width=2)
                t.text((3, TILE - 12), f"{max(gw, gh) * s:.0f}px iou{r['iou']:.2f} s{r['score']:.3f}", fill=(255, 255, 0))
                sheet.paste(tile, ((k % PER) * TILE, top + (k // PER) * (TILE + 14)))
            top += (len(pool) + PER - 1) // PER * (TILE + 14)
        sheet.save(out / f'{name}_misses.png'); print('SHEET', out / f'{name}_misses.png', flush=True)


def main(out):
    from sarbench.twins import clean_image_ids
    out.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(GT)); images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images'])); groups = holdout_groups(gt)
    boxes = json.load(open(LOW)); by_image = defaultdict(list)
    for p in boxes:
        by_image[p['image_id']].append(p)
    rows = []
    for g in gt['annotations']:
        if g['category_id'] in CLASSES and g['image_id'] in clean:
            oc, (v, sc), p = outcome(g, by_image[g['image_id']])
            im = images[g['image_id']]
            rows.append({'cls': g['category_id'], 'image_id': g['image_id'], 'bbox': g['bbox'], 'outcome': oc, 'iou': v,
                         'score': sc, 'box': p, 'group': groups.get(g['image_id'], 'other'), 'width': im['width'],
                         'side': max(g['bbox'][2], g['bbox'][3]) * 512 / im['width'],
                         'edge': min(g['bbox'][0], g['bbox'][1], im['width'] - g['bbox'][0] - g['bbox'][2],
                                     im['height'] - g['bbox'][1] - g['bbox'][3]) <= 2})
    for cls, name in CLASSES.items():
        R = [r for r in rows if r['cls'] == cls]
        print(f'\n=== {name}: {len(R)} labels on {len({r["image_id"] for r in R})} clean holdout images', flush=True)
        print(f"{'group':14s} {'n':>5s} " + ' '.join(f'{o:>12s}' for o in OUTCOMES))
        for grp in ['new <0.6', 'mid 0.6-0.7', 'near 0.7-0.9', 'flipped', 'twins', 'other', 'all']:
            S = [r for r in R if grp == 'all' or r['group'] == grp]
            if S:
                c = Counter(r['outcome'] for r in S)
                print(f'{grp:14s} {len(S):5d} ' + ' '.join(f'{100 * c[o] / len(S):11.0f}%' for o in OUTCOMES))
        N = [r for r in R if r['group'] == 'new <0.6']
        print('new scenes by size (side in 512-px pixels) and image width:')
        for lo, hi in ((0, 8), (8, 12), (12, 16), (16, 24), (24, 40), (40, 999)):
            S = [r for r in N if lo <= r['side'] < hi]
            if S:
                c = Counter(r['outcome'] for r in S)
                print(f'  side {lo:3d}-{hi:<3d} n {len(S):4d}  ' + '  '.join(f'{o} {100 * c[o] / len(S):3.0f}%' for o in OUTCOMES))
        for w in (256, 512):
            S = [r for r in N if r['width'] == w]
            if S:
                c = Counter(r['outcome'] for r in S)
                print(f'  {w}-px images n {len(S):4d}  ' + '  '.join(f'{o} {100 * c[o] / len(S):3.0f}%' for o in OUTCOMES))
        S = [r for r in N if r['edge']]
        if S:
            c = Counter(r['outcome'] for r in S)
            print(f'  touching the image edge n {len(S)}  ' + '  '.join(f'{o} {100 * c[o] / len(S):3.0f}%' for o in OUTCOMES))
        per_image = defaultdict(list)
        for r in N:
            per_image[r['image_id']].append(r['outcome'] in ('found', 'found low'))
        none = [v for v in per_image.values() if not any(v)]
        print(f'  new-scene images: {len(per_image)}, with none of their {name}s found: {len(none)} '
              f'({sum(map(len, none))} labels); all found: {sum(all(v) for v in per_image.values())}')
        near = [r for r in N if r['outcome'] == 'near miss']
        if near:
            fx = Counter(f for r in near for f in fixes(r, r['box']) or ['none'])
            off = [abs((r['box']['bbox'][0] + r['box']['bbox'][2] / 2) - (r['bbox'][0] + r['bbox'][2] / 2)) / max(r['bbox'][2], 1)
                   for r in near]
            ratio = [r['box']['bbox'][2] * r['box']['bbox'][3] / max(r['bbox'][2] * r['bbox'][3], 1) for r in near]
            print(f'  near misses {len(near)}: median IoU {np.median([r["iou"] for r in near]):.2f}, score '
                  f'{np.median([r["score"] for r in near]):.3f}; centre offset / width {np.median(off):.2f}; '
                  f'area ratio box/label {np.median(ratio):.2f}; one change reaches 0.5: {dict(fx)}')
    # 3. oracle: near-miss boxes snapped onto their labels (both branches of the ranking keep their scores)
    s = Scorer()
    raw, low = split(boxes)
    s.row('baseline: round + hedge + tail', submitted(raw, low, s.widths))
    snap = {id(r['box']): r['bbox'] for r in rows if r['outcome'] == 'near miss'}
    moved = [{**p, 'bbox': list(snap[id(p)])} if id(p) in snap else p for p in boxes]
    raw2, low2 = split(moved)
    s.row('ORACLE: near-miss bridge/playground boxes snapped', submitted(raw2, low2, s.widths))
    json.dump([{k: v for k, v in r.items() if k != 'box'} for r in rows], open(out / 'rows.json', 'w'))
    sheets(rows, images, by_image, out)
    print('JOB_L_DONE', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
