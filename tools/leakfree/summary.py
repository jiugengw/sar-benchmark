"""Leak rates of the official course split, image groups, and a leak-free re-split of all labelled images.

Input: a coursesearch run (job_ai: OUT/ids.csv, OUT/pix/ from pixsearch.py, OUT/scale/ from scalesearch.py with
ALL_PAIRS=1) and the labels (train/instances.json, released-labels/{val,test}_instances.json).

Links (two images show the same radar pixels), rules checked against each search's controls:
  pix    same zoom, at the found flip / turn and shift, over the shared region:
           shared >= 0.25: r >= R_WIDE;  0.10 <= shared < 0.25: r >= R_THIN
  scale  zoom 1.05-2 (one image a magnified part of the other), >= 30% of the magnified image on the other: r >= R_SCALE
Kinds, for the leak rate (first that applies): copy (same zoom, >= 90% shared, no flip), flipped copy (>= 90% shared,
flipped or turned), overlap (same zoom, 10-90% shared), zoom copy.

Groups: union-find over every link (any splits). Re-split: groups, shuffled with a fixed seed and taken largest first,
each go whole to the split (train / val / test) that most needs its (class, image size) mix, with the official split's
per-split (class, image size) counts as targets (the TA's class mix in each split); so no link crosses two splits.

    python tools/leakfree/summary.py SEARCH_OUT data/course-data OUT_DIR
Output: OUT_DIR/summary.json, links.csv, split.csv (file, official split, new split, group, class, size).
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

R_WIDE, R_THIN, WIDE = 0.97, 0.99, 0.25
R_SCALE, MIN_ZOOM = 0.98, 1.05     # the course null (radar only) reaches 0.976
SPLITS = ('train', 'val', 'test')
SEED = 0


def pix_pass(df, col='r_opt'):
    return ((df.shared >= WIDE) & (df[col] >= R_WIDE)) | ((df.shared >= 0.1) & (df.shared < WIDE) & (df[col] >= R_THIN))


def scale_pass(df, col='r_opt'):
    return (df.zoom >= MIN_ZOOM) & (df[col] >= R_SCALE)


class UF:
    def __init__(self): self.p = {}
    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def labels(data):
    cls = {}
    for split, p in (('train', data / 'train' / 'instances.json'), ('val', data / 'released-labels' / 'val_instances.json'),
                     ('test', data / 'released-labels' / 'test_instances.json')):
        d = json.loads(p.read_text()); names = {c['id']: c['name'] for c in d['categories']}
        fn = {im['id']: im['file_name'] for im in d['images']}
        per = defaultdict(Counter)
        for a in d['annotations']: per[a['image_id']][names[a['category_id']]] += 1
        for i, f in fn.items(): cls[(split, f)] = per[i].most_common(1)[0][0]
    return cls


def controls(d, kind):
    res = {}
    if kind == 'pix':
        null = pd.read_csv(d / 'pairs_null.csv'); null = null[null.r_opt > -2]
        nq = json.loads((d / 'search.json').read_text())['null']['queries']
        ok = null[pix_pass(null)]
        res['null: queries with a false link'] = int(ok.q.nunique()); res['null: false link rate %'] = round(100 * ok.q.nunique() / nq, 3)
        mx = null.groupby(pd.cut(null.shared, [0.1, 0.25, 0.5, 1.01], right=False), observed=True).r_opt.max().round(3)
        res['null: max r by shared'] = {str(k): float(v) for k, v in mx.items()}
        pl = pd.read_csv(d / 'pairs_planted.csv')
        pl['f'] = pl.planted.str.split(':').str[0].astype(float); pl['ok'] = pl.planted.str.split(':').str[3].astype(int)
        truth = json.loads((d / 'planted_truth.json').read_text())
        tot = Counter(v[1] for v in truth.values())
        g = pl[(pl.ok == 1) & pix_pass(pl)]
        res['planted recall (shared: found / planted)'] = {str(f): f'{g[g.f == f].r.nunique()} / {tot[f]}' for f in sorted(tot)}
    else:
        for s in ('null1', 'null2'):
            n = pd.read_csv(d / f'pairs_{s}.csv')
            res[f'{s}: pairs passing'] = int(scale_pass(n).sum()); res[f'{s}: max r (zoom >= {MIN_ZOOM})'] = round(float(n[n.zoom >= MIN_ZOOM].r_opt.max()), 3) if len(n) else None
        truth = json.loads((d / 'planted_truth.json').read_text())
        for s, kind_ in (('planted1', 'coarse'), ('planted2', 'fine')):
            p = pd.read_csv(d / f'pairs_{s}.csv'); p = p[p.planted.astype(str).str.count(':') == 3]
            f = p.planted.str.split(':'); p = p.assign(ok=f.str[3].astype(int), inside=f.str[2].astype(int))
            g = p[(p.ok == 1) & scale_pass(p)]
            n_in = sum(1 for v in truth.values() if v['kind'].startswith(kind_) and v['inside']); n_out = sum(1 for v in truth.values() if v['kind'].startswith(kind_) and not v['inside'])
            res[f'{s} recall'] = {'fully inside': f"{g[g.inside == 1].fine.nunique() if kind_ == 'coarse' else g[g.inside == 1].coarse.nunique()} / {n_in}",
                                  'half outside': f"{g[g.inside == 0].fine.nunique()} / {n_out}"}
    return res


def resplit(groups, info, target):
    """groups: list of id lists; info[id] = (class, size); target[split][(class, size)]: images wanted (the official
    split's counts). Returns {id: split}."""
    cur = {s: Counter() for s in SPLITS}; out = {}
    rng = np.random.default_rng(SEED); order = list(rng.permutation(len(groups)))
    order.sort(key=lambda j: -len(groups[j]))                     # stable: largest first, seeded order among equals
    for j in order:
        comp = Counter(info[i] for i in groups[j])
        score = {s: sum(c * (target[s].get(k, 0) - cur[s][k]) / max(target[s].get(k, 0), 1) for k, c in comp.items()) for s in SPLITS}
        s = max(SPLITS, key=lambda s_: score[s_])
        cur[s].update(comp)
        for i in groups[j]: out[i] = s
    return out


def main(search, data, out):
    search, data, out = Path(search), Path(data), Path(out); out.mkdir(parents=True, exist_ok=True); res = {}
    ids = pd.read_csv(search / 'ids.csv'); cls = labels(data)
    ids['cls'] = [cls[(s, f)] for s, f in zip(ids.split, ids.file_name)]
    ids['size'] = np.where((ids.width == 512) & (ids.height == 512), '512', np.where((ids.width == 256) & (ids.height == 256), '256', 'other'))
    split = dict(zip(ids.id, ids.split)); info = {i: (c, z) for i, c, z in zip(ids.id, ids.cls, ids['size'])}

    res['controls'] = {'same zoom (pixsearch)': controls(search / 'pix', 'pix'), 'zoom (scalesearch)': controls(search / 'scale', 'scale')}

    # ---- links
    p = pd.read_csv(search / 'pix' / 'pairs_main.csv'); p = p[(p.r_opt > -2) & pix_pass(p)]
    p = p.assign(a=p.q, b=p.r, kind=np.where(p.shared >= 0.9, np.where(p.t == 0, 'copy', 'flipped copy'), 'overlap'))
    z = pd.read_csv(search / 'scale' / 'pairs_main1.csv'); z = z[scale_pass(z)]
    z = z.assign(a=z.fine, b=z.coarse, kind='zoom copy', shared=z.share_fine)
    L = pd.concat([p[['a', 'b', 'kind', 't', 'shared', 'r_opt']], z[['a', 'b', 'kind', 't', 'shared', 'r_opt', 'zoom']]], ignore_index=True)
    L['a_split'] = L.a.map(split); L['b_split'] = L.b.map(split)
    L.to_csv(out / 'links.csv', index=False)
    res['links'] = {'all': len(L), 'by kind': L.kind.value_counts().to_dict()}

    # ---- leak rate of the official split (val / test image with a link to a train image)
    rank = ['copy', 'flipped copy', 'overlap', 'zoom copy']; best = {}
    for x in L.itertuples():
        for u, v in ((x.a, x.b), (x.b, x.a)):
            if split[u] != 'train' and split[v] == 'train':
                if u not in best or rank.index(x.kind) < rank.index(best[u]): best[u] = x.kind
    leak = {}
    for s in ('val', 'test'):
        sel = ids[ids.split == s]; k = Counter(best.get(i, 'new') for i in sel.id)
        leak[s] = {'images': len(sel), 'leaked': sum(v for kk, v in k.items() if kk != 'new'),
                   '%': round(100 * sum(v for kk, v in k.items() if kk != 'new') / len(sel), 1), 'by kind (first that applies)': dict(k),
                   'per class %': {c: round(100 * sum(i in best for i in g.id) / len(g), 1) for c, g in sel.groupby('cls')}}
    res['official split leak rate'] = leak

    # ---- groups
    uf = UF()
    for i in ids.id: uf.find(i)
    for x in L.itertuples(): uf.union(int(x.a), int(x.b))
    G = defaultdict(list)
    for i in ids.id: G[uf.find(i)].append(int(i))
    groups = sorted(G.values(), key=len, reverse=True)
    res['groups'] = {'groups': len(groups), 'largest': [len(g) for g in groups[:10]], 'single images': sum(len(g) == 1 for g in groups),
                     'groups spanning official splits': sum(len({split[i] for i in g}) > 1 for g in groups)}

    # ---- re-split
    sizes = {s: int((ids.split == s).sum()) for s in SPLITS}
    target = {s: Counter(info[i] for i in ids.id[ids.split == s]) for s in SPLITS}   # the TA's mix in every split
    new = resplit(groups, info, target)
    gid = {i: k for k, g in enumerate(groups) for i in g}
    ids['new_split'] = ids.id.map(new); ids['group'] = ids.id.map(gid)
    cross = sum(new[int(x.a)] != new[int(x.b)] for x in L.itertuples())
    res['new split'] = {'images': {s: int((ids.new_split == s).sum()) for s in SPLITS}, 'official sizes': sizes,
                        'links crossing splits (must be 0)': int(cross),
                        'per class (train / val / test)': {c: [int(((ids.cls == c) & (ids.new_split == s)).sum()) for s in SPLITS] for c in sorted(ids.cls.unique())},
                        'official per class (train / val / test)': {c: [int(((ids.cls == c) & (ids.split == s)).sum()) for s in SPLITS] for c in sorted(ids.cls.unique())},
                        'per size (train / val / test)': {z_: [int(((ids['size'] == z_) & (ids.new_split == s)).sum()) for s in SPLITS] for z_ in ('512', '256', 'other')},
                        'images that change split': int((ids.new_split != ids.split).sum())}
    sp = ids.assign(file=ids.split + '/' + ids.file_name)[['file', 'split', 'new_split', 'group', 'cls', 'size', 'width', 'height']]
    sp = sp.rename(columns={'split': 'official_split', 'cls': 'class'}); sp.to_csv(out / 'split.csv', index=False, lineterminator='\n')
    res['new split']['split.csv md5'] = hashlib.md5((out / 'split.csv').read_bytes()).hexdigest()
    res['rules'] = {'pix': f'shared >= {WIDE}: r >= {R_WIDE}; 0.10-{WIDE}: r >= {R_THIN}', 'scale': f'zoom >= {MIN_ZOOM}: r >= {R_SCALE}', 'seed': SEED}
    (out / 'summary.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
