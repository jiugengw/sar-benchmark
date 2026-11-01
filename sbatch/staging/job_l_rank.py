# job_l_rank: two re-ranking signals the image classifier doesn't give (1 Nov, user: "there has to be a way"). Refit
# 340514's holdout boxes, scored as submitted (round + hedge + tail; official-mix baseline 58.55).
#  rank   images hold few bridges / playgrounds (~3 / ~2 per holdout image), so an image's k-th box of such a class is
#         more often wrong: score x gamma^(k - K) beyond its top K boxes of that class.
#  nms    stricter class-wise NMS for the small classes: overlapping candidates at IoU >= t kept once.
#   python job_l_rank.py
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_k import LOW, Scorer, split, submitted

FEW = {2: 'airport', 3: 'bridge', 5: 'harbor', 7: 'playground'}


def rank_factor(raw, classes, K, gamma):
    groups = defaultdict(list)
    for p in raw:
        if p['category_id'] in classes:
            groups[p['image_id'], p['category_id']].append(p)
    f = {}
    for ps in groups.values():
        for k, p in enumerate(sorted(ps, key=lambda p: -p['score'])):
            if k >= K:
                f[p['image_id'], p['category_id'], p['score']] = gamma ** (k - K + 1)
    return lambda p: f.get((p['image_id'], p['category_id'], p['score']), 1.0)


def nms(raw, classes, t):
    import torch
    from torchvision.ops import nms as tv_nms
    keep, groups = [p for p in raw if p['category_id'] not in classes], defaultdict(list)
    for p in raw:
        if p['category_id'] in classes:
            groups[p['image_id'], p['category_id']].append(p)
    for ps in groups.values():
        b = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (p['bbox'] for p in ps)], dtype=torch.float32)
        keep += [ps[k] for k in tv_nms(b, torch.tensor([p['score'] for p in ps]), t).tolist()]
    return keep


def main():
    s = Scorer()
    raw, low = split(json.load(open(LOW)))
    s.row('baseline: round + hedge + tail', submitted(raw, low, s.widths))
    bp = {3, 7}
    for K, gamma in ((1, 0.7), (2, 0.7), (3, 0.7), (2, 0.5), (3, 0.5), (5, 0.5)):
        s.row(f'rank bridge+playground: top {K}, then x{gamma}^k', submitted(raw, low, s.widths, rank_factor(raw, bp, K, gamma)))
    for K, gamma in ((2, 0.7), (3, 0.5)):
        s.row(f'rank airport+bridge+harbor+playground {K}/{gamma}', submitted(raw, low, s.widths, rank_factor(raw, set(FEW), K, gamma)))
    for t in (0.5, 0.3):
        s.row(f'NMS {t} for bridge+playground', submitted(nms(raw, bp, t), low, s.widths))
    print('JOB_L_RANK_DONE', flush=True)


if __name__ == '__main__':
    main()
