# job_k: two recall checks (1 Nov). Same model, one pass, nothing taken from training labels. Holdout twin of the final
# recipe = refit 340514 (training split), scored as submitted (round + hedge + low-score tail), clean official-mix and
# new6 (job_g.py's groups). Baseline: official-mix 58.55 (hard-classes-ceiling note, tail study 28 Oct).
#  rescore  A. image-class rescoring: every box's score times g(p), p = the two-head classifier's probability of the
#              box's class for its image (training-split heads runs/s1-heads-337761, branch + conv 50/50 as submitted).
#              Rescored originals stay in the original band (0.001 + 0.999 s g), so hedge tiers and tail keep their
#              order; g = 1 gives the baseline ranking exactly.
#  export   B. the one2one branch's boxes for the same holdout (make_submission.py --detection-branch one2one).
#  merge    B. one2one + one2many boxes merged by class-wise NMS (IoU 0.7), or only one2one boxes that no one2many box
#              of the class already covers, then the usual round + hedge + tail.
#   python job_k.py rescore|export|merge
import contextlib, io, json, subprocess, sys, tempfile, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT, RELIABLE, holdout_groups

LOW = Path('results/tail-lowconf-340514/predictions_lowconf.json')  # 340514 one2many boxes down to 1e-5, 1000/image
HEADS = Path('runs/s1-heads-337761')
O2O = Path('results/k-one2one-340514')
WEIGHTS = 'runs/ft40-s1r-fliprefit-340514/train/weights/best.pt'


class Scorer:
    def __init__(self):
        from pycocotools.coco import COCO
        from official_mix_score import NAMES, official_shares
        from sarbench.twins import clean_image_ids
        self.gt = json.load(open(GT)); self.images = {im['id']: im for im in self.gt['images']}
        self.widths = {i: im['width'] for i, im in self.images.items()}
        cats = [c['name'] for c in sorted(self.gt['categories'], key=lambda c: c['id'])]
        assert tuple(cats) == NAMES, cats
        self.names = NAMES
        self.clean = set(clean_image_ids(self.gt['images']))
        self.small = {i for i in self.clean if self.images[i]['width'] == 256}
        groups = holdout_groups(self.gt)
        self.new = sorted(i for i in self.images if groups.get(i) == 'new <0.6')
        self.shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'),
                                      Path('data/course-data/val/images.json'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.G = COCO(GT)
        print(f'holdout {len(self.images)} images, clean {len(self.clean)} ({len(self.small)} at 256 px), '
              f'new scenes {len(self.new)}', flush=True)
        print(f"\n{'variant':44s} {'off-mix':>7s} {'new6':>6s} {'new9':>6s} | "
              + ' '.join(f'{n[:8]:>8s}' for n in NAMES) + '   (per class: new scenes)', flush=True)

    def per_class(self, boxes, ids):
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(self.G, self.G.loadRes([dict(p) for p in boxes]), 'bbox')
            ev.params.imgIds = sorted(ids); ev.evaluate(); ev.accumulate()
        prec = ev.eval['precision'][:, :, :, 0, -1]
        return {self.names[k]: (100 * prec[:, :, k][prec[:, :, k] > -1].mean() if (prec[:, :, k] > -1).any()
                                else float('nan')) for k in range(len(self.names))}

    def row(self, label, boxes):
        boxes = [p for p in boxes if p['image_id'] in self.images]
        x, y, new = (self.per_class(boxes, ids) for ids in (self.small, self.clean - self.small, self.new))
        isnan = lambda v: v != v
        mix = np.mean([y[n] if isnan(x[n]) else x[n] if isnan(y[n]) else self.shares[n] * x[n] + (1 - self.shares[n]) * y[n]
                       for n in self.names])
        new6, new9 = np.mean([new[n] for n in RELIABLE]), np.nanmean(list(new.values()))
        print(f'{label:44s} {mix:7.2f} {new6:6.2f} {new9:6.2f} | ' + ' '.join(f'{new[n]:8.1f}' for n in self.names),
              flush=True)
        return mix, new6


def tail(hedged, low):
    """job_s.py's tail: boxes below 0.001 appended under every hedged box, each image and class filled to 100."""
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


def submitted(raw, low, widths, g=None):
    """round + (rescore) + hedge + tail, as make_submission.py --final builds the ZIP."""
    from hedge_study import CHOSEN_TIERS, hedge_tiers
    from round_boxes import round_predictions
    rounded = round_predictions(raw, widths, 'whole-pixel-classes')
    low = round_predictions(low, widths, 'whole-pixel-classes') if low else []
    if g is not None:
        rounded = [{**p, 'score': 0.001 + 0.999 * p['score'] * g(p)} for p in rounded]
        low = [{**p, 'score': p['score'] * g(p)} for p in low]
    return tail(hedge_tiers(rounded, CHOSEN_TIERS), low)


def split(boxes):
    return [p for p in boxes if p['score'] >= 0.001], [p for p in boxes if p['score'] < 0.001]


def rescore():
    s = Scorer()
    raw, low = split(json.load(open(LOW)))
    val_ids = json.load(open(HEADS / 'branch_head.json'))['val_ids']
    probs = (np.load(HEADS / 'branch_head.holdout_probs.npy') + np.load(HEADS / 'conv_head.holdout_probs.npy')) / 2
    P = {int(i): pr for i, pr in zip(val_ids, probs)}
    missing = {p['image_id'] for p in raw} - set(P)
    truth = {}
    import csv
    with open('data/course-data/train/labels.csv', newline='') as stream:
        truth = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    acc = np.mean([truth[i] == int(np.argmax(P[i])) for i in s.images if i in P])
    print(f'classifier probabilities for {len(P)} images; {len(missing)} holdout images with boxes have none (g = 1); '
          f'holdout accuracy {100 * acc:.2f}', flush=True)
    prob = lambda p: P[p['image_id']][p['category_id'] - 1] if p['image_id'] in P else 1.0
    s.row('baseline: round + hedge + tail', submitted(raw, low, s.widths))
    s.row('check g = 1 (must equal baseline)', submitted(raw, low, s.widths, lambda p: 1.0))
    for w in (0.25, 0.5, 1.0, 2.0):
        s.row(f'rescore g = p^{w}', submitted(raw, low, s.widths, lambda p, w=w: prob(p) ** w))
    for a in (0.2, 0.5):
        s.row(f'rescore g = {a} + {1 - a:.1f} p', submitted(raw, low, s.widths, lambda p, a=a: a + (1 - a) * prob(p)))
    top = {i: int(np.argmax(pr)) for i, pr in P.items()}
    s.row('rescore g = 1 for the image class, else 0.5',
          submitted(raw, low, s.widths, lambda p: 1.0 if top.get(p['image_id'], p['category_id'] - 1) == p['category_id'] - 1 else 0.5))
    print('RESCORE_DONE', flush=True)


def export():
    O2O.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(GT))
    with tempfile.TemporaryDirectory(prefix='job-k-') as t:
        view = Path(t) / 'data'; (view / 'train').mkdir(parents=True); (view / 'val').mkdir()
        (view / 'train/labels.csv').symlink_to(Path('data/course-data/train/labels.csv').resolve())
        for im in gt['images']:
            target = view / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Path('data/course-data/train', im['file_name']).resolve())
        (view / 'val/images.json').write_text(json.dumps({'images': gt['images']}))
        z = Path(t) / 'one2one.zip'
        subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', WEIGHTS,
                        '--detection-branch', 'one2one', '--data', str(view), '--out', str(z),
                        '--conf', '0.001', '--max-det', '300'], check=True)
        boxes = json.loads(zipfile.ZipFile(z).read('detection.json'))
    (O2O / 'predictions_one2one.json').write_text(json.dumps(boxes)); print('ONE2ONE_BOXES', len(boxes), flush=True)


def xyxy(p):
    x, y, w, h = p['bbox']; return [x, y, x + w, y + h]


def merge():
    import torch
    from torchvision.ops import batched_nms, box_iou
    s = Scorer()
    raw, low = split(json.load(open(LOW)))
    o2o = [p for p in json.load(open(O2O / 'predictions_one2one.json')) if p['score'] >= 0.001]
    by_image = defaultdict(lambda: ([], []))
    for p in raw:
        by_image[p['image_id']][0].append(p)
    for p in o2o:
        by_image[p['image_id']][1].append(p)
    nms_union, uncovered = [], []
    for image_id, (many, one) in by_image.items():
        both = many + one
        b = torch.tensor([xyxy(p) for p in both], dtype=torch.float32)
        sc = torch.tensor([p['score'] for p in both]); cl = torch.tensor([p['category_id'] for p in both])
        nms_union += [both[k] for k in batched_nms(b, sc, cl, 0.7).tolist()]
        uncovered += many
        if one and many:
            iou = box_iou(b[len(many):], b[:len(many)])
            same = cl[len(many):, None] == cl[None, :len(many)]
            best = torch.where(same, iou, torch.zeros_like(iou)).max(1).values
            uncovered += [p for p, v in zip(one, best.tolist()) if v < 0.5]
        else:
            uncovered += one
    print(f'one2many {len(raw)} boxes >= 0.001, one2one {len(o2o)}; NMS union {len(nms_union)}; '
          f'one2many + uncovered one2one {len(uncovered)}', flush=True)
    s.row('baseline: one2many, round + hedge + tail', submitted(raw, low, s.widths))
    s.row('one2one alone, round + hedge (no tail)', submitted(o2o, [], s.widths))
    s.row('NMS union of both branches + tail', submitted(nms_union, low, s.widths))
    s.row('one2many + uncovered one2one (IoU<0.5) + tail', submitted(uncovered, low, s.widths))
    print('MERGE_DONE', flush=True)


if __name__ == '__main__':
    {'rescore': rescore, 'export': export, 'merge': merge}[sys.argv[1]]()
