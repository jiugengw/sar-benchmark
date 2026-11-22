# job_dh (1 Nov, last hour): small density head on the frozen final detector, trained on all labelled images, then
# used to boost the final ZIP's original boxes (score >= 0.001) of bridge and playground. No holdout check.
#   python job_dh.py FINAL.zip OUT_DIR
import json, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ultralytics import YOLO

DET = 'runs/soup-s1fs/alldata-s1fs-six.pt'
TRAIN = Path('data/course-data/train'); VAL = Path('data/course-data/val')
BOOST = {3, 7}  # bridge, playground
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev = 'cuda'
model = YOLO(DET).model.to(dev).float().eval()
for p in model.parameters():
    p.requires_grad_(False)
feats = {}
model.model[-1].register_forward_pre_hook(lambda m, inp: feats.__setitem__('x', inp[0]))


def load(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float() / 255


@torch.no_grad()
def features(batch):
    with torch.autocast('cuda', dtype=torch.float16):
        model(batch.to(dev))
    xs = [x.float() for x in feats['x']]
    k = next(i for i, x in enumerate(xs) if x.shape[-1] <= 128)
    size = xs[k].shape[-2:]
    return torch.cat([xs[k]] + [F.interpolate(x, size=size, mode='bilinear', align_corners=False) for x in xs[k + 1:]], 1)


coco = json.load(open(TRAIN / 'instances.json'))
anns = {}
for a in coco['annotations']:
    anns.setdefault(a['image_id'], []).append(a)
images = [im for im in coco['images'] if (TRAIN / im['file_name']).is_file()]
probe = features(load(TRAIN / images[0]['file_name'])[None])
C, H = probe.shape[1], probe.shape[-1]
print(f'{len(images)} training images; feature {C} channels at {H}x{H}', flush=True)
head = nn.Sequential(nn.Conv2d(C, 96, 3, padding=1), nn.SiLU(), nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
                     nn.Conv2d(96, 9, 1)).to(dev)
nn.init.constant_(head[-1].bias, -4.0)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing='ij')


def target(im):
    t = torch.zeros(9, H, H)
    s = H / im['width']
    for a in anns.get(im['id'], []):
        x, y, w, h = a['bbox']
        cx, cy = (x + w / 2) * s, (y + h / 2) * s
        sig = max(0.8, min(w, h) * s / 6)
        t[a['category_id'] - 1] = torch.maximum(t[a['category_id'] - 1], torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
    return t


def focal(logit, t):
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = t.ge(0.99).float()
    l = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - t) ** 4 * p ** 2 * (1 - p).log())
    return l.sum() / pos.sum().clamp(min=1)


start, BUDGET, B = time.time(), float(sys.argv[3]) if len(sys.argv) > 3 else 900, 16
rng = np.random.default_rng(0)
step = 0
for epoch in range(3):
    order = rng.permutation(len(images))
    for i in range(0, len(order), B):
        ims = [images[j] for j in order[i:i + B]]
        x = torch.stack([load(TRAIN / im['file_name']) for im in ims])
        t = torch.stack([target(im) for im in ims])
        if rng.random() < 0.5:
            x, t = x.flip(-1), t.flip(-1)
        if rng.random() < 0.5:
            x, t = x.flip(-2), t.flip(-2)
        f = features(x)
        loss = focal(head(f), t.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 50 == 0:
            print(f'epoch {epoch} step {step} loss {loss.item():.3f} {time.time() - start:.0f}s', flush=True)
        if time.time() - start > BUDGET:
            break
    if time.time() - start > BUDGET:
        break
torch.save(head.state_dict(), out / 'density_head.pt')
print(f'trained {step} steps in {time.time() - start:.0f}s', flush=True)

with zipfile.ZipFile(src) as z:
    members = {n: z.read(n) for n in z.namelist()}
dets = json.loads(members['detection.json'])
vimgs = {im['id']: im for im in json.load(open(VAL / 'images.json'))['images']}
dens = {}
head.eval()
ids = sorted(vimgs)
with torch.no_grad():
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        x = torch.stack([load(VAL / vimgs[k]['file_name']) for k in chunk])
        d = F.max_pool2d(head(features(x)).sigmoid(), 3, 1, 1).cpu().numpy()
        for k, m in zip(chunk, d):
            dens[k] = m
print(f'density maps for {len(dens)} images', flush=True)
for gmax in (1.0, 2.0):
    new, changed = [], 0
    for p in dets:
        q = dict(p)
        if p['category_id'] in BOOST and p['score'] >= 0.001 and p['image_id'] in dens:
            im = vimgs[p['image_id']]; s = H / im['width']; x, y, w, h = p['bbox']
            cx = int(min(H - 1, max(0, (x + w / 2) * s))); cy = int(min(H - 1, max(0, (y + h / 2) * s)))
            d = float(dens[p['image_id']][p['category_id'] - 1, cy, cx])
            q['score'] = 0.001 + (p['score'] - 0.001) * (1 + gmax * d) / (1 + gmax)  # stays in [0.001, 1]: above every hedge copy and tail box
            changed += 1
        new.append(q)
    target_zip = out / f'density-boost-g{gmax:g}.zip'
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, data in members.items():
            z.writestr(n, json.dumps(new).encode() if n == 'detection.json' else data)
    print(f'{target_zip}: {changed} boxes boosted, {len(new)} boxes', flush=True)
print('JOB_D_DONE', flush=True)
