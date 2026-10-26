"""End-to-end check of DatasetFlip: flip TRAINING images (as stored for YOLO, 512 px) with their YOLO labels and compare
with the labels of the HOLDOUT images that are flipped copies of them. Most boxes should match exactly (IoU ~1).

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/dataset_flip_check.py   # needs data/yolo11s-sar-512
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from finetune_detector import DatasetFlip
from ultralytics.utils.instance import Instances
Y = Path('data/yolo11s-sar-512'); N = 64
imap = json.loads((Y / 'image_map.json').read_text())
widths = {k.split('/')[-1]: v['width'] for k, v in imap.items()}
by_orig = {v['file_name']: k for k, v in imap.items()}  # images/00001.jpg -> images/train/00001.png or images/val/...
F = {1: lambda a: a[..., :, ::-1], 2: lambda a: a[..., ::-1, :], 3: lambda a: a[..., ::-1, ::-1]}
def th(keys): return np.stack([np.asarray(Image.open(Y / k).convert('L').resize((N, N), Image.BILINEAR), dtype=np.float32) for k in keys])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
def yolo_boxes(key, S=512):
    p = Y / key.replace('images/', 'labels/').replace('.png', '.txt')
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return np.array([[float(c), (float(x) - float(w) / 2) * S, (float(y) - float(h) / 2) * S, (float(x) + float(w) / 2) * S, (float(y) + float(h) / 2) * S] for c, x, y, w, h in rows]).reshape(-1, 5)
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
train = [k for k in imap if k.startswith('images/train/')]; val = [k for k in imap if k.startswith('images/val/')]
P, sp = norm(th(train)); X = th(val)
res = defaultdict(list); flip_tool = DatasetFlip(widths)
for t, f in F.items():
    A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
    for i in np.nonzero(C.max(1) >= 0.95)[0][:60]:
        src = train[int(C[i].argmax())]; tb = yolo_boxes(src); hb = yolo_boxes(val[i])
        if not len(tb) or not len(hb): continue
        img = np.asarray(Image.open(Y / src).convert('RGB'))
        lab = {'img': img, 'im_file': str(Y / src), 'dataset_flip': t,
               'instances': Instances(tb[:, 1:].astype(np.float32), np.zeros((0, 0, 2), np.float32), None, bbox_format='xyxy', normalized=False)}
        out = flip_tool(lab)
        ob = out['instances'].bboxes
        for k, b in enumerate(ob):
            best = max((iou(b, h[1:]) for h in hb if h[0] == tb[k, 0]), default=0)
            res[(widths[src.split('/')[-1]], t)].append(best)
        # pixels: the flipped training image should equal the holdout image up to JPEG noise
        hv = np.asarray(Image.open(Y / val[i]).convert('L'), dtype=np.float32); fv = out['img'][..., 0].astype(np.float32)
        res[(widths[src.split('/')[-1]], t, 'pix')].append(np.corrcoef(hv.ravel(), fv.ravel())[0, 1])
for k, v in sorted(res.items(), key=str):
    v = np.array(v)
    print(k, f'n={len(v)}', f'IoU>=0.99: {100*(v>=0.99).mean():.0f}%  median {np.median(v):.3f}' if len(k) == 2 else f'pixel corr median {np.median(v):.4f}')
