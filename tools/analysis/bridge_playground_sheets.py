"""Contact sheets of clean-holdout bridge / playground labels: found, missed in images where we find some, missed in
images where we find none. Each tile: 96 px of context around the label, x3, label corners green, our best box of the
same class red (score shown).

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/bridge_playground_sheets.py results/ft40-s1r-memorise-337666/one2many nonsubmissions/261026/bridge-miss-audit
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'models/yolo'))
from sarbench.twins import clean_image_ids
raw, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
gt = json.loads((raw / 'instances_val_holdout.json').read_text())
clean = set(clean_image_ids(gt['images']))
names = {c['id']: c['name'] for c in gt['categories']}
ims = {im['id']: im for im in gt['images']}
preds = defaultdict(list)
for p in json.loads((raw / 'predictions_val.json').read_text()):
    preds[p['image_id']].append(p)
def iou(a, b):
    iw = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)
C, Z, N = 96, 3, 15
def corners(d, x0, y0, x1, y1, color, L=6):
    for (x, y, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        d.line([(x, y), (x + dx * L, y)], fill=color, width=2); d.line([(x, y), (x, y + dy * L)], fill=color, width=2)
for cname in ['bridge', 'playground']:
    cid = [k for k, v in names.items() if v == cname][0]
    rows = []
    for a in gt['annotations']:
        if a['category_id'] != cid or a['image_id'] not in clean: continue
        same = [(iou(a['bbox'], p['bbox']), p) for p in preds[a['image_id']] if p['category_id'] == cid]
        best = max(same, key=lambda t: t[0]) if same else (0, None)
        rows.append([a, best[0], best[1]])
    per = defaultdict(list)
    for r in rows: per[r[0]['image_id']].append(r[1] >= 0.5)
    groups = {'found': [r for r in rows if r[1] >= 0.5],
              'missed-in-images-where-we-find-some': [r for r in rows if r[1] < 0.5 and any(per[r[0]['image_id']])],
              'missed-in-images-where-we-find-none': [r for r in rows if not any(per[r[0]['image_id']])]}
    for g, rs in groups.items():
        random.Random(0).shuffle(rs)
        rs = rs[:N]
        sheet = Image.new('RGB', (5 * (C * Z + 6), 3 * (C * Z + 22)), 'white')
        for k, (a, v, p) in enumerate(rs):
            im = Image.open(ROOT / 'data/course-data/train' / ims[a['image_id']]['file_name']).convert('L')
            x, y, w, h = a['bbox']; cx, cy = x + w / 2, y + h / 2
            x0, y0 = int(cx - C / 2), int(cy - C / 2)
            tile = im.crop((x0, y0, x0 + C, y0 + C)).resize((C * Z, C * Z), Image.Resampling.NEAREST).convert('RGB')
            d = ImageDraw.Draw(tile)
            corners(d, (x - x0) * Z - 3, (y - y0) * Z - 3, (x + w - x0) * Z + 3, (y + h - y0) * Z + 3, (0, 220, 0))
            if p is not None and v > 0:
                px, py, pw, ph = p['bbox']
                d.rectangle([(px - x0) * Z, (py - y0) * Z, (px + pw - x0) * Z, (py + ph - y0) * Z], outline=(255, 40, 40), width=1)
            col, row = k % 5, k // 5
            sheet.paste(tile, (col * (C * Z + 6), row * (C * Z + 22)))
            ImageDraw.Draw(sheet).text((col * (C * Z + 6) + 2, row * (C * Z + 22) + C * Z + 4),
                f"img {a['image_id']}  {w:.0f}x{h:.0f}px  IoU {v:.2f}" + (f"  score {p['score']:.2f}" if p is not None and v > 0 else ''), fill='black')
        sheet.save(out / f'{cname}-{g}.jpg', quality=90)
        print(cname, g, len(groups[g]), 'labels; sheet', out / f'{cname}-{g}.jpg')
