"""Crop sheet for one class: label green, our best box red; the 18 worst found boxes by IoU, then 6 typical ones.
Needs box_error_study.py run for the class first.

    python tools/analysis/worst_box_crops.py 8      # writes reports/class-review/worst_boxes_8.png
"""

import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = json.load(open(ROOT / f'results/pp-memorise-lr2/box_errors_{CLASS}.json'))
gt = json.loads((ROOT / 'results/pp-memorise-lr2/raw/instances_val_holdout.json').read_text())
files = {im['id']: im['file_name'] for im in gt['images']}
labels = {}
for a in gt['annotations']:
    if a['category_id'] == CLASS:
        labels.setdefault(a['image_id'], []).append(a['bbox'])
found = sorted([r for r in rows if 'pred' in r], key=lambda r: r['iou'])
picks = found[:18] + found[len(found) // 2: len(found) // 2 + 6]
TILE, COLS = 200, 6
sheet = Image.new('RGB', (COLS * TILE, ((len(picks) + COLS - 1) // COLS) * (TILE + 18)), 'white')
draw_sheet = ImageDraw.Draw(sheet)
for k, r in enumerate(picks):
    im = Image.open(ROOT / 'data/course-data/train' / files[r['image']]).convert('RGB')
    gx, gy, gw, gh = [b for b in labels[r['image']] if abs(b[2] - r['w']) < 1e-6 and abs(b[3] - r['h']) < 1e-6][0]
    side = max(gw, gh) * 2.2
    cx, cy = gx + gw / 2, gy + gh / 2
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    scale = TILE / side
    crop = im.crop(tuple(int(round(v)) for v in box)).resize((TILE, TILE), Image.NEAREST)
    d = ImageDraw.Draw(crop)
    def rect(b, colour):
        x0, y0 = (b[0] - box[0]) * scale, (b[1] - box[1]) * scale
        d.rectangle([x0, y0, x0 + b[2] * scale, y0 + b[3] * scale], outline=colour, width=2)
    for b in labels[r['image']]:
        rect(b, (0, 220, 0))
    rect(r['pred'], (255, 40, 40))
    x, y = (k % COLS) * TILE, (k // COLS) * (TILE + 18)
    sheet.paste(crop, (x, y))
    draw_sheet.text((x + 4, y + TILE + 2), f"img {r['image']}  IoU {r['iou']:.2f}  {r['w']:.0f}x{r['h']:.0f}px", fill='black')
out = ROOT / f'reports/class-review/worst_boxes_{CLASS}.png'
sheet.save(out)
print(out, len(picks))
