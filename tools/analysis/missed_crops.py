"""Random missed labels (no same-class box with IoU >= 0.5 at any score) on the clean holdout: label green,
our same-class boxes red (score >= 0.05), other-class boxes blue. 8 per class, one row each.

    python tools/analysis/missed_crops.py 5 6 8      # harbor, oil tank, ship -> reports/class-review/missed_crops.png
"""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models/yolo")); sys.path.insert(0, str(ROOT))
from postprocess_study import apply
from sarbench.twins import clean_image_ids
raw = ROOT / "results/pp-memorise-lr2/raw"
gt = json.loads((raw / "instances_val_holdout.json").read_text())
clean = set(clean_image_ids(gt["images"]))
files = {im["id"]: im["file_name"] for im in gt["images"]}
names = {c["id"]: c["name"] for c in gt["categories"]}
preds = apply(json.loads((raw / "predictions_val.json").read_text()), "nms", 0.7)
by_image = {}
for p in preds:
    by_image.setdefault(p["image_id"], []).append(p)
def iou(a, b):
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = w * h; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
TILE, PER = 170, 8
classes = [int(c) for c in sys.argv[1:]] or [5, 6, 8]
sheet = Image.new("RGB", (PER * TILE, len(classes) * (TILE + 16)), "white")
ds = ImageDraw.Draw(sheet)
for row, cls in enumerate(classes):
    missed = []
    for a in gt["annotations"]:
        if a["category_id"] != cls or a["image_id"] not in clean:
            continue
        same = [p for p in by_image.get(a["image_id"], []) if p["category_id"] == cls]
        best = max((iou(a["bbox"], p["bbox"]) for p in same), default=0)
        if best < 0.5:
            missed.append((a, best))
    random.Random(0).shuffle(missed)
    print(names[cls], "missed", len(missed))
    for k, (a, best) in enumerate(missed[:PER]):
        im = Image.open(ROOT / "data/course-data/train" / files[a["image_id"]]).convert("RGB")
        x, y, w, h = a["bbox"]; side = max(w, h) * 3 + 16; cx, cy = x + w / 2, y + h / 2
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2); s = TILE / side
        crop = im.crop(tuple(int(round(v)) for v in box)).resize((TILE, TILE), Image.NEAREST); d = ImageDraw.Draw(crop)
        def rect(b, colour):
            d.rectangle([(b[0] - box[0]) * s, (b[1] - box[1]) * s, (b[0] + b[2] - box[0]) * s, (b[1] + b[3] - box[1]) * s], outline=colour, width=2)
        for p in by_image.get(a["image_id"], []):
            if p["score"] >= 0.05:
                rect(p["bbox"], (255, 40, 40) if p["category_id"] == cls else (60, 120, 255))
        rect(a["bbox"], (0, 220, 0))
        sheet.paste(crop, (k * TILE, row * (TILE + 16)))
        ds.text((k * TILE + 3, row * (TILE + 16) + TILE + 2), f"{names[cls]} {a['image_id']} {w:.0f}x{h:.0f} IoU {best:.2f}", fill="black")
out = ROOT / "reports/class-review/missed_crops.png"; sheet.save(out); print(out)
