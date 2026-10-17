"""Build the local holdout comparison viewer: reports/holdout-viewer/index.html plus its data/ folder.

Every saved run under results/ that has instances_val_holdout.json and predictions_val.json scores the same
939-image labeled holdout. This script precomputes compact per-image data for each run, so the page can compare up
to three runs on all 939 images without loading the raw COCO files:

    python tools/build_holdout_viewer.py                      # every run tracked in git
    python tools/build_holdout_viewer.py --runs results/yolo26m_p2_candidate_330230/seed_0 results/yolo26m-100-p2-328339/seed_0

Output (regenerated, ignored by git):
    reports/holdout-viewer/data/index.js         images, ground truth, run list with overall scores
    reports/holdout-viewer/data/runs/<slug>.js   per-run detections (score >= 0.05, top 100), image classes, stats

The data files are plain scripts that assign to window.*, so index.html works when opened straight from disk
(file://), where browsers block fetch() of local JSON. Images are referenced from data/course-data/train/images/.

Per-image detection stats use IoU >= 0.5 and score >= 0.3: detections are taken in descending score order and each is
matched to the unmatched ground-truth box of the same class with the highest IoU, as in COCO. A detection of another
class, or one with IoU < 0.5, is a false positive. The per-image AP50 uses every stored detection with the same
matching and COCO's 101-point interpolated precision.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_SCORE = 0.05  # detections below this score are not stored
MAX_DETS = 100  # per image, highest scores first (COCO's maxDets)
IOU = 0.5
STAT_SCORE = 0.3  # score threshold of the precomputed per-image stats
# Official ranking: delta_m = 50 * [(F1 / 0.9331 - 1) + (mAP / 0.2613 - 1)], against the TA's ViT + LoRA reference.
REF_F1, REF_MAP = 0.9331, 0.2613
BEST_RUN = 'yolo26m_p2_candidate_330230/seed_0'
BASELINE_RUN = 'yolo26m-100-p2-328339/seed_0'  # E5, the detector the best model continues from


# ---------------------------------------------------------------- per-image matching (mirrored in index.html)

def iou_xywh(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_image(gt, dets, score_thr=STAT_SCORE, iou_thr=IOU):
    """Greedy COCO-style matching on one image.

    gt: [(category_id, [x, y, w, h]), ...]; dets: [(category_id, [x, y, w, h], score), ...].
    Detections with score >= score_thr are taken in descending score order; each matches the unmatched ground-truth
    box of its class with the highest IoU, if that IoU >= iou_thr. Returns dict with tp, fp, fn, recall, mean_iou
    (None without matches), det_match (per input detection: index of the matched gt box, -1 for a false positive,
    None when below score_thr) and gt_match (per gt box: index of the matching detection or -1)."""
    order = sorted((i for i, d in enumerate(dets) if d[2] >= score_thr), key=lambda i: -dets[i][2])
    gt_match = [-1] * len(gt)
    det_match = [None] * len(dets)
    ious = []
    for i in order:
        cat, box, _ = dets[i]
        best, best_iou = -1, iou_thr
        for g, (gcat, gbox) in enumerate(gt):
            if gt_match[g] >= 0 or gcat != cat:
                continue
            v = iou_xywh(box, gbox)
            if v >= best_iou:
                best, best_iou = g, v
        det_match[i] = best
        if best >= 0:
            gt_match[best] = i
            ious.append(best_iou)
    tp = len(ious)
    fp = len(order) - tp
    fn = len(gt) - tp
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'recall': tp / len(gt) if gt else None,
        'mean_iou': sum(ious) / tp if tp else None,
        'det_match': det_match, 'gt_match': gt_match,
    }


def image_ap50(gt, dets, iou_thr=IOU):
    """AP at IoU 0.5 of one image's detections, over all classes pooled (a wrong-class detection is a false
    positive), with COCO's 101-point interpolation. None for an image without ground truth."""
    if not gt:
        return None
    m = match_image(gt, dets, score_thr=-1.0, iou_thr=iou_thr)
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    tp = fp = 0
    recalls, precisions = [], []
    for i in order:
        if m['det_match'][i] >= 0:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / len(gt))
        precisions.append(tp / (tp + fp))
    for k in range(len(precisions) - 2, -1, -1):  # precision envelope
        precisions[k] = max(precisions[k], precisions[k + 1])
    total, j = 0.0, 0
    for r in (k / 100 for k in range(101)):
        while j < len(recalls) and recalls[j] < r:
            j += 1
        total += precisions[j] if j < len(recalls) else 0.0
    return total / 101


def delta_m(f1, mAP):
    if f1 is None or mAP is None:
        return None
    return 50 * ((f1 / REF_F1 - 1) + (mAP / REF_MAP - 1))


# ---------------------------------------------------------------- run discovery and labels

def discover_runs():
    """Run folders (relative to the repo) with a tracked holdout ground truth and predictions next to it."""
    out = subprocess.check_output(['git', 'ls-files', 'results/**/instances_val_holdout.json'], cwd=ROOT, text=True)
    runs = []
    for line in out.split():
        folder = Path(line).parent
        if (ROOT / folder / 'predictions_val.json').exists():
            runs.append(folder.as_posix())
    return sorted(runs)


def rel_name(folder):
    """Run path relative to results/, e.g. 'yolo26m_p2_candidate_330230/seed_0'."""
    parts = Path(folder).parts
    return '/'.join(parts[1:] if parts and parts[0] == 'results' else parts)


def slug(folder):
    return re.sub(r'[^A-Za-z0-9._-]+', '--', rel_name(folder)).strip('-')


def family(name):
    n = name.lower()
    if 'denodet' in n:
        return 'DenoDet V2'
    if 'yolo' in n or 'rtdetr' in n:
        return 'YOLO / RT-DETR'
    if 'resnet' in n:
        return 'ResNet34 + Faster R-CNN'
    if 'vit' in n:
        return 'ViT + Faster R-CNN'
    return 'TerraMind + Faster R-CNN'


LABELS = {
    BEST_RUN: 'YOLO26m-P2 final candidate (best, 330230)',
    BASELINE_RUN: 'E5 · YOLO26m-P2 100 ep (328339)',
    'yolo26m_p2_box_bias_20261017/seed_0': 'YOLO26m-P2 continuation + box-edge bias',
    'yolo26m_p2_quick_head_330230/one2many': 'YOLO26m-P2 continuation + 1024 head (330230)',
    'E7-rtdetr-l-p2-328598/seed_0': 'E7 · RT-DETR-L P2 (328598)',
    'yolo26s-p2-E2-328622/seed_0': 'E2 · YOLO26s-P2 (328622)',
    'yolo26s-p2-E3-328622/seed_0': 'E3 · YOLO26s-P2 (328622)',
    'phase3-yolo26m-joint-20261017-1516/seed_0': 'Phase 3 · YOLO26m joint',
    'phase3-yolo26m-joint-20261017-1516/submission-pipeline': 'Phase 3 · YOLO26m joint (submission pipeline)',
    'denodet-v2-329477/seed_0': 'DenoDet V2 (329477)',
    'denodet-v2-329477-epoch32/seed_0': 'DenoDet V2 epoch 32 (329477)',
}
# Official Codabench delta_m (submissions/README.md) of runs whose uploaded ZIP is unambiguously this model.
OFFICIAL_DM = {
    BEST_RUN: 36.80,
    BASELINE_RUN: 23.18,
    'E7-rtdetr-l-p2-328598/seed_0': 22.85,
    'phase3-yolo26m-joint-20261017-1516/submission-pipeline': 23.80,
    'yolo11s-100-p2-328058/seed_0': 18.11,
}
MODEL_NAMES = {'terramind': 'TerraMind', 'vit': 'ViT', 'resnet34': 'ResNet34'}
ADAPT_NAMES = {'moelora': 'MoE-LoRA', 'lora': 'LoRA', 'full': 'full'}


def label(folder):
    name = rel_name(folder)
    if name in LABELS:
        return LABELS[name]
    parts = [p for p in name.split('/') if p != 'seed_0']
    m = re.fullmatch(r'yolo(\d+)([nslmx])-(?:(\d+)-)?(p2|standard|baseline)-(\d+)', parts[0])
    if m:
        gen, size, epochs, kind, job = m.groups()
        text = f'YOLO{gen}{size}' + ('-P2' if kind == 'p2' else '')
        text += f' {epochs} ep' if epochs else ''
        text += ' baseline' if kind == 'baseline' else ' standard' if kind == 'standard' else ''
        return f'{text} ({job})'
    m = re.fullmatch(r'yolo26m_p2_finetune_(\d+)', parts[0])
    if m and len(parts) == 3:
        stage = {'continue_1': 'continuation 1', 'continue_2': 'continuation 2',
                 'e5_spatial_head': 'E5 + spatial head'}.get(parts[1], parts[1])
        return f'YOLO26m-P2 {stage} · {parts[2]} ({m.group(1)})'
    # Faster R-CNN family: <experiment>/<backbone>_<init>_<adapt>[_extra] or <backbone>_<init>_<adapt>[_extra]
    model_part = next((p for p in parts if re.match(r'(terramind|vit|resnet34)_pretrained_', p)), parts[-1])
    m = re.match(r'(terramind|vit|resnet34)_pretrained_(moelora|lora|full)_?(.*)', model_part)
    model = f'{MODEL_NAMES[m.group(1)]} {ADAPT_NAMES[m.group(2)]}' if m else model_part
    extra = [m.group(3)] if m and m.group(3) else []
    extra += [p for p in parts if p != model_part]
    extra = [re.sub(r'_seed0$', '', e).replace('_', ' ') for e in extra]
    return f'{model} · {" / ".join(extra)}' if extra else model


# ---------------------------------------------------------------- loading

def load_ground_truth(gt_path, labels_csv):
    gt = json.loads(Path(gt_path).read_text())
    from sarbench.twins import clean_image_ids
    clean = set(clean_image_ids(gt['images']))
    true_label = {}
    if Path(labels_csv).exists():
        with open(labels_csv) as f:
            true_label = {row['file_name']: int(row['label']) for row in csv.DictReader(f)}
    boxes = {}
    for a in gt['annotations']:
        boxes.setdefault(a['image_id'], []).append((a['category_id'], [float(v) for v in a['bbox']]))
    images = []
    for im in sorted(gt['images'], key=lambda im: im['id']):
        b = boxes.get(im['id'], [])
        label_ = true_label.get(im['file_name'])
        if label_ is None and b:
            label_ = b[0][0] - 1
        images.append({'id': im['id'], 'file': im['file_name'], 'w': im['width'], 'h': im['height'],
                       'label': label_, 'clean': im['id'] in clean, 'gt': b})
    categories = [c['name'] for c in sorted(gt['categories'], key=lambda c: c['id'])]
    return gt, images, categories


def load_classifications(path):
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    out = {}
    for key, value in raw.items():
        out[int(key)] = int(value['label'] if isinstance(value, dict) else value)
    return out


def reported_f1(metrics_path):
    """(full, clean) macro-F1 written by the training script, for runs without per-image classifications."""
    if not metrics_path.exists():
        return None, None
    m = json.loads(metrics_path.read_text())
    full = clean = None
    for key in ('val', 'final_val'):
        if isinstance(m.get(key), dict) and 'macro_f1' in m[key]:
            full = m[key]['macro_f1']
            ch = m[key].get('clean_holdout')
            if isinstance(ch, dict):
                clean = ch.get('macro_f1')
            break
    if full is None and isinstance(m.get('classification'), dict):
        full = m['classification'].get('macro_f1')
        ch = m.get('clean_holdout', {})
        clean = ch.get('classification', {}).get('macro_f1') if isinstance(ch, dict) else None
    return full, clean


def r(v, nd=4):
    return None if v is None else round(float(v), nd)


# ---------------------------------------------------------------- one run

def build_run(folder, images, categories, out_dir):
    from sarbench.metrics import classification_metrics, detection_metrics
    t0 = time.time()
    run_dir = ROOT / folder
    gt_path = run_dir / 'instances_val_holdout.json'
    predictions = json.loads((run_dir / 'predictions_val.json').read_text())
    image_ids = [im['id'] for im in images]
    clean_ids = [im['id'] for im in images if im['clean']]

    full = detection_metrics(str(gt_path), predictions, image_ids)
    clean = detection_metrics(str(gt_path), predictions, clean_ids)

    per_image = {}
    for p in predictions:
        if p['score'] >= MIN_SCORE:
            per_image.setdefault(p['image_id'], []).append(
                (int(p['category_id']), [float(v) for v in p['bbox']], float(p['score'])))

    classes = load_classifications(run_dir / 'image_classification_predictions.json')
    f1_source = 'per-image' if classes else 'metrics.json'
    f1_full = f1_clean = None
    if classes:
        covered = [im for im in images if im['id'] in classes and im['label'] is not None]
        if len(covered) == len(images):
            f1_full = classification_metrics([im['label'] for im in covered], [classes[im['id']] for im in covered],
                                             len(categories))['macro_f1']
            cl = [im for im in covered if im['clean']]
            f1_clean = classification_metrics([im['label'] for im in cl], [classes[im['id']] for im in cl],
                                              len(categories))['macro_f1']
        else:
            f1_source = f'per-image ({len(covered)}/{len(images)} images)'
    if f1_full is None:
        f1_full, f1_clean = reported_f1(run_dir / 'metrics.json')
        if f1_full is None:
            f1_source = None

    dets_out, stats_out, cls_out = [], [], []
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'cls_wrong': 0}
    for im in images:
        # Stats are computed on the rounded values the page receives, so the page reproduces them exactly.
        dets = [(cat, [round(v, 1) for v in box], round(s * 1000) / 1000)
                for cat, box, s in sorted(per_image.get(im['id'], []), key=lambda d: -d[2])[:MAX_DETS]]
        gt_boxes = [(cat, [round(v, 1) for v in box]) for cat, box in im['gt']]
        m = match_image(gt_boxes, dets)
        ap = image_ap50(gt_boxes, dets)
        flat = []
        for cat, box, s in dets:
            flat += box + [cat, round(s * 1000)]
        dets_out.append(flat)
        stats_out.append([m['tp'], m['fp'], m['fn'],
                          -1 if m['mean_iou'] is None else round(m['mean_iou'] * 1000),
                          -1 if ap is None else round(ap * 1000)])
        for k in ('tp', 'fp', 'fn'):
            totals[k] += m[k]
        if classes is not None:
            c = classes.get(im['id'], -1)
            cls_out.append(c)
            totals['cls_wrong'] += int(c != im['label'])
    name = rel_name(folder)
    s = slug(folder)
    payload = {'slug': s, 'cls': cls_out if classes is not None else None, 'dets': dets_out, 'stats': stats_out}
    path = out_dir / 'runs' / f'{s}.js'
    path.write_text('window.HOLDOUT_RUNS=window.HOLDOUT_RUNS||{};window.HOLDOUT_RUNS[' + json.dumps(s) + ']='
                    + json.dumps(payload, separators=(',', ':')) + ';\n')
    summary = {
        'slug': s, 'path': folder, 'name': name, 'label': label(folder), 'family': family(name),
        'best': name == BEST_RUN, 'baseline': name == BASELINE_RUN, 'official_dm': OFFICIAL_DM.get(name),
        'has_cls': classes is not None, 'f1_source': f1_source,
        'full': {'mAP': r(full['mAP']), 'AP50': r(full['AP50']), 'AP_small': r(full['AP_small']),
                 'F1': r(f1_full), 'dm': r(delta_m(f1_full, full['mAP']), 2)},
        'clean': {'mAP': r(clean['mAP']), 'AP50': r(clean['AP50']), 'AP_small': r(clean['AP_small']),
                  'F1': r(f1_clean), 'dm': r(delta_m(f1_clean, clean['mAP']), 2)},
        'at_0.3': totals,
        'stored_dets': sum(len(d) // 6 for d in dets_out),
        'bytes': path.stat().st_size,
    }
    print(f'  {name}: mAP {full["mAP"]:.4f} / clean {clean["mAP"]:.4f}, F1 {f1_full if f1_full is None else round(f1_full, 4)}'
          f' [{f1_source}], {summary["stored_dets"]} dets, {summary["bytes"] / 1e6:.2f} MB, {time.time() - t0:.0f}s',
          flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', help='run folders (default: every results/** folder tracked in git with '
                                              'instances_val_holdout.json and predictions_val.json)')
    ap.add_argument('--out', default='reports/holdout-viewer')
    ap.add_argument('--data', default='data/course-data', help='course data root (labels.csv, images)')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    runs = [Path(p).as_posix().rstrip('/') for p in args.runs] if args.runs else discover_runs()
    runs = [p[len(str(ROOT)) + 1:] if p.startswith(str(ROOT) + '/') else p for p in runs]
    usable, skipped = [], []
    for folder in runs:
        missing = [f for f in ('instances_val_holdout.json', 'predictions_val.json') if not (ROOT / folder / f).exists()]
        (skipped.append((folder, 'missing ' + ', '.join(missing))) if missing else usable.append(folder))
    if not usable:
        sys.exit('no usable runs')

    out_dir = (ROOT / args.out).resolve()
    (out_dir / 'data' / 'runs').mkdir(parents=True, exist_ok=True)
    gt, images, categories = load_ground_truth(ROOT / usable[0] / 'instances_val_holdout.json',
                                               ROOT / args.data / 'train' / 'labels.csv')
    reference = {im['id'] for im in images}
    for folder in list(usable):  # every compared run must score the same holdout images
        ids = {im['id'] for im in json.loads((ROOT / folder / 'instances_val_holdout.json').read_text())['images']}
        if ids != reference:
            usable.remove(folder)
            skipped.append((folder, f'different holdout ({len(ids)} images)'))
    print(f'{len(images)} holdout images ({sum(im["clean"] for im in images)} clean), {len(usable)} runs')

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_run, folder, images, categories, out_dir / 'data') for folder in usable]
        summaries = [f.result() for f in futures]

    order = {'YOLO / RT-DETR': 0, 'DenoDet V2': 1, 'TerraMind + Faster R-CNN': 2, 'ViT + Faster R-CNN': 3,
             'ResNet34 + Faster R-CNN': 4}
    summaries.sort(key=lambda s: (not s['best'], order.get(s['family'], 9), -(s['full']['mAP'] or 0)))
    image_root = os.path.relpath(ROOT / args.data / 'train', out_dir).replace(os.sep, '/')
    index = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'image_root': image_root,
        'classes': categories,
        'iou': IOU, 'min_score': MIN_SCORE, 'max_dets': MAX_DETS, 'stat_score': STAT_SCORE,
        'ref': {'F1': REF_F1, 'mAP': REF_MAP},
        'defaults': [s['slug'] for s in summaries if s['best']] + [s['slug'] for s in summaries if s['baseline']],
        # image: [id, file, width, height, class index, clean (1/0), [[x, y, w, h, category_id], ...]]
        'images': [[im['id'], im['file'], im['w'], im['h'], im['label'], int(im['clean']),
                    [[round(v, 1) for v in box] + [cat] for cat, box in im['gt']]] for im in images],
        'runs': summaries,
        'skipped': [{'path': p, 'reason': why} for p, why in skipped],
    }
    (out_dir / 'data' / 'index.js').write_text('window.HOLDOUT=' + json.dumps(index, separators=(',', ':')) + ';\n')
    # remove stale run files from earlier builds with other --runs
    keep = {f'{s["slug"]}.js' for s in summaries}
    if not args.runs:
        for f in (out_dir / 'data' / 'runs').glob('*.js'):
            if f.name not in keep:
                f.unlink()
    total = sum(f.stat().st_size for f in (out_dir / 'data').rglob('*.js'))
    print(f'wrote {out_dir / "data"} ({total / 1e6:.1f} MB); open {out_dir / "index.html"}')
    for p, why in skipped:
        print(f'skipped {p}: {why}')


if __name__ == '__main__':
    main()
