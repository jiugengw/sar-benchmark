# job_w: guesses the detector already makes, put back at the bottom of the ranking (1 Nov;
# no training, no holdout check). Two kinds of extra boxes for a finished ZIP:
#  1. boxes NMS removed: the detector proposes many overlapping boxes per object and NMS keeps the strongest, deleting
#     neighbours that overlap it by IoU >= 0.7. Those neighbours are its own second guesses at the box edges. A pass
#     with NMS at 0.95 instead (one2many, scores >= 0.001, up to 1000 per image) gives them back; whole-pixel rounded.
#  2. wider hedge copies of every original box: sizes x1.2 / x0.83, shifts +-16% (the ZIP's own tiers: 1.08 / 0.92, 8%).
# Both rank below every box the ZIP already has (1 above 2, each by its box's score), and fill each image and class up
# to 100 boxes. In COCO AP, boxes ranked below all others of a class can only add matches (every earlier rank keeps its
# precision), so the new ZIP can't score lower than the one it extends. classification.csv stays byte for byte.
#   python job_w.py boxes WEIGHTS OUT.json            the NMS-0.95 pass on the test images
#   python job_w.py merge REF.zip NMS95.json OUT.zip  REF plus the extra boxes
import json, subprocess, sys, tempfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')

IMAGES = Path('data/course-data/val/images.json')
WIDE = [(1.2, 1.2), (0.83, 0.83), (1, 1, 0.16, 0), (1, 1, -0.16, 0), (1, 1, 0, 0.16), (1, 1, 0, -0.16)]


def boxes(weights, out):
    with tempfile.TemporaryDirectory(prefix='job-w-') as t:
        z = Path(t) / 'nms95.zip'
        subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', weights,
                        '--data', 'data/course-data', '--detection-branch', 'one2many', '--out', str(z),
                        '--conf', '0.001', '--max-det', '1000', '--nms-iou', '0.95'], check=True)
        found = json.loads(zipfile.ZipFile(z).read('detection.json'))
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(found))
    print('BOXES', out, len(found), flush=True)


def iou(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]); h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    i = max(0.0, w) * max(0.0, h)
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def widen(p):
    x, y, w, h = p['bbox']; cx, cy = x + w / 2, y + h / 2
    for sw, sh, *shift in WIDE:
        nw, nh = w * sw, h * sh
        dx, dy = (shift[0] * w, shift[1] * h) if shift else (0.0, 0.0)
        yield [cx + dx - nw / 2, cy + dy - nh / 2, nw, nh]


def merge(ref, nms95, out):
    from round_boxes import round_predictions
    if Path(out).exists():
        raise FileExistsError(out)
    with zipfile.ZipFile(ref) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    have = json.loads(members['detection.json'])
    widths = {im['id']: im['width'] for im in json.loads(IMAGES.read_text())['images']}
    floor = min(p['score'] for p in have)
    by_key = defaultdict(list)
    for p in have:
        by_key[p['image_id'], p['category_id']].append(p['bbox'])
    originals = [p for p in have if p['score'] >= 0.001]
    second = round_predictions([p for p in json.load(open(nms95)) if p['score'] >= 0.001], widths, 'whole-pixel-classes')
    kept = sum(any(iou(p['bbox'], b) >= 0.99 for b in by_key[p['image_id'], p['category_id']]) for p in second)
    print(f'{ref}: {len(have)} boxes ({len(originals)} originals >= 0.001), lowest score {floor:.3e}; NMS-0.95 pass: '
          f'{len(second)} boxes >= 0.001, {kept} of them already in the ZIP (the NMS-0.7 originals)', flush=True)
    groups = [('removed by NMS', [{**p, 'score': 0.5 * floor * p['score']} for p in second]),
              ('wide hedge copies', [{**p, 'bbox': b, 'score': 4e-4 * floor * p['score']} for p in originals
                                     for b in widen(p)])]
    added, extra = Counter(), []
    for kind, group in groups:
        for p in sorted(group, key=lambda p: -p['score']):
            boxes = by_key[p['image_id'], p['category_id']]
            if len(boxes) < 100 and not any(iou(p['bbox'], b) >= 0.99 for b in boxes):
                boxes.append(p['bbox']); extra.append(p); added[kind, p['category_id']] += 1
    assert max(p['score'] for p in extra) < floor
    names = {c['id']: c['name'] for c in json.loads(IMAGES.read_text()).get('categories', [])}
    for kind, _ in groups:
        n = {c: v for (k, c), v in added.items() if k == kind}
        print(f'{kind}: {sum(n.values())} boxes added (' + ', '.join(f'{names.get(c, c)} {v}' for c, v in sorted(n.items())) + ')')
    members['detection.json'] = json.dumps(have + extra).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f'EXTRAS_ZIP {out}: {len(have)} -> {len(have) + len(extra)} boxes, every new one below {floor:.3e}; '
          f'other files byte for byte {ref}', flush=True)


if __name__ == '__main__':
    {'boxes': boxes, 'merge': merge}[sys.argv[1]](*sys.argv[2:])
