# job_v: the second downsampling layer at stride 1 too, at inference only (1 Nov, user: "i made them stride 1 alr, is
# it i shd stride 1 one more time"). The weights don't change: layer 1's conv slides by 1 pixel instead of 2, so every
# feature map is twice as fine and the head's strides halve. The model then sees a 512-px image the way it already
# sees a 256-px one (upscaled x2), where it finds 61% of the bridges against 20% in 512-px images. Boxes can't exceed
# 15 x stride per side (DFL), so at the finer stride nothing over ~240 px fits: the finer stride is used only on images
# whose own label is a small-object class (FINE), the others keep the model as it is. Input stays 512.
#   python job_v.py patch SRC OUT                             SRC with layer 1 at stride 1 (strides halved)
#   python job_v.py export WEIGHTS holdout|test OUT [NMS CONF MAX]   boxes as make_submission.py's passes make them
#   python job_v.py verdict COARSE.json FINE.json             holdout: coarse everywhere vs routed by the two heads' labels
#   python job_v.py pack REF.zip FINE.json FINE95.json COARSE95.json VARIANT OUT.zip NMS95.json
#       REF's boxes for the unrouted images, round + hedge + tail of FINE's for the routed ones; NMS95.json = the
#       NMS-0.95 boxes for job_w (COARSE95 for unrouted images, FINE95 for routed)
import csv, io, json, subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT
from job_k import HEADS, Scorer, split, submitted

FINE = {'bp': {3, 7}, 'six': {1, 3, 4, 6, 7, 8}, 'all': set(range(1, 10))}  # category ids: bridge, playground; + aircraft,
# car, oil_tank, ship; every class. A variant ending in 512 (bp512, ...) routes only 512-px images: the 256-px ones are
# already seen at 2x, so the finer stride would show them at 4x.
TEST_IMAGES = Path('data/course-data/val/images.json')


def patch(src, out):
    import torch
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    m = ckpt.get('ema') or ckpt['model']
    conv = m.model[1].conv
    assert tuple(conv.stride) == (2, 2) and tuple(m.model[0].conv.stride) == (1, 1), (m.model[0].conv.stride, conv.stride)
    conv.stride = (1, 1)
    head = m.model[-1]
    print(f'{src}: layer 1 {type(m.model[1]).__name__} stride 2 -> 1; head strides {head.stride.tolist()} -> '
          f'{(head.stride / 2).tolist()}', flush=True)
    head.stride = head.stride / 2
    head.shape = None  # anchors are rebuilt from the (now larger) feature maps at the first forward pass
    m.stride = m.stride / 2
    ckpt.update(model=m, ema=None, optimizer=None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print('PATCHED', out, flush=True)


def export(weights, which, out, nms='0.7', conf='1e-5', max_det='1000'):
    if which == 'holdout':
        images, root = json.load(open(GT))['images'], Path('data/course-data/train')
    else:
        images, root = json.load(open(TEST_IMAGES))['images'], Path('data/course-data/val')
    with tempfile.TemporaryDirectory(prefix='job-v-') as t:
        v = Path(t) / 'data'; (v / 'train').mkdir(parents=True); (v / 'val').mkdir()
        (v / 'train/labels.csv').symlink_to(Path('data/course-data/train/labels.csv').resolve())
        for im in images:
            target = v / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((root / im['file_name']).resolve())
        (v / 'val/images.json').write_text(json.dumps({'images': images}))
        z = Path(t) / 'boxes.zip'
        subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', weights,
                        '--detection-branch', 'one2many', '--data', str(v), '--out', str(z), '--conf', conf,
                        '--max-det', max_det, '--nms-iou', nms, '--batch', '4'], check=True)
        boxes = json.loads(zipfile.ZipFile(z).read('detection.json'))
    widths = {im['id']: im['width'] for im in images}
    over = sum(p['bbox'][0] + p['bbox'][2] > widths[p['image_id']] + 1 or p['bbox'][1] + p['bbox'][3] > widths[p['image_id']] + 1
               for p in boxes)
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(boxes))
    print(f'BOXES {out}: {len(boxes)} ({sum(p["score"] >= 0.001 for p in boxes)} >= 0.001) on {len(images)} images; '
          f'{over} boxes past the image edge', flush=True)


def holdout_labels():
    ids = json.load(open(HEADS / 'branch_head.json'))['val_ids']
    probs = (np.load(HEADS / 'branch_head.holdout_probs.npy') + np.load(HEADS / 'conv_head.holdout_probs.npy')) / 2
    return {int(i): int(np.argmax(p)) + 1 for i, p in zip(ids, probs)}


def test_labels(ref_zip):
    rows = csv.DictReader(io.StringIO(zipfile.ZipFile(ref_zip).read('classification.csv').decode()))
    return {int(r['image_id']): int(r['label']) + 1 for r in rows}


def routed(labels, variant, widths):
    name, only512 = (variant[:-3], True) if variant.endswith('512') else (variant, False)
    return {i for i, c in labels.items() if c in FINE[name] and (not only512 or widths.get(i) == 512)}


def verdict(coarse_json, fine_json):
    s = Scorer()
    coarse, fine = json.load(open(coarse_json)), json.load(open(fine_json))
    labels = holdout_labels()
    variants = [('coarse everywhere', set())] + [(f'fine for {k}', routed(labels, k, s.widths))
                                                 for k in ('bp', 'six', 'all', 'bp512', 'six512', 'all512')]
    rows = {}
    for name, ids in variants:
        boxes = [p for p in coarse if p['image_id'] not in ids] + [p for p in fine if p['image_id'] in ids]
        raw, low = split(boxes)
        rows[name] = s.row(f'{name} ({len(ids)} images)', submitted(raw, low, s.widths))
    base = rows['coarse everywhere']
    best = max((k for k in rows if k != 'coarse everywhere'), key=lambda k: rows[k][0])
    gain, dn = rows[best][0] - base[0], rows[best][1] - base[1]
    print(f'VERDICT {"GO" if gain >= 0.3 and dn >= 0 else "STOP"} {best}: official-mix {gain:+.2f} (needs >= +0.30), '
          f'new6 {dn:+.2f} (needs >= 0)', flush=True)


def pack(ref_zip, fine_json, fine95_json, coarse95_json, variant, out_zip, nms95_out):
    if Path(out_zip).exists():
        raise FileExistsError(out_zip)
    widths = {im['id']: im['width'] for im in json.load(open(TEST_IMAGES))['images']}
    ids = routed(test_labels(ref_zip), variant, widths)
    with zipfile.ZipFile(ref_zip) as z:
        members = {n: z.read(n) for n in z.namelist()}
    have = json.loads(members['detection.json'])
    keep = [p for p in have if p['image_id'] not in ids]
    raw, low = split([p for p in json.load(open(fine_json)) if p['image_id'] in ids])
    new = submitted(raw, low, widths)
    members['detection.json'] = json.dumps(keep + new).encode()
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, data in members.items():
            z.writestr(n, data)
    nms95 = [p for p in json.load(open(coarse95_json)) if p['image_id'] not in ids] + \
            [p for p in json.load(open(fine95_json)) if p['image_id'] in ids]
    Path(nms95_out).write_text(json.dumps(nms95))
    print(f'ROUTED_ZIP {out_zip}: {len(ids)} of {len(widths)} images at the finer stride ({len(have) - len(keep)} boxes '
          f'-> {len(new)}, {len(raw)} >= 0.001), {len(keep)} boxes of the other images as in {ref_zip}; '
          f'classification.csv unchanged; NMS-0.95 boxes for job_w: {len(nms95)}', flush=True)


if __name__ == '__main__':
    {'patch': patch, 'export': export, 'verdict': verdict, 'pack': pack}[sys.argv[1]](*sys.argv[2:])
