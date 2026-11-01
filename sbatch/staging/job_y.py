# job_y: does job_x's retrained detection head help, and the ZIP if it does (1 Nov, user: "think of how u can modify
# the detector slightly without retraining it all", then "Start it").
#  head SRC TRAINED OUT    SRC with TRAINED's detection head (the last layer, both branches) and nothing else: backbone
#                          and neck stay SRC's bit for bit, so the two-head classifier's labels can't move. Prints how
#                          far TRAINED's frozen layers drifted anyway (BatchNorm statistics) and how much the head moved.
#  export WEIGHTS OUT      holdout boxes as make_submission.py --final's passes 1 and 4 make them (one2many, down to
#                          1e-5, 1000 per image; the boxes >= 0.001 are pass 1's).
#  verdict BASE NEW        both scored as submitted (round + hedge + tail). GO only if official-mix gains >= 0.3 and
#                          new6 doesn't drop (the plan agreed on 1 Nov); otherwise +65.80 stays.
#   python job_y.py head SRC TRAINED OUT | export WEIGHTS OUT | verdict BASE NEW
import json, subprocess, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT
from job_k import Scorer, split, submitted


def head(src, trained, out):
    import torch
    S, T = (torch.load(p, map_location='cpu', weights_only=False) for p in (src, trained))
    ms, mt = (c.get('ema') or c['model'] for c in (S, T))  # the model YOLO() loads (last.pt mid-run keeps it in 'ema')
    sd, td = ms.state_dict(), mt.state_dict()
    assert sd.keys() == td.keys(), set(sd) ^ set(td)
    last = max(int(k.split('.')[1]) for k in sd)
    keys = [k for k in sd if k.startswith(f'model.{last}.')]
    drift = max(((sd[k].float() - td[k].float()).abs().max().item(), k) for k in sd
                if k not in keys and sd[k].is_floating_point())
    moved = sum((sd[k].float() - td[k].float()).abs().sum().item() for k in keys) / sum(
        sd[k].float().abs().sum().item() for k in keys if sd[k].is_floating_point())
    print(f'{trained}: head = layer {last} ({len(keys)} tensors, mean change {100 * moved:.1f}% of |w|); largest '
          f'change elsewhere {drift[0]:.2e} ({drift[1]}), not taken', flush=True)
    ms.load_state_dict({**sd, **{k: td[k].to(sd[k].dtype) for k in keys}})
    S.update(model=ms, ema=None, optimizer=None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(S, out)
    check = (lambda c: c.get('ema') or c['model'])(torch.load(out, map_location='cpu', weights_only=False)).state_dict()
    src_sd = (lambda c: c.get('ema') or c['model'])(torch.load(src, map_location='cpu', weights_only=False)).state_dict()
    same = all(torch.equal(check[k], src_sd[k]) for k in check if k not in keys)
    new = all(torch.equal(check[k], td[k].to(check[k].dtype)) for k in keys)
    print(f'HEAD_SAVED {out}: other layers identical to {src}: {same}; head identical to {trained}: {new}', flush=True)
    assert same and new


def export(weights, out):
    gt = json.load(open(GT))
    with tempfile.TemporaryDirectory(prefix='job-y-') as t:
        view = Path(t) / 'data'; (view / 'train').mkdir(parents=True); (view / 'val').mkdir()
        (view / 'train/labels.csv').symlink_to(Path('data/course-data/train/labels.csv').resolve())
        for im in gt['images']:
            target = view / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Path('data/course-data/train', im['file_name']).resolve())
        (view / 'val/images.json').write_text(json.dumps({'images': gt['images']}))
        z = Path(t) / 'boxes.zip'
        subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', weights,
                        '--detection-branch', 'one2many', '--data', str(view), '--out', str(z),
                        '--conf', '1e-5', '--max-det', '1000'], check=True)
        boxes = json.loads(zipfile.ZipFile(z).read('detection.json'))
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(boxes))
    print('BOXES', out, len(boxes), flush=True)


def verdict(base, new):
    s = Scorer()
    scores = []
    for name, path in (('as is', base), ('new head', new)):
        raw, low = split(json.load(open(path)))
        s.row(f'{name}: round + hedge', submitted(raw, [], s.widths))
        scores.append(s.row(f'{name}: round + hedge + tail', submitted(raw, low, s.widths)))
    (m0, n0), (m1, n1) = scores
    go = m1 - m0 >= 0.3 and n1 >= n0
    print(f'VERDICT {"GO" if go else "STOP"}: official-mix {m1 - m0:+.2f} (needs >= +0.30), new6 {n1 - n0:+.2f} '
          f'(needs >= 0)', flush=True)


if __name__ == '__main__':
    {'head': head, 'export': export, 'verdict': verdict}[sys.argv[1]](*sys.argv[2:])
