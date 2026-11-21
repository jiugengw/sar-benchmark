# Label-transfer probe on the leak-free test (21 Nov; NOT ALLOWED, a record only: docs/NOT_ALLOWED_TRICKS.md). Runs the
# 30 Oct full-size matcher (sbatch/staging/transfer_full.py) on the new split: each new test image's best match among
# the new train images (as is and in the three flips), then the matched image's boxes of the static classes (airport,
# wind turbine, playground, bridge) replace the model's at score 1.0, for the two ranges of the 30 Oct probes.
#   python probe_transfer.py IN.zip OUTDIR        (run from the leak-free workspace, next to transfer.py / transfer_full.py)
import json, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import STATIC, train, transfer  # noqa: E402
from transfer_full import match_full  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
TE = Path('data/course-data/test'); test = json.load(open(TE / 'images.json'))['images']; images = {im['id']: im for im in test}
best = match_full(test, TE, train['images'])
sims = np.array([best.get(im['id'], (-1,))[0] for im in test])
print('best full-size similarity, quantiles 10..90%:', np.round(np.quantile(sims, np.linspace(0.1, 0.9, 9)), 3).tolist())
print('share at or above 0.45/0.6/0.7/0.8/0.9/0.98:', [round(float(np.mean(sims >= t)), 4) for t in (0.45, 0.6, 0.7, 0.8, 0.9, 0.98)])
(out / 'matches.json').write_text(json.dumps({str(i): v for i, v in best.items()}))
z = zipfile.ZipFile(src); members = {n: z.read(n) for n in z.namelist()}; preds = json.loads(members['detection.json'])
for name, lo, hi in (('all', 0.45, 1.01), ('near', 0.45, 0.98)):
    boxes, nimg, nbox = transfer(preds, best, images, lo, hi, STATIC, 'replace')
    dst = out / f'transferfull-{name}-not-allowed.zip'
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
        for n, d in members.items():
            o.writestr(n, json.dumps(boxes).encode() if n == 'detection.json' else d)
    print(f'{name}: {nimg} test images matched at {lo}-{min(hi, 1.0)}, {nbox} boxes copied -> {dst}', flush=True)
