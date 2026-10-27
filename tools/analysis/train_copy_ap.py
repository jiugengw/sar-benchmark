"""Memorise precision vs twin recognition: AP of a holdout-split model on the TRAINING copies of the holdout twins
(images it trained on, scored against their own labels), next to its AP on the twins themselves.

If the training copies score far above the twins, the model memorised the boxes but loses them when brightness or
sharpness change (the jitter test's target). If they score about the same, memorise itself is not pixel-precise
(a stronger or longer memorise is the fix). Analysis only; needs a GPU and twin-copies.json from
twin_label_agreement.py. Run from the repository root:

    python tools/analysis/train_copy_ap.py results/twin-copies.json runs/ft40-s1r-flipmem-338727/train/weights/best.pt \\
        runs/third-stage-331789-package/image_classifier.pt results/ft40-s1r-flipmem-338727/one2many
"""
import json, subprocess, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from sarbench.metrics import detection_metrics
from hedge_study import CHOSEN_TIERS, hedge_tiers
from round_boxes import round_predictions
pairs, weights, classifier, holdout = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
copy_of = {int(t): c for t, c in json.load(open(pairs)).items()}
train = json.load(open('data/course-data/train/instances.json'))
copies = sorted(set(copy_of.values())); keep = set(copies)
with tempfile.TemporaryDirectory(prefix='sar-train-copies-') as tmp:
    tmp = Path(tmp)
    gt = {'images': [im for im in train['images'] if im['id'] in keep], 'categories': train['categories'],
          'annotations': [a for a in train['annotations'] if a['image_id'] in keep]}
    (tmp / 'gt.json').write_text(json.dumps(gt))
    view = tmp / 'data'; (view / 'train').mkdir(parents=True); (view / 'val').mkdir()
    (view / 'train/labels.csv').symlink_to(Path('data/course-data/train/labels.csv').resolve())
    for im in gt['images']:
        target = view / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path('data/course-data/train', im['file_name']).resolve())
    (view / 'val/images.json').write_text(json.dumps({'images': gt['images']}))
    subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', weights, '--classifier-checkpoint',
                    classifier, '--detection-branch', 'one2many', '--data', view, '--out', tmp / 'copies.zip',
                    '--batch', '16', '--nms-iou', '0.7'], check=True, stdout=subprocess.DEVNULL)
    predictions = json.loads(zipfile.ZipFile(tmp / 'copies.zip').read('detection.json'))
    widths = {im['id']: im['width'] for im in gt['images']}
    on_copies = detection_metrics(str(tmp / 'gt.json'), hedge_tiers(round_predictions(predictions, widths, 'whole-pixel-classes'),
                                                               CHOSEN_TIERS), copies)
hg = json.load(open(holdout / 'instances_val_holdout.json'))
widths = {im['id']: im['width'] for im in hg['images']}
on_twins = detection_metrics(str(holdout / 'instances_val_holdout.json'), hedge_tiers(round_predictions(
    json.load(open(holdout / 'predictions_val.json')), widths, 'whole-pixel-classes'), CHOSEN_TIERS), sorted(copy_of))
names = {c['id']: c['name'] for c in train['categories']}
for label, r in (('training copies (trained on)', on_copies), ('twins (holdout)', on_twins)):
    print(f"{label:30s} mAP {100 * r['mAP']:5.1f}  AP50 {100 * r['AP50']:5.1f}  AP75 {100 * r['AP75']:5.1f}  | "
          + ', '.join(f'{names.get(k, k)} {100 * v:.1f}' for k, v in r['per_class_AP'].items()), flush=True)
