"""Package the cross-validated edge correction without rerunning the detector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quick_box_bias import apply
from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--base-zip', type=Path, default=Path('submissions/yolo26m_p2_finetune_329644/continue_1/yolo26m_p2_finetune_40ep_329644.zip'))
    parser.add_argument('--results', type=Path, default=Path('results/yolo26m_p2_finetune_329644/continue_1/one2many'))
    parser.add_argument('--audit', type=Path, default=Path('results/yolo26m_p2_finetune_329644/quick_box_bias_crossfit.json'))
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if not audit['accepted']:
        raise RuntimeError('Cross-validation did not support the correction')
    params = audit['fit_parameters']
    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    original = json.loads((args.results / 'predictions_val.json').read_text())
    corrected = apply(original, params, images)
    all_ids = list(images)
    clean_ids = clean_image_ids(gt['images'])
    # Classifications come from the same unchanged classifier as the base ZIP.
    from sarbench.metrics import classification_metrics
    import csv
    with (ROOT / 'data/course-data/train/labels.csv').open() as stream:
        truth = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    classified = {int(k): int(v['label'] if isinstance(v, dict) else v)
                  for k, v in json.loads((args.results / 'image_classification_predictions.json').read_text()).items()}
    scores = {}
    for name, ids in [('full', all_ids), ('clean', clean_ids)]:
        metrics = detection_metrics(str(gt_path), corrected, ids)
        metrics.update(classification_metrics([truth[i] for i in ids], [classified[i] for i in ids]))
        metrics['images'] = len(ids)
        metrics['combined_delta_m'] = 50*(metrics['macro_f1']/0.9331+metrics['mAP']/0.2613-2)
        scores[name] = metrics
        print('CALIBRATED_HOLDOUT', name, {k:metrics[k] for k in ['mAP','macro_f1','combined_delta_m']}, flush=True)
    result = ROOT / 'results' / args.tag / 'seed_0'
    result.mkdir(parents=True, exist_ok=False)
    write(result / 'predictions_val.json', corrected)
    for name in ['instances_val_holdout.json', 'image_classification_predictions.json']:
        shutil.copy2(args.results / name, result / name)
    full = dict(scores['full'], clean_holdout=scores['clean'])
    write(result / 'metrics.json', {'val': full, 'calibration': str(args.audit),
          'clean_scores_use_calibration_fit_images': True,
          'note': 'Final fitted scores use the calibration images; use the disjoint, twin-grouped '
                  'cross-validation results to judge the gain.'})
    package_dir = ROOT / 'submissions' / args.tag
    package_dir.mkdir(parents=True, exist_ok=False)
    package = package_dir / f'{args.tag}.zip'
    official_images = json.loads((ROOT / 'data/course-data/val/images.json').read_text())['images']
    official_map = {im['id']: im for im in official_images}
    with zipfile.ZipFile(args.base_zip) as archive:
        labels_bytes = archive.read('classification.csv')
        base_detections = json.loads(archive.read('detection.json'))
    changed = apply(base_detections, params, official_map)
    assert all(p['image_id'] in official_map and 1 <= p['category_id'] <= 9
               and p['bbox'][2] > 0 and p['bbox'][3] > 0 for p in changed)
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('classification.csv', labels_bytes)
        archive.writestr('detection.json', json.dumps(changed, separators=(',', ':')))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert archive.read('classification.csv') == labels_bytes
    write(package_dir / 'holdout.metrics.json', {'val': full,
          'clean_scores_use_calibration_fit_images': True})
    shutil.copy2(args.audit, package_dir / 'box_bias_crossfit.json')
    write(package_dir / 'candidate.json', {'base_zip': str(args.base_zip),
          'base_zip_sha256': hashlib.sha256(args.base_zip.read_bytes()).hexdigest(),
          'zip_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
          'official_score': None, 'calibration': params, 'no_ensemble': True,
          'no_test_time_augmentation': True, 'image_size': 512,
          'unchanged_classification': True, 'images': len(official_images),
          'detections': len(changed), 'crossfit_gains_mAP_percentage_points':
          [100*t['mAP_gain'] for t in audit['trials']]})
    (package_dir / 'scoring_result').mkdir()
    (package_dir / 'scoring_result/.gitkeep').touch()
    gains = ', '.join(f"{100*t['mAP_gain']:.3f}" for t in audit['trials'])
    (package_dir / 'README.md').write_text(f'''# YOLO26m-P2 with conservative box-edge correction

Submission: [{package.name}]({package.name}). Official score pending.

Uses the frozen detector from the first continuation (that original model scored
official Delta m 34.5697), with classification predictions copied from
`{args.base_zip}`. One deterministic box-edge correction follows detection.
No ensemble, extra image pass, test-time augmentation, or official labels are used.

For each class and source width, fit half the median matched-box edge residual,
clipped to one pixel at 512px. At least 12 matches at IoU >= 0.5 and confidence
>= 0.1 are required. Fit on one clean-holdout half and evaluate on the other,
then reverse; repeat with a second fixed seed. Connected twin groups stay
together. Cross-validation mAP gains in percentage points: **{gains}**.

Final offsets use all 720 clean holdout images. The saved final full/clean scores
therefore use calibration fit images and are descriptive, not independent
estimates. See `box_bias_crossfit.json` for the held-out calibration evidence.
Classifications in this ZIP are byte-for-byte unchanged from its base ZIP.
`candidate.json` records hashes, parameters, and image counts.
''')
    print('BOX_BIAS_CANDIDATE_READY', package, flush=True)


if __name__ == '__main__':
    main()
