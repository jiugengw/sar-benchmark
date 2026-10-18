"""Predict the 939 holdout images with a fine-tuned Dome-DETR and score them on the full and clean holdout.

Each image is placed once on a 512 x 512 canvas without upscaling (shrunk only if larger, otherwise padded) and passed
once through the model (no test-time augmentation), the way a submission would run it. Boxes are mapped back to original pixels and to our category ids, then written in the same
layout as the YOLO runs (results/<tag>/predictions_val.json, instances_val_holdout.json) for sarbench.twins.score.

Dome-DETR has no image classifier. For a rough F1, each image is given the class of its highest-scoring box; judge
Dome on detection mAP, and pair a winner with an image classifier afterwards.

    python models/dome/export_predictions.py --tag dome-m-test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.dome.common import DATA, ROOT, SLOT_TO_CATEGORY, use_dome  # noqa: E402
from models.dome.transforms import fit_image  # noqa: E402

KEEP = 300  # boxes kept per image, highest scores first


def best_weights(train: Path) -> Path:
    """Dome keeps the best epoch of the final no-augmentation stage (best_stg2) and of the stage before (best_stg1)."""
    for name in ('best_stg2.pth', 'best_stg1.pth', 'last.pth'):
        if (train / name).is_file():
            return train / name
    raise FileNotFoundError(f'no checkpoint in {train}')


def best_epoch_map(log: Path) -> float | None:
    if not log.is_file():
        return None
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    values = [row['test_coco_eval_bbox'][0] for row in rows if 'test_coco_eval_bbox' in row]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--weights', type=Path, help='default: the best checkpoint in runs/<tag>/train')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()

    run = ROOT / 'runs' / args.tag
    weights = args.weights or best_weights(run / 'train')
    use_dome()
    import torch
    import torchvision.transforms.v2.functional as F
    from PIL import Image
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(run / 'config.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    state = torch.load(weights, map_location='cpu', weights_only=False)
    model = cfg.model
    model.load_state_dict(state['ema']['module'] if state.get('ema') else state['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    parameters = sum(p.numel() for p in model.parameters())
    print(f'{weights}: {parameters / 1e6:.2f}M parameters', flush=True)

    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / 'data/instances_val_holdout.json', out / 'instances_val_holdout.json')
    images = json.loads((out / 'instances_val_holdout.json').read_text())['images']
    predictions, classes = [], {}
    with torch.no_grad():
        for index, image in enumerate(images, 1):
            with Image.open(DATA / 'train' / image['file_name']) as pil:
                pil = pil.convert('RGB')
                width, height = pil.size
                canvas, scale = fit_image(pil, args.size)
                pixels = F.pil_to_tensor(canvas).float().div(255)
            outputs = model(pixels.unsqueeze(0).to(device))  # no targets: nothing from the labels reaches the model
            extent = args.size / scale  # the canvas in original pixels; Dome scales normalised boxes by this
            sizes = torch.tensor([[extent, extent]], dtype=torch.float32, device=device)
            result = postprocessor(outputs, sizes)[0]
            scores, order = result['scores'].sort(descending=True)
            order = order[:KEEP]
            boxes, labels = result['boxes'][order].tolist(), result['labels'][order].tolist()
            for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores[:KEEP].tolist()):
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                if x2 > x1 and y2 > y1:
                    predictions.append({'image_id': int(image['id']), 'category_id': SLOT_TO_CATEGORY[int(label)],
                                        'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
            top = SLOT_TO_CATEGORY[int(labels[0])] if labels else 1
            classes[int(image['id'])] = top - 1  # labels.csv uses category id - 1
            if index % 200 == 0:
                print(f'predicted {index}/{len(images)} holdout images', flush=True)
    (out / 'predictions_val.json').write_text(json.dumps(predictions) + '\n')
    (out / 'image_classification_predictions.json').write_text(json.dumps(classes) + '\n')

    from sarbench.twins import score
    print(f'== {args.tag} Dome-DETR-M (YOLO control continue_2 one2many: mAP 0.4498 full / 0.4189 clean; '
          f'F1 here is only "class of the top box")', flush=True)
    report = score(out, DATA)
    report['weights'] = str(weights)
    report['parameters'] = parameters
    report['dome_log_best_holdout_mAP'] = best_epoch_map(run / 'train/log.txt')
    (out / 'holdout.full-clean.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
