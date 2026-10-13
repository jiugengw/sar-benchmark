"""Write a submission ZIP, or predict the course validation images from a completed run.

Example: python make_submission.py --run runs/terramind-moelora-seed0/terramind_pretrained_moelora \
    --data data/course-data --out submissions/terramind_pretrained_moelora_seed0.zip
"""
import argparse
import json
import zipfile
from pathlib import Path

def write_submission(images, preds, size=512, out="submission.zip", geometry='stretch'):
    """images: from images.json; preds[file_name] = (label,
    boxes [x1, y1, x2, y2] on the model input, classes, scores).
    Both the classification label and detection classes are zero-based (0..8).
    """
    rows, dets = ["image_id,label"], []
    for im in images:
        label, boxes, classes, scores = preds[im["file_name"]]
        rows.append(f"{im['id']},{int(label)}")  # the integer id
        if geometry == 'pad':
            from sarbench.data import padded_size
            rh, rw = padded_size((im['height'], im['width']), size)
        else:
            rh, rw = size, size
        sx, sy = im["width"] / rw, im["height"] / rh
        for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
            x1, x2 = [min(im['width'], max(0, x * sx)) for x in (x1, x2)]
            y1, y2 = [min(im['height'], max(0, y * sy)) for y in (y1, y2)]
            if x2 <= x1 or y2 <= y1:
                continue
            box = [x1, y1, x2 - x1, y2 - y1]
            dets.append({"image_id": im["id"],
                         "category_id": int(c) + 1,  # label + 1
                         "bbox": [float(v) for v in box],
                         "score": float(s)})
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("classification.csv", "\n".join(rows) + "\n")
        z.writestr("detection.json", json.dumps(dets))


def inference_transform(geometry='stretch'):
    from torchvision.transforms import v2
    import torch
    from sarbench.data import MEAN, STD, ResizePad

    return v2.Compose([
        v2.ToImage(), ResizePad() if geometry == 'pad' else v2.Resize((512, 512)),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize([MEAN], [STD]),
    ])


class SubmissionImages:
    """Unlabeled images, with the same preprocessing used during validation."""

    def __init__(self, split_root, images, geometry='stretch'):
        self.root = Path(split_root)
        self.images = images
        self.transform = inference_transform(geometry)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        from PIL import Image

        info = self.images[index]
        with Image.open(self.root / info['file_name']) as image:
            tensor = self.transform(image.convert('L'))
        return tensor, info['file_name']


def main():
    import torch
    from torch.utils.data import DataLoader
    from sarbench.adapters import add_adapters
    from sarbench.backbones import build_backbone
    from sarbench.model import MultiTaskModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='folder containing metrics.json and model.pt')
    parser.add_argument('--data', type=Path, required=True, help='course data root')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--out', type=Path, required=True, help='submission ZIP path')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    config = json.loads((args.run / 'metrics.json').read_text())['args']
    split_root = args.data / args.split
    images = json.loads((split_root / 'images.json').read_text())['images']
    if not images:
        raise ValueError('The submission split contains no images.')
    if len({image['file_name'] for image in images}) != len(images):
        raise ValueError('Submission image file names must be unique.')
    if len({image['id'] for image in images}) != len(images):
        raise ValueError('Submission image IDs must be unique.')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')

    # The checkpoint contains all weights; loading it requires no pretrained download.
    backbone = build_backbone(config['backbone'], pretrained=False)
    if config['adapt'] != 'full':
        add_adapters(backbone, config['adapt'])
    model = MultiTaskModel(backbone, task_routing=config['adapt'] == 'moelora',
                           anchor_profile=config.get('anchor_profile', 'default'))
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu', weights_only=True))
    model.cuda().eval()
    geometry = config.get('geometry', 'stretch')
    loader = DataLoader(SubmissionImages(split_root, images, geometry), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    predictions = {}
    with torch.inference_mode():
        for step, (batch, names) in enumerate(loader, 1):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, detections, _ = model(batch.cuda())
            labels = logits.argmax(dim=1).tolist()
            for name, label, detection in zip(names, labels, detections):
                # Faster R-CNN outputs category IDs 1..9; write_submission expects 0..8.
                predictions[name] = (label, detection['boxes'].cpu().tolist(),
                                     (detection['labels'] - 1).cpu().tolist(),
                                     detection['scores'].cpu().tolist())
            if step % 20 == 0 or step == len(loader):
                print(f'prediction batch {step}/{len(loader)}; {len(predictions)}/{len(images)} images', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.zip.tmp')
    write_submission(images, predictions, out=temporary, geometry=geometry)
    temporary.replace(args.out)
    print(f'Saved {args.out}: {len(images)} classification rows and '
          f'{sum(len(prediction[1]) for prediction in predictions.values())} detections', flush=True)


if __name__ == '__main__':
    main()
