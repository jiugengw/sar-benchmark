"""Evaluate original-pixel boxes and package both course submission outputs."""
import time
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from mmdet.structures import DetDataSample

from models.terramind.make_submission import write_submission
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.twins import clean_image_ids


class PredictionImages(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, path = self.records[index]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(pixels).unsqueeze(0).expand(3, -1, -1).contiguous()
        sample = DetDataSample(metainfo=dict(img_id=image_id, ori_shape=(512, 512), img_shape=(512, 512),
                                             scale_factor=(1., 1.), img_path=str(path)))
        return dict(inputs=tensor, data_samples=sample)


def predict(model, records, batch_size, workers):
    from mmengine.dataset import pseudo_collate
    loader = DataLoader(PredictionImages(records), batch_size=batch_size, num_workers=workers,
                        shuffle=False, collate_fn=pseudo_collate, pin_memory=True)
    predictions = {}
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            samples = model.test_step(batch)
            for sample in samples:
                det = sample.pred_instances.cpu()
                predictions[int(sample.img_id)] = (int(sample.image_logits.argmax()),
                    det.bboxes.tolist(), det.labels.tolist(), det.scores.tolist())
            if index % 50 == 0 or index == len(loader):
                print(f"Prediction {index}/{len(loader)} batches, {len(predictions)}/{len(records)} images", flush=True)
    return predictions


def export_run(model, args, metadata, dataset, checkpoint, settings):
    split = json.loads((dataset / "split.json").read_text())
    image_map = json.loads((args.converted / "image_map.json").read_text())
    paths = {int(info["image_id"]): args.converted / relative for relative, info in image_map.items()}
    ids = list(map(int, split["val_image_ids"]))
    predictions = predict(model, [(image_id, paths[image_id]) for image_id in ids], args.batch, args.workers)
    coco = json.loads((dataset / "instances_val_holdout.json").read_text())
    dets = []
    for image in coco["images"]:
        _, boxes, classes, scores = predictions[int(image["id"])]
        sx, sy = image["width"] / 512, image["height"] / 512
        for (x1, y1, x2, y2), label, score in zip(boxes, classes, scores):
            x1, x2 = min(max(x1*sx, 0), image["width"]), min(max(x2*sx, 0), image["width"])
            y1, y2 = min(max(y1*sy, 0), image["height"]), min(max(y2*sy, 0), image["height"])
            if x2 > x1 and y2 > y1:
                dets.append(dict(image_id=image["id"], category_id=label+1,
                                 bbox=[x1, y1, x2-x1, y2-y1], score=score))
    with (args.data / "train/labels.csv").open(newline="") as stream:
        labels = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    metrics = dict(classification=classification_metrics([labels[i] for i in ids], [predictions[i][0] for i in ids]),
                   detection=detection_metrics(str(dataset / "instances_val_holdout.json"), dets, ids),
                   checkpoint=str(checkpoint), settings=settings)
    metrics["delta_m"] = 50 * (metrics["classification"]["macro_f1"] / 0.9331 - 1 +
                               metrics["detection"]["mAP"] / 0.2613 - 1)
    # The same scores on the clean holdout subset: images with no near-duplicate twin among the training images.
    clean = clean_image_ids(coco["images"])
    metrics["clean_holdout"] = dict(
        images=len(clean),
        classification=classification_metrics([labels[i] for i in clean], [predictions[i][0] for i in clean]),
        detection=detection_metrics(str(dataset / "instances_val_holdout.json"), dets, clean))
    metrics["clean_holdout"]["delta_m"] = 50 * (metrics["clean_holdout"]["classification"]["macro_f1"] / 0.9331 - 1 +
                                                metrics["clean_holdout"]["detection"]["mAP"] / 0.2613 - 1)
    result = Path("results") / args.output.name / "seed_0"
    result.mkdir(parents=True, exist_ok=True)
    for destination in [args.output, result]:
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (destination / "predictions_val.json").write_text(json.dumps(dets) + "\n")
        (destination / "instances_val_holdout.json").write_text(json.dumps(coco) + "\n")
        (destination / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"Holdout: F1={metrics['classification']['macro_f1']:.4f}, mAP={metrics['detection']['mAP']:.4f}, delta_m={metrics['delta_m']:.2f}", flush=True)
    clean_scores = metrics["clean_holdout"]
    print(f"Clean holdout ({clean_scores['images']} images): F1={clean_scores['classification']['macro_f1']:.4f}, "
          f"mAP={clean_scores['detection']['mAP']:.4f}, delta_m={clean_scores['delta_m']:.2f}", flush=True)
    images = json.loads((args.data / "val/images.json").read_text())["images"]
    public = predict(model, [(int(im["id"]), args.data / "val" / im["file_name"]) for im in images], args.batch, args.workers)
    package = Path("nonsubmissions") / time.strftime("%d%m%y") / args.output.name
    package.mkdir(parents=True, exist_ok=True)
    output = package / f"{args.output.name}.zip"
    if output.exists():
        raise FileExistsError(output)
    write_submission(images, {im["file_name"]: public[int(im["id"])] for im in images}, out=output)
    (package / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Submission ZIP: {output}", flush=True)
