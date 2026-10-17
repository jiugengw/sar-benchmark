"""Reuse the existing 512px grayscale images and saved split in COCO format."""
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageStat


def prepare(data: Path, converted: Path, split_path: Path, output: Path):
    coco = json.loads((data / "train/instances.json").read_text())
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    image_map = json.loads((converted / "image_map.json").read_text())
    mapping = {int(info["image_id"]): (relative, info) for relative, info in image_map.items()}
    split = json.loads(split_path.read_text())
    with (data / "train/labels.csv").open(newline="") as stream:
        labels = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    classes = [item["name"] for item in categories]
    if classes != manifest["classes"] or [int(c["id"]) for c in categories] != list(range(1, 10)):
        raise ValueError("Expected the benchmark's nine classes with category IDs 1..9")
    train_ids = set(map(int, split["train_image_ids"]))
    val_ids = set(map(int, split["val_image_ids"]))
    if train_ids & val_ids or train_ids | val_ids != set(labels):
        raise ValueError("Saved train/validation split must exactly cover labels.csv without overlap")
    if len(train_ids) != 8453 or len(val_ids) != 939:
        raise ValueError("Expected the existing 8453/939 comparison split")
    for name, ids in (("train", train_ids), ("val", val_ids)):
        if ids != set(manifest["split_image_ids"][name]):
            raise ValueError(f"Converted image split disagrees with saved split: {name}")
    output.mkdir(parents=True, exist_ok=True)
    originals = {int(im["id"]): im for im in coco["images"]}
    total, squares, pixels = 0.0, 0.0, 0
    for name, ids in (("train", train_ids), ("val", val_ids)):
        images, annotations = [], []
        for index, image_id in enumerate(sorted(ids), 1):
            relative, info = mapping[image_id]
            path = converted / relative
            if info["csv_image_label"] != labels[image_id]:
                raise ValueError(f"Image class mismatch for {image_id}")
            with Image.open(path) as image:
                if image.size != (512, 512) or image.mode != "L":
                    raise ValueError(f"Expected a 512x512 grayscale image: {path}")
                if name == "train":
                    stats = ImageStat.Stat(image)
                    total += stats.sum[0]
                    squares += stats.sum2[0]
                    pixels += 512 * 512
            # Reference the already converted PNG in place; do not duplicate 1.4 GB of images.
            images.append(dict(id=image_id, file_name=str(path.resolve()), width=512, height=512))
            if index % 1000 == 0:
                print(f"Preparing {name}: {index}/{len(ids)} images", flush=True)
        for ann in coco["annotations"]:
            image_id = int(ann["image_id"])
            if image_id not in ids:
                continue
            if int(ann["category_id"]) - 1 != labels[image_id]:
                raise ValueError(f"Box category disagrees with image class for {image_id}")
            im = originals[image_id]
            x, y, w, h = map(float, ann["bbox"])
            sx, sy = 512 / im["width"], 512 / im["height"]
            if min(w, h) <= 0 or min(x, y) < 0 or x + w > im["width"] + 1e-4 or y + h > im["height"] + 1e-4:
                raise ValueError(f"Invalid source box: {ann['id']}")
            # Segmentation masks are not used by this box detector and needlessly
            # duplicate storage when the split COCO files are written.
            annotations.append(dict(id=int(ann["id"]), image_id=image_id,
                                    category_id=int(ann["category_id"]),
                                    bbox=[x * sx, y * sy, w * sx, h * sy],
                                    area=w * h * sx * sy, iscrowd=int(ann.get("iscrowd", 0))))
        payload = dict(images=images, annotations=annotations, categories=categories)
        with (output / f"{name}.json").open("w") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.write("\n")
    mean = total / pixels
    std = math.sqrt(max(squares / pixels - mean ** 2, 1e-12))
    metadata = dict(classes=classes, mean=mean, std=std,
                    train_images=len(train_ids), val_images=len(val_ids),
                    source_split=str(split_path.resolve()), image_root=str(converted.resolve()),
                    image_size=512, normalization_source="training split only",
                    source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [split_path, manifest_path, data / "train/instances.json"]})
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    holdout = {"images": [im for im in coco["images"] if int(im["id"]) in val_ids],
               "annotations": [dict(id=int(ann["id"]), image_id=int(ann["image_id"]),
                                    category_id=int(ann["category_id"]), bbox=ann["bbox"],
                                    area=ann["area"], iscrowd=int(ann.get("iscrowd", 0)))
                               for ann in coco["annotations"] if int(ann["image_id"]) in val_ids],
               "categories": categories}
    (output / "instances_val_holdout.json").write_text(json.dumps(holdout) + "\n")
    print(f"Prepared {len(train_ids)} training / {len(val_ids)} holdout images; mean={mean:.4f}, std={std:.4f}", flush=True)
    return metadata
