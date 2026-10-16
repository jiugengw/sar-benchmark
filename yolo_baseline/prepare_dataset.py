#!/usr/bin/env python3
"""Convert the project COCO annotations to YOLO labels on the saved TerraMind split.

Images follow SARMultiTask's 512 x 512 grayscale stretch resize. YOLO's image
loader expands each grayscale PNG to three identical channels for its pretrained
RGB model. Box coordinates are normalized before and after resize identically.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def yolo_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_w, box_h = map(float, bbox)
    if box_w <= 0 or box_h <= 0 or x < 0 or y < 0 or x + box_w > width + 1e-5 or y + box_h > height + 1e-5:
        raise ValueError(f"Invalid/out-of-bounds COCO box {bbox} for {width}x{height}")
    return ((x + box_w / 2) / width, (y + box_h / 2) / height, box_w / width, box_h / height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/course-data"))
    parser.add_argument("--split", type=Path,
                        default=Path("runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo11s-sar-512"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    source = args.data / "train"
    with (source / "labels.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    coco = json.loads((source / "instances.json").read_text())
    split = json.loads(args.split.read_text())
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        annotations.setdefault(int(ann["image_id"]), []).append(ann)
    class_items = sorted(coco["categories"], key=lambda item: int(item["id"]))
    class_to_yolo = {int(item["id"]): index for index, item in enumerate(class_items)}
    names = [item["name"] for item in class_items]

    train_indices = [int(index) for index in split["train_indices"]]
    val_ids = {int(image_id) for image_id in split["val_image_ids"]}
    if len(train_indices) != len(set(train_indices)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("Saved split has duplicate rows/image IDs")
    if set(train_indices) & set(split["val_indices"]):
        raise ValueError("Saved split has overlapping train and validation indices")
    if len(train_indices) + len(val_ids) != len(rows):
        raise ValueError(f"Saved split sizes do not cover labels.csv: {len(train_indices)} + {len(val_ids)} != {len(rows)}")

    train_ids = set()
    for index in train_indices:
        row = rows[index]
        row_image_id = int(row["image_id"])
        if row_image_id not in image_by_id:
            raise ValueError(f"labels.csv image ID {row_image_id} is absent from COCO")
        train_ids.add(row_image_id)
    if train_ids & val_ids or train_ids | val_ids != set(image_by_id):
        raise ValueError("Saved train and validation IDs do not exactly cover the COCO images")

    args.out.mkdir(parents=True, exist_ok=True)
    id_to_row = {int(row["image_id"]): row for row in rows}
    split_records = {"train": sorted(train_ids), "val": sorted(val_ids)}
    image_map: dict[str, dict] = {}
    annotation_count = 0
    for split_name, image_ids in split_records.items():
        image_dir = args.out / "images" / split_name
        label_dir = args.out / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            info = image_by_id[image_id]
            row = id_to_row[image_id]
            image_name = Path(info["file_name"]).name
            source_path = source / row["file_name"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            width, height = int(info["width"]), int(info["height"])
            with Image.open(source_path) as original:
                gray = original.convert("L").resize((args.size, args.size), Image.Resampling.BILINEAR)
                gray.save(image_dir / f"{Path(image_name).stem}.png", format="PNG")

            lines = []
            for ann in annotations.get(image_id, []):
                category_id = int(ann["category_id"])
                if category_id not in class_to_yolo:
                    raise ValueError(f"Unknown category ID {category_id}")
                if class_to_yolo[category_id] != int(row["label"]):
                    raise ValueError(f"Image {image_id} class label and COCO box category disagree")
                cx, cy, bw, bh = yolo_box(ann["bbox"], width, height)
                # Reconstruct the original COCO box to catch coordinate-conversion mistakes.
                x, y, box_w, box_h = map(float, ann["bbox"])
                reconstructed = ((cx - bw / 2) * width, (cy - bh / 2) * height,
                                 bw * width, bh * height)
                if any(abs(actual - expected) > 1e-5 for actual, expected in
                       zip(reconstructed, (x, y, box_w, box_h))):
                    raise ValueError(f"YOLO coordinate round-trip failed for image {image_id}: {ann['bbox']}")
                lines.append(f"{class_to_yolo[category_id]} {cx:.9f} {cy:.9f} {bw:.9f} {bh:.9f}")
            if not lines:
                raise ValueError(f"Image {image_id} has no boxes; this dataset expects >= 1 box per image")
            (label_dir / f"{Path(image_name).stem}.txt").write_text("\n".join(lines) + "\n")
            annotation_count += len(lines)
            image_map[f"images/{split_name}/{Path(image_name).stem}.png"] = {
                "image_id": image_id,
                "file_name": info["file_name"],
                "width": width,
                "height": height,
                "csv_image_label": int(row["label"]),
                "boxes": len(lines),
            }

    dataset_yaml = args.out / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(names))
    )
    (args.out / "image_map.json").write_text(json.dumps(image_map, indent=2) + "\n")
    manifest = {
        "source_dataset": str(args.data),
        "source_split": str(args.split),
        "image_size": [args.size, args.size],
        "resize": "grayscale PIL bilinear stretch to 512 x 512, matching SARMultiTask geometry=stretch",
        "channels": "saved as 8-bit grayscale PNG; Ultralytics OpenCV loader expands to three identical RGB/BGR channels",
        "box_format": "class_id x_center y_center width height, normalized to 0..1; zero-based class IDs follow COCO category order",
        "classes": names,
        "counts": {"train_images": len(train_ids), "val_images": len(val_ids), "annotations": annotation_count},
        "split_image_ids": split_records,
    }
    (args.out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset_yaml": str(dataset_yaml), **manifest["counts"], "classes": names}, indent=2))


if __name__ == "__main__":
    main()
