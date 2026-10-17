#!/usr/bin/env python3
"""Validate stock Deformable DETR, then train its four-level P2 SAR variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deformable_detr_support import make_p2_model
from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


class SarDetectionDataset(Dataset):
    def __init__(self, image_ids, image_map, dataset_root, processor):
        self.dataset_root = dataset_root
        self.processor = processor
        self.records = []
        for relative, info in image_map.items():
            image_id = int(info["image_id"])
            if image_id not in image_ids:
                continue
            split_name = "train" if relative.startswith("images/train/") else "val"
            label_path = dataset_root / "labels" / split_name / f"{Path(relative).stem}.txt"
            boxes, classes = [], []
            for row in label_path.read_text().splitlines():
                category, cx, cy, width, height = map(float, row.split())
                classes.append(int(category))
                boxes.append([cx, cy, width, height])
            self.records.append((image_id, dataset_root / relative, classes, boxes))
        self.records.sort(key=lambda row: row[0])
        if len(self.records) != len(image_ids):
            raise ValueError(f"Dataset split has {len(self.records)} images; expected {len(image_ids)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, path, classes, boxes = self.records[index]
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.uint8).copy()
        processed = self.processor(images=array, return_tensors="pt")
        return image_id, processed["pixel_values"][0], processed["pixel_mask"][0], {
            "class_labels": torch.tensor(classes, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }


def collate(records):
    image_ids, pixel_values, pixel_masks, labels = zip(*records)
    return list(image_ids), torch.stack(pixel_values), torch.stack(pixel_masks), list(labels)


def build_predictions(model, processor, loader, device) -> list[dict]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for image_ids, pixel_values, pixel_masks, _labels in loader:
            outputs = model(pixel_values=pixel_values.to(device), pixel_mask=pixel_masks.to(device))
            sizes = torch.tensor([[512, 512]] * len(image_ids), device=device)
            decoded = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.001)
            for image_id, output in zip(image_ids, decoded):
                boxes = output["boxes"].detach().cpu().tolist()
                classes = output["labels"].detach().cpu().tolist()
                scores = output["scores"].detach().cpu().tolist()
                for (x1, y1, x2, y2), category, score in zip(boxes, classes, scores):
                    predictions.append({
                        "image_id": int(image_id),
                        "category_id": int(category) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    })
    return predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/deformable-detr-standard"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/yolo11s-sar-512"))
    parser.add_argument("--split", type=Path, default=Path("runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/instances_val_holdout.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoImageProcessor, DeformableDetrConfig, DeformableDetrForObjectDetection

    checkpoint = resolve(args.checkpoint)
    dataset_root = resolve(args.dataset_root)
    split_path = resolve(args.split)
    output = resolve(args.output)
    ground_truth = resolve(args.ground_truth)
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing staged standard Deformable DETR model: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Deformable DETR training requires the allocated CUDA GPU")
    output.mkdir(parents=True, exist_ok=True)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.workers)
    device = torch.device("cuda:0")

    class_names = json.loads((dataset_root / "conversion_manifest.json").read_text())["classes"]
    split = json.loads(split_path.read_text())
    train_ids = list(map(int, split["train_image_ids"]))
    val_ids = list(map(int, split["val_image_ids"]))
    image_map = json.loads((dataset_root / "image_map.json").read_text())
    experiment_config = {
        "experiment": "E8",
        "architecture": "Deformable DETR ResNet-50-P2",
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "dataset_root": str(dataset_root),
        "ground_truth": str(ground_truth),
        "split": str(split_path),
        "class_names": class_names,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch,
        "workers": args.workers,
        "optimizer": "AdamW with backbone lr=2e-5, transformer/head lr=2e-4, cosine schedule",
        "augmentation": "resize and pretrained processor normalization; no stochastic geometric augmentation",
        "feature_levels": "stride 4, 8, 16, 32; P3-P5 weights transferred, new P2 projection initialized randomly",
        "checkpoint_selection": "best validation COCO mAP; final architecture selection uses the combined Δm after classifier training",
        "seed": seed,
    }
    (output / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2) + "\n")
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    processor.size = {"height": 512, "width": 512}

    # First run the untouched standard pretrained model and record its P3/P4/P5 path.
    standard_config = DeformableDetrConfig.from_pretrained(checkpoint, local_files_only=True)
    standard_config.use_pretrained_backbone = False
    standard = DeformableDetrForObjectDetection.from_pretrained(
        checkpoint, config=standard_config, local_files_only=True
    ).to(device).eval()
    first_path = dataset_root / next(relative for relative, info in image_map.items() if int(info["image_id"]) == train_ids[0])
    with Image.open(first_path) as source_image:
        sample = np.asarray(source_image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)
    preflight_inputs = processor(images=sample, return_tensors="pt")
    preflight_inputs = {key: value.to(device) for key, value in preflight_inputs.items()}
    preflight_inputs["pixel_values"] = torch.nn.functional.interpolate(
        preflight_inputs["pixel_values"], size=(256, 256), mode="bilinear", align_corners=False
    )
    preflight_inputs["pixel_mask"] = torch.ones((1, 256, 256), dtype=torch.long, device=device)
    with torch.inference_mode():
        standard_outputs = standard(**preflight_inputs)
        stock_features, _ = standard.model.backbone(
            preflight_inputs["pixel_values"], preflight_inputs["pixel_mask"]
        )
    standard_summary = {
        "status": "passed",
        "architecture": "stock pretrained Deformable DETR ResNet-50 before P2 changes",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "standard_backbone_feature_shapes_at_256": [list(feature.shape[-2:]) for feature, _mask in stock_features],
        "query_logits_shape": list(standard_outputs.logits.shape),
        "validation_scope": "pretrained standard-model forward preflight before constructing the P2 variant",
    }
    (output / "standard_preflight.json").write_text(json.dumps(standard_summary, indent=2) + "\n")
    del standard, standard_outputs, stock_features
    torch.cuda.empty_cache()

    model, config, transfer = make_p2_model(checkpoint, class_names)
    model.to(device)
    model.eval()
    p2_hook_output = []
    handle = model.model.backbone.register_forward_hook(lambda _module, _inputs, result: p2_hook_output.append(result))
    with torch.inference_mode():
        test_inputs = processor(images=sample, return_tensors="pt")
        test_pixel_values = torch.nn.functional.interpolate(test_inputs["pixel_values"].to(device), size=(256, 256), mode="bilinear", align_corners=False)
        test_pixel_mask = torch.ones((1, 256, 256), dtype=torch.long, device=device)
        test_outputs = model(pixel_values=test_pixel_values, pixel_mask=test_pixel_mask)
    handle.remove()
    p2_features = [feature for feature, _mask in p2_hook_output[0][0]]
    p2_shapes = [list(feature.shape[-2:]) for feature in p2_features]
    if p2_shapes != [[64, 64], [32, 32], [16, 16], [8, 8]]:
        raise RuntimeError(f"Deformable DETR decoder is not using stride-4 P2 features: {p2_shapes}")
    (output / "p2_preflight.json").write_text(json.dumps({
        "status": "passed",
        "decoder_feature_shapes_at_256": p2_shapes,
        "query_logits_shape": list(test_outputs.logits.shape),
        "transfer": transfer,
    }, indent=2) + "\n")
    del test_outputs, p2_features, p2_hook_output, model
    torch.cuda.empty_cache()

    model, config, transfer = make_p2_model(checkpoint, class_names)
    model.to(device)
    processor.size = {"height": 512, "width": 512}
    processor.save_pretrained(output / "processor")
    train_dataset = SarDetectionDataset(set(train_ids), image_map, dataset_root, processor)
    val_dataset = SarDetectionDataset(set(val_ids), image_map, dataset_root, processor)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate)
    backbone_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_parameters if ".backbone." in name else other_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": 2e-5, "name": "backbone"},
        {"params": other_parameters, "lr": 2e-4, "name": "transformer_and_heads"},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_map = float("-inf")
    best_epoch = 0
    history = []
    weights_dir = output / "train/weights/best"
    gt_images = json.loads(ground_truth.read_text())["images"]
    val_ids_from_gt = [int(item["id"]) for item in gt_images]
    clean_val_ids = clean_image_ids(gt_images)  # holdout images with no near-duplicate twin among training images
    if set(val_ids_from_gt) != set(val_ids):
        raise ValueError("Detection ground truth and saved Deformable DETR validation split differ")
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _image_ids, pixel_values, pixel_masks, labels in train_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            pixel_masks = pixel_masks.to(device, non_blocking=True)
            labels = [{key: value.to(device, non_blocking=True) for key, value in item.items()} for item in labels]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_masks, labels=labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Deformable DETR loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        epoch_predictions = build_predictions(model, processor, val_loader, device)
        gt_path = ground_truth
        map_metrics = detection_metrics(gt_path, epoch_predictions, val_ids)
        map_score = float(map_metrics["mAP"])
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)),
               "validation_map": map_score, "validation_per_class_ap": map_metrics["per_class_AP"],
               "validation_map_clean": detection_metrics(gt_path, epoch_predictions, clean_val_ids)["mAP"],
               "validation_clean_images": len(clean_val_ids)}
        history.append(row)
        print(f"Deformable DETR-P2 epoch {epoch}/{args.epochs}: loss={row['train_loss']:.5f} mAP={map_score:.5f}", flush=True)
        if map_score > best_map:
            best_map = map_score
            best_epoch = epoch
            weights_dir.mkdir(parents=True, exist_ok=True)
            config.id2label = {index: name for index, name in enumerate(class_names)}
            config.label2id = {name: index for index, name in enumerate(class_names)}
            model.save_pretrained(weights_dir, safe_serialization=True)
            processor.save_pretrained(weights_dir)
            (weights_dir / "checkpoint_metrics.json").write_text(json.dumps({
                "epoch": epoch, "mAP": map_score, "per_class_AP": map_metrics["per_class_AP"]
            }, indent=2) + "\n")
        (output / "validation_history.json").write_text(json.dumps(history, indent=2) + "\n")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    compute = {
        "experiment": "E8",
        "architecture": "Deformable DETR ResNet-50-P2, four-level stride-4/8/16/32",
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "best_checkpoint": str(weights_dir),
        "best_epoch_by_detection_mAP": best_epoch,
        "best_detection_mAP": best_map,
        "training_epochs": args.epochs,
        "optimizer": "AdamW, backbone lr=2e-5, transformer/head lr=2e-4, cosine schedule",
        "seed": seed,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "transfer": transfer,
    }
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
