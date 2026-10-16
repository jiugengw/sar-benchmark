#!/usr/bin/env python3
"""Fit a supervised image-classification head on frozen YOLO detection features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from image_head import ImageClassifierHead, classification_metrics, pool_detection_features


class ImageLabels(Dataset):
    def __init__(self, image_ids: list[int], labels: dict[int, int], image_map: dict, root: Path):
        path_by_id = {int(info["image_id"]): relative for relative, info in image_map.items()}
        self.records = []
        for image_id in image_ids:
            relative = path_by_id.get(image_id)
            if relative is None:
                raise ValueError(f"Image ID {image_id} is missing from the converted image map")
            self.records.append((root / relative, labels[image_id]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
        return tensor, int(label)


def extract_features(detector, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(pool_detection_features(inputs[0]).detach().cpu())

    handle = detector.model.model[-1].register_forward_pre_hook(capture)
    features, labels = [], []
    try:
        detector.model.eval()
        with torch.inference_mode():
            for images, targets in loader:
                captured.clear()
                detector.model(images.to(device, non_blocking=True))
                if len(captured) != 1 or captured[0].shape[0] != len(targets):
                    raise RuntimeError("Could not capture one set of YOLO FPN features per image batch")
                features.append(captured.pop())
                labels.append(targets.clone())
    finally:
        handle.remove()
    return torch.cat(features), torch.cat(labels).long()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/course-data"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/yolo11s-sar-512"))
    parser.add_argument("--split", type=Path, default=Path("runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("The classification head requires the allocated CUDA GPU")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")

    class_names = json.loads((args.dataset_root / "conversion_manifest.json").read_text())["classes"]
    split = json.loads(args.split.read_text())
    with (args.data / "train" / "labels.csv").open(newline="") as stream:
        label_rows = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    image_map = json.loads((args.dataset_root / "image_map.json").read_text())
    train_ids = list(map(int, split["train_image_ids"]))
    val_ids = list(map(int, split["val_image_ids"]))
    if len(train_ids) != 8453 or len(val_ids) != 939:
        raise ValueError(f"Unexpected saved split sizes: train={len(train_ids)}, holdout={len(val_ids)}")

    train_loader = DataLoader(
        ImageLabels(train_ids, label_rows, image_map, args.dataset_root),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ImageLabels(val_ids, label_rows, image_map, args.dataset_root),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    detector = YOLO(str(args.detector_checkpoint))
    detector.model.to(device).eval()
    start = time.perf_counter()
    train_features, train_targets = extract_features(detector, train_loader, device)
    val_features, val_targets = extract_features(detector, val_loader, device)
    extraction_seconds = time.perf_counter() - start
    del detector
    torch.cuda.empty_cache()

    feature_dim = int(train_features.shape[1])
    head = ImageClassifierHead(feature_dim, len(class_names), dropout=0.2).to(device)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)
    counts = torch.bincount(train_targets, minlength=len(class_names)).float()
    class_weights = counts.sum() / (len(class_names) * counts.clamp_min(1))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(0)
    batch = TensorDataset(train_features, train_targets)
    loader = DataLoader(batch, batch_size=args.batch, shuffle=True, generator=generator)

    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_losses = []
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"image-head epoch {epoch}/{args.epochs}: train loss={mean_loss:.4f}", flush=True)
    head_training_seconds = time.perf_counter() - training_start

    head.eval()
    with torch.inference_mode():
        predictions = head(val_features).argmax(dim=1).cpu().tolist()
    val_metrics = classification_metrics(val_targets.cpu().tolist(), predictions, class_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "sar-yolo-image-head-v1",
        "feature_dim": feature_dim,
        "num_classes": len(class_names),
        "class_names": class_names,
        "dropout": 0.2,
        "state_dict": head.state_dict(),
        "feature_source": "global average pooling of every feature map passed to YOLO's final Detect module",
        "supervision": "data/course-data/train/labels.csv image labels on the saved 8453/939 TerraMind split",
        "architecture": "LayerNorm + Dropout(0.2) + Linear(feature_dim, 9)",
        "training": {
            "epochs": args.epochs,
            "checkpoint_selection": "fixed final epoch; holdout was not used for checkpoint selection",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "class_weighting": "inverse training frequency",
            "seed": 0,
            "batch_size": args.batch,
            "feature_extraction_seconds": extraction_seconds,
            "head_training_seconds": head_training_seconds,
        },
        "holdout": val_metrics,
    }
    torch.save(payload, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"holdout": val_metrics, "checkpoint": str(args.output), **payload["training"]}, indent=2) + "\n")
    print(json.dumps({"epochs": args.epochs, "holdout_macro_f1": val_metrics["macro_f1"], "checkpoint": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
