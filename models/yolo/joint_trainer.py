"""Ultralytics trainer extensions for aligned joint SAR classification/detection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from models.yolo.image_head import classification_metrics
from models.yolo.joint_model import JointDetectionModel


F1_REFERENCE = 0.9331
MAP_REFERENCE = 0.2613


def combined_delta_m(macro_f1: float, map_score: float) -> float:
    return 100.0 * 0.5 * (
        (macro_f1 - F1_REFERENCE) / F1_REFERENCE
        + (map_score - MAP_REFERENCE) / MAP_REFERENCE
    )


def detection_class_metrics(metrics) -> list[dict]:
    """Serialize per-class detection precision, recall, F1 and COCO AP."""
    return [
        {
            "class": str(row["Class"]),
            "images": int(row["Images"]),
            "instances": int(row["Instances"]),
            "precision": float(row["Box-P"]),
            "recall": float(row["Box-R"]),
            "f1": float(row["Box-F1"]),
            "map50": float(row["mAP50"]),
            "map50_95": float(row["mAP50-95"]),
        }
        for row in metrics.summary()
    ]


def align_image_labels(image_paths: list[str], class_by_stem: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    targets = [class_by_stem.get(Path(str(path)).stem, -1) for path in image_paths]
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return target_tensor, target_tensor.ge(0)


class JointDetectionTrainer(DetectionTrainer):
    """Train detection and image labels together and select by combined validation Δm."""

    context: dict = {}

    @classmethod
    def configure(cls, **context) -> None:
        cls.context = context

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        context = self.context
        self.dataset_root = Path(context["dataset_root"]).resolve()
        self.data_root = Path(context["data_root"]).resolve()
        self.split_path = Path(context["split_path"]).resolve()
        self.classes_path = Path(context["classes_path"]).resolve()
        self.classifier_checkpoint = context.get("classifier_checkpoint")
        self.architecture_family = context.get("architecture_family", "yolo11")
        self.classification_loss_weight = float(context.get("classification_loss_weight", 0.5))
        self.shared_lr_scale = float(context.get("shared_lr_scale", 0.1))
        self.class_names = json.loads(self.classes_path.read_text())["classes"]
        self.split = json.loads(self.split_path.read_text())

        label_path = self.data_root / "train" / "labels.csv"
        with label_path.open(newline="") as stream:
            labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
        image_map = json.loads((self.dataset_root / "image_map.json").read_text())
        labeled_train_ids = set(map(int, self.split["train_image_ids"])) & set(labels_by_id)
        self.class_by_stem = {
            Path(info["file_name"]).stem: labels_by_id[int(info["image_id"])]
            for relative, info in image_map.items()
            if int(info["image_id"]) in labeled_train_ids
        }
        mapped_train = {
            int(info["image_id"])
            for info in image_map.values()
            if int(info["image_id"]) in labeled_train_ids and Path(info["file_name"]).stem in self.class_by_stem
        }
        if mapped_train != labeled_train_ids:
            raise ValueError(
                f"Image-label map differs from labeled IDs in the saved train split: mapped={len(mapped_train)}, expected={len(labeled_train_ids)}"
            )
        counts = torch.bincount(
            torch.tensor([labels_by_id[image_id] for image_id in sorted(labeled_train_ids)]),
            minlength=len(self.class_names),
        ).float()
        if len(counts) != len(self.class_names) or (counts == 0).any():
            raise ValueError("Training labels must contain all nine image classes")
        weights = counts.sum() / (len(self.class_names) * counts)
        self.class_weights = weights.tolist()

        self.val_dataset = None
        self.val_loader = None
        self.best_combined_fitness = float("-inf")
        self.best_classifier_metrics: dict | None = None
        self.best_detection_metrics: list[dict] | None = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the joint model, transfer detector weights, and preserve E2's trained classifier."""
        model = JointDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            class_weights=self.class_weights,
            classification_loss_weight=self.classification_loss_weight,
        )
        model = self.set_model_names_for_load(model)
        if weights is not None:
            source_head = getattr(weights, "model", [None])[-1]
            target_head = model.model[-1]
            source_levels = len(getattr(source_head, "cv2", []))
            target_levels = len(getattr(target_head, "cv2", []))
            if source_levels == 3 and target_levels == 4:
                if self.architecture_family == "yolo26":
                    from models.yolo.train_yolo26s_p2 import transfer_standard_weights_to_p2
                elif self.architecture_family == "yolo11":
                    from models.yolo.train_100_epoch import transfer_standard_weights_to_p2
                else:
                    raise ValueError(f"Unsupported joint-training architecture family: {self.architecture_family}")
                # The transfer helpers use the Ultralytics YOLO wrapper interface;
                # the trainer supplies the underlying DetectionModel modules.
                transfer_standard_weights_to_p2(SimpleNamespace(model=weights), SimpleNamespace(model=model))
            else:
                model.load(weights, verbose=verbose)

        if self.classifier_checkpoint:
            payload = torch.load(self.classifier_checkpoint, map_location="cpu", weights_only=True)
            if int(payload["feature_dim"]) != model.image_classifier.network[-1].in_features:
                raise ValueError(
                    f"E1 classifier feature dimension does not match the joint {self.architecture_family}-P2 model"
                )
            if payload["class_names"] != self.class_names:
                raise ValueError("E1 classifier class mapping differs from the experiment mapping")
            model.image_classifier.load_state_dict(payload["state_dict"], strict=True)
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        """Attach one label to each original image path; unknown labels get a masked target."""
        image_paths = batch.get("im_file", [])
        target_tensor, valid_tensor = align_image_labels(image_paths, self.class_by_stem)
        if not len(target_tensor):
            raise RuntimeError("YOLO batch did not include image paths for classification-label alignment")
        batch["sar_image_class"] = target_tensor
        batch["sar_image_class_valid"] = valid_tensor
        return super().preprocess_batch(batch)

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Use lower LR for shared backbone/neck parameters and the configured LR for both heads."""
        if str(name).lower() not in {"adamw", "auto"}:
            raise ValueError("Joint YOLO experiments currently use AdamW")
        target = model.module if hasattr(model, "module") else model
        detect_prefix = f"model.{len(target.model) - 1}."
        groups = {
            "shared_decay": [],
            "shared_no_decay": [],
            "heads_decay": [],
            "heads_no_decay": [],
        }
        for parameter_name, parameter in target.named_parameters():
            if not parameter.requires_grad:
                continue
            is_head = parameter_name.startswith("image_classifier.") or parameter_name.startswith(detect_prefix)
            no_decay = parameter.ndim <= 1 or parameter_name.endswith(".bias")
            key = ("heads_" if is_head else "shared_") + ("no_decay" if no_decay else "decay")
            groups[key].append(parameter)
        param_groups = []
        for key, parameters in groups.items():
            if not parameters:
                continue
            is_head = key.startswith("heads_")
            no_decay = key.endswith("no_decay")
            param_groups.append({
                "params": parameters,
                "lr": lr if is_head else lr * self.shared_lr_scale,
                "initial_lr": lr if is_head else lr * self.shared_lr_scale,
                "weight_decay": 0.0 if no_decay else decay,
                "param_group": key,
            })
        return torch.optim.AdamW(param_groups, betas=(momentum, 0.999), eps=1e-8)

    def _classification_validation_loader(self):
        if self.val_loader is not None:
            return self.val_loader
        from models.yolo.train_image_head import ImageLabels

        with (self.data_root / "train" / "labels.csv").open(newline="") as stream:
            labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
        image_map = json.loads((self.dataset_root / "image_map.json").read_text())
        val_ids = list(map(int, self.split["val_image_ids"]))
        self.val_dataset = ImageLabels(val_ids, labels_by_id, image_map, self.dataset_root)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=min(32, self.args.batch),
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        return self.val_loader

    @torch.inference_mode()
    def _evaluate_image_classifier(self) -> dict:
        target = self.ema.ema if self.ema is not None else (self.model.module if hasattr(self.model, "module") else self.model)
        target.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        for images, labels in self._classification_validation_loader():
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            target(images)
            feature_maps = target._joint_feature_maps
            if feature_maps is None:
                raise RuntimeError("Validation did not capture the Detect feature maps for classification")
            from models.yolo.image_head import pool_detection_features

            logits = target.image_classifier(pool_detection_features(feature_maps))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        return classification_metrics(y_true, y_pred, self.class_names)

    def validate(self):
        metrics, detector_fitness = super().validate()
        if metrics is None:
            return metrics, detector_fitness
        classifier_metrics = self._evaluate_image_classifier()
        map_score = next((float(value) for key, value in metrics.items() if key.endswith("mAP50-95(B)")), None)
        if map_score is None:
            raise KeyError("Detection validation did not report COCO mAP50-95")
        delta_m = combined_delta_m(classifier_metrics["macro_f1"], map_score)
        metrics["metrics/image_macro_f1"] = float(classifier_metrics["macro_f1"])
        metrics["metrics/combined_delta_m"] = float(delta_m)
        self.latest_classifier_metrics = classifier_metrics
        if delta_m > self.best_combined_fitness:
            self.best_combined_fitness = delta_m
            self.best_classifier_metrics = classifier_metrics
            self.best_detection_metrics = detection_class_metrics(self.validator.metrics)
        # BaseTrainer.save_model uses best_fitness == fitness to write best.pt.
        self.best_fitness = self.best_combined_fitness
        return metrics, delta_m
