"""RT-DETR model and trainer for joint image and object classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel

from yolo_baseline.image_head import ImageClassifierHead, classification_metrics, pool_detection_features
from yolo_baseline.joint_trainer import combined_delta_m, detection_class_metrics


class JointRTDETRDetectionModel(RTDETRDetectionModel):
    """Add image-label loss from the decoder's live multi-scale input features."""

    def __init__(
        self,
        cfg,
        nc: int = 80,
        ch: int = 3,
        verbose: bool = True,
        class_weights: list[float] | None = None,
        classification_loss_weight: float = 0.5,
        classifier_dropout: float = 0.2,
    ):
        super().__init__(cfg=cfg, nc=nc, ch=ch, verbose=verbose)
        # Ultralytics' RT-DETR loss reads ``self.nc``, but the current
        # DetectionModel initializer only stores the configured count in yaml.
        self.nc = int(self.yaml["nc"])
        input_projections = self.model[-1].input_proj
        feature_dim = sum(int(projection[0].in_channels) for projection in input_projections)
        self.image_classifier = ImageClassifierHead(feature_dim, num_classes=nc, dropout=classifier_dropout)
        weights = class_weights or [1.0] * nc
        if len(weights) != nc:
            raise ValueError(f"Expected {nc} image-class weights, found {len(weights)}")
        self.register_buffer("image_class_weights", torch.tensor(weights, dtype=torch.float32))
        self.classification_loss_weight = float(classification_loss_weight)
        self._joint_feature_maps = None
        self._joint_detection_loss = None
        self._joint_classification_loss = None
        self.model[-1].register_forward_pre_hook(self._capture_decoder_inputs)

    def _capture_decoder_inputs(self, _module, inputs) -> None:
        self._joint_feature_maps = inputs[0]

    def loss(self, batch: dict, preds=None):
        """Combine RT-DETR's normal detection loss with aligned whole-image labels."""
        if preds is None:
            self._joint_feature_maps = None
        detection_loss, detection_items = super().loss(batch, preds)
        targets = batch.get("sar_image_class")
        if targets is None:
            # The standard RT-DETR validator does not carry whole-image labels.
            # Its detector metrics are combined with a separate labeled pass.
            self._joint_detection_loss = detection_loss
            self._joint_classification_loss = None
            return detection_loss, detection_items

        feature_maps = self._joint_feature_maps
        if feature_maps is None:
            raise RuntimeError("The RT-DETR decoder hook did not capture this training batch")
        logits = self.image_classifier(pool_detection_features(feature_maps))
        targets = targets.to(device=logits.device, dtype=torch.long)
        valid = batch.get("sar_image_class_valid", targets >= 0).to(device=logits.device, dtype=torch.bool)
        if valid.any():
            classification_loss = F.cross_entropy(
                logits[valid],
                targets[valid],
                weight=self.image_class_weights.to(dtype=logits.dtype),
            )
        else:
            classification_loss = logits.sum() * 0.0

        # RT-DETR's detector loss is batch-scaled. Apply the same scaling to the
        # image-level term so lambda=0.5 has a stable meaning across batch sizes.
        classification_term = self.classification_loss_weight * classification_loss * int(valid.sum())
        total_loss = detection_loss + classification_term
        self._joint_detection_loss = detection_loss
        self._joint_classification_loss = classification_loss
        items = {name: value.detach() for name, value in detection_items.items()}
        items["image_cls_loss"] = classification_loss.detach()
        return total_loss, items


class JointRTDETRTrainer(RTDETRTrainer):
    """RT-DETR trainer with aligned image labels and combined-Δm selection."""

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
        self.classifier_checkpoint = Path(context["classifier_checkpoint"]).resolve()
        self.classification_loss_weight = float(context["classification_loss_weight"])
        self.shared_lr_scale = float(context["shared_lr_scale"])
        self.classifier_lr = float(context["classifier_lr"])
        self.class_names = json.loads(self.classes_path.read_text())["classes"]
        self.split = json.loads(self.split_path.read_text())

        with (self.data_root / "train" / "labels.csv").open(newline="") as stream:
            labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
        image_map = json.loads((self.dataset_root / "image_map.json").read_text())
        train_ids = set(map(int, self.split["train_image_ids"]))
        self.class_by_stem = {
            Path(info["file_name"]).stem: labels_by_id[int(info["image_id"])]
            for info in image_map.values()
            if int(info["image_id"]) in train_ids
        }
        mapped_train_ids = {
            int(info["image_id"])
            for info in image_map.values()
            if int(info["image_id"]) in train_ids and Path(info["file_name"]).stem in self.class_by_stem
        }
        if mapped_train_ids != train_ids:
            raise ValueError(
                f"Converted image map differs from the saved training split: mapped={len(mapped_train_ids)}, expected={len(train_ids)}"
            )
        counts = torch.bincount(
            torch.tensor([labels_by_id[image_id] for image_id in sorted(train_ids)]),
            minlength=len(self.class_names),
        ).float()
        if len(counts) != len(self.class_names) or (counts == 0).any():
            raise ValueError("Training labels must contain all image classes")
        self.class_weights = (counts.sum() / (len(self.class_names) * counts)).tolist()

        self.val_dataset = None
        self.val_loader = None
        self.best_combined_fitness = float("-inf")
        self.best_classifier_metrics: dict | None = None
        self.best_detection_metrics: list[dict] | None = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = JointRTDETRDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            class_weights=self.class_weights,
            classification_loss_weight=self.classification_loss_weight,
        )
        model = self.set_model_names_for_load(model)
        if weights is not None:
            model.load(weights, verbose=verbose)

        payload = torch.load(self.classifier_checkpoint, map_location="cpu", weights_only=True)
        if int(payload["feature_dim"]) != model.image_classifier.network[-1].in_features:
            raise ValueError("E7 image-classifier dimensions do not match the RT-DETR-L-P2 decoder features")
        if payload["class_names"] != self.class_names:
            raise ValueError("E7 image-classifier class mapping differs from the dataset class mapping")
        model.image_classifier.load_state_dict(payload["state_dict"], strict=True)
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        paths = batch.get("im_file", [])
        targets = [self.class_by_stem.get(Path(str(path)).stem, -1) for path in paths]
        if not targets:
            raise RuntimeError("RT-DETR batch did not include image paths for image-label alignment")
        batch["sar_image_class"] = torch.tensor(targets, dtype=torch.long)
        batch["sar_image_class_valid"] = batch["sar_image_class"].ge(0)
        return super().preprocess_batch(batch)

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        if str(name).lower() not in {"adamw", "auto"}:
            raise ValueError("Joint RT-DETR experiments use AdamW")
        target = model.module if hasattr(model, "module") else model
        decoder_prefix = f"model.{len(target.model) - 1}."
        groups: dict[str, list[torch.nn.Parameter]] = {
            "shared_decay": [], "shared_no_decay": [],
            "detector_decay": [], "detector_no_decay": [],
            "classifier_decay": [], "classifier_no_decay": [],
        }
        for parameter_name, parameter in target.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter_name.startswith("image_classifier."):
                family = "classifier"
            elif parameter_name.startswith(decoder_prefix):
                family = "detector"
            else:
                family = "shared"
            no_decay = parameter.ndim <= 1 or parameter_name.endswith(".bias")
            groups[f"{family}_{'no_decay' if no_decay else 'decay'}"].append(parameter)

        param_groups = []
        for name, parameters in groups.items():
            if not parameters:
                continue
            family, decay_group = name.split("_", 1)
            group_lr = {
                "shared": lr * self.shared_lr_scale,
                "detector": lr,
                "classifier": self.classifier_lr,
            }[family]
            param_groups.append({
                "params": parameters,
                "lr": group_lr,
                "initial_lr": group_lr,
                "weight_decay": 0.0 if decay_group == "no_decay" else decay,
                "param_group": name,
            })
        return torch.optim.AdamW(param_groups, betas=(momentum, 0.999), eps=1e-8)

    def _classification_validation_loader(self):
        if self.val_loader is not None:
            return self.val_loader
        from yolo_baseline.train_image_head import ImageLabels

        with (self.data_root / "train" / "labels.csv").open(newline="") as stream:
            labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
        image_map = json.loads((self.dataset_root / "image_map.json").read_text())
        self.val_dataset = ImageLabels(
            list(map(int, self.split["val_image_ids"])), labels_by_id, image_map, self.dataset_root
        )
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
            target._joint_feature_maps = None
            if feature_maps is None:
                raise RuntimeError("RT-DETR validation did not capture decoder features")
            logits = target.image_classifier(pool_detection_features(feature_maps))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        return classification_metrics(y_true, y_pred, self.class_names)

    def validate(self):
        metrics, _detector_fitness = super().validate()
        if metrics is None:
            return metrics, None
        classifier_metrics = self._evaluate_image_classifier()
        map_score = next((float(value) for key, value in metrics.items() if key.endswith("mAP50-95(B)")), None)
        if map_score is None:
            raise KeyError("RT-DETR validation did not report COCO mAP50-95")
        delta_m = combined_delta_m(classifier_metrics["macro_f1"], map_score)
        metrics["metrics/image_macro_f1"] = float(classifier_metrics["macro_f1"])
        metrics["metrics/combined_delta_m"] = float(delta_m)
        if delta_m > self.best_combined_fitness:
            self.best_combined_fitness = delta_m
            self.best_classifier_metrics = classifier_metrics
            self.best_detection_metrics = detection_class_metrics(self.validator.metrics)
        self.best_fitness = self.best_combined_fitness
        return metrics, delta_m
