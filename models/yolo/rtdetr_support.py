"""Pretrained weight transfer helpers for Ultralytics RT-DETR-L-P2."""

from __future__ import annotations

import re

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel


STANDARD_DECODER_INDEX = 28
P2_DECODER_INDEX = 32


def transfer_standard_rtdetr_to_p2(source_model, target_model) -> dict[str, int]:
    """Copy shared RT-DETR weights, shifting three decoder levels onto P3-P5."""
    source = source_model if isinstance(source_model, torch.nn.Module) else source_model.model
    target = target_model if isinstance(target_model, torch.nn.Module) else target_model.model
    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    backbone_neck = decoder = level_projection = 0

    for key, value in source_state.items():
        if key.startswith(f"model.{STANDARD_DECODER_INDEX}."):
            target_key = key.replace(
                f"model.{STANDARD_DECODER_INDEX}.", f"model.{P2_DECODER_INDEX}.", 1
            )
            projection_match = re.search(r"input_proj\.(\d+)\.", target_key)
            if projection_match:
                source_level = int(projection_match.group(1))
                # P2 is the new level zero. The stock P3/P4/P5 projections map to 1/2/3.
                target_key = target_key[:projection_match.start(1)] + str(source_level + 1) + target_key[projection_match.end(1):]
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    transferred[target_key] = value
                    level_projection += 1
            elif target_key in target_state and target_state[target_key].shape == value.shape:
                transferred[target_key] = value
                decoder += 1
        elif not key.startswith(f"model.{STANDARD_DECODER_INDEX + 1}."):
            if key in target_state and target_state[key].shape == value.shape:
                transferred[key] = value
                backbone_neck += 1

    source_embed_key = f"model.{STANDARD_DECODER_INDEX}.level_embed"
    target_embed_key = f"model.{P2_DECODER_INDEX}.level_embed"
    if source_embed_key in source_state and target_embed_key in target_state:
        source_embed = source_state[source_embed_key]
        target_embed = target_state[target_embed_key].clone()
        if source_embed.ndim == 2 and target_embed.shape[0] == source_embed.shape[0] + 1 and target_embed.shape[1] == source_embed.shape[1]:
            target_embed[1:] = source_embed
            transferred[target_embed_key] = target_embed
            level_projection += 1

    missing_keys, unexpected_keys = target.load_state_dict(transferred, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected transferred RT-DETR keys: {unexpected_keys[:8]}")
    if backbone_neck < 100 or decoder < 10 or level_projection < 10:
        raise RuntimeError(
            f"Insufficient RT-DETR transfer: backbone/neck={backbone_neck}, decoder={decoder}, projections={level_projection}"
        )
    return {
        "backbone_neck_tensors": backbone_neck,
        "decoder_tensors": decoder,
        "level_projection_tensors": level_projection,
        "new_p2_parameters": sum(parameter.numel() for parameter in target.model[28:32].parameters()),
        "skipped_target_keys": len(missing_keys),
    }


class RTDETRP2Trainer(RTDETRTrainer):
    """RT-DETR trainer that transfers the stock decoder onto a four-level P2 model."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        )
        if weights is not None:
            self.transfer_summary = transfer_standard_rtdetr_to_p2(weights, model)
        return model
