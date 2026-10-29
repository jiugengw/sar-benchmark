"""Frozen TerraMind-1.0-base features added into the YOLO detector's backbone: one network, one forward pass.

TerraMind (IBM/ESA; pretrained on Sentinel-1 radar and Sentinel-2 optical images; named in the course rules as an
allowed generic pretrained model, and the TA's own baseline backbone) reads the same 512 px image as the detector,
through its Sentinel-1 GRD input: the grey image as both VV and VH, standardised as in sarbench/data.py. Its last
block's tokens (32 x 32 x 768 at 512 px, stride 16) go through one 1x1 convolution per fusion point and are ADDED to
the detector's own feature maps after backbone layers 6 and 10 (stride 8 and 16 with the stride-1 stem), resized where
the grids differ. The convolutions start at zero, so the fused model starts exactly as the detector it is built from.
TerraMind stays frozen (eval mode, no gradients); the detector and the two convolutions train.

Why: new-scene bridges and playgrounds are mostly missed, not mislocated, and a bridge is defined by its context (a
road over water); TerraMind attends over the whole scene, the detector's backbone mostly locally.

Budget: TerraMind's encoder (about 86M parameters) is frozen and counts towards the 130M total; the two convolutions
add about 1M trainable parameters to the detector.

    from terramind_fusion import TerraMindTrainer
    YOLO('refit2.pt').train(trainer=TerraMindTrainer, ...)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MEAN, STD = 0.2526, 0.2133  # gray pixels in [0, 1] over the 9,392 training images (sarbench/data.py)
FUSE_AT = (6, 10)  # backbone layers whose outputs get TerraMind features added
LAYER_KEYS = ('f', 'i', 'type', 'np')  # attributes Ultralytics' forward loop reads from every layer


class TerraMindStem(nn.Module):
    """The detector's first layer, unchanged, plus TerraMind run on the same input; its tokens wait for the fusion
    layers further down the backbone."""

    def __init__(self, layer: nn.Module, pretrained: bool = True):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.layer = layer
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.terramind.requires_grad_(False).eval()
        for key in LAYER_KEYS:
            setattr(self, key, getattr(layer, key))
        self.tokens = None

    def __getstate__(self):  # a batch's tokens never go into a checkpoint or an EMA copy
        state = self.__dict__.copy()
        state['tokens'] = None
        return state

    def train(self, mode: bool = True):
        super().train(mode)
        self.terramind.eval()  # frozen: no dropout or drop-path
        return self

    def forward(self, x):
        gray = (x[:, :1] - MEAN) / STD  # the detector's 3 channels are the same grey image
        with torch.no_grad():
            tokens = self.terramind({'S1GRD': gray.repeat(1, 2, 1, 1)})[-1]  # [B, N, 768], encoder norm applied
        side = round(tokens.shape[1] ** 0.5)
        self.tokens = tokens.transpose(1, 2).reshape(len(x), -1, side, side)
        return self.layer(x)


class TerraMindFusion(nn.Module):
    """A backbone layer's output plus a 1x1 projection of the stem's TerraMind tokens (zero at the start)."""

    def __init__(self, layer: nn.Module, stem: TerraMindStem, channels: int, last: bool):
        super().__init__()
        self.layer = layer
        self.proj = nn.Conv2d(768, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.source = [stem]  # in a list, so the stem (and TerraMind) is not registered a second time
        self.last = last
        for key in LAYER_KEYS:
            setattr(self, key, getattr(layer, key))

    def forward(self, x):
        y = self.layer(x)
        stem = self.source[0]
        tokens = stem.tokens.to(self.proj.weight.dtype)
        if tokens.shape[-2:] != y.shape[-2:]:
            tokens = F.interpolate(tokens, size=y.shape[-2:], mode='bilinear', align_corners=False)
        if self.last:
            stem.tokens = None
        return y + self.proj(tokens)


def unwrap(layer: nn.Module) -> nn.Module:
    """The plain detector layer inside a TerraMind wrapper (for code that runs backbone layers on their own)."""
    return layer.layer if isinstance(layer, (TerraMindStem, TerraMindFusion)) else layer


def attach(model: nn.Module, pretrained: bool = True) -> nn.Module:
    """Wrap an Ultralytics DetectionModel's first layer and FUSE_AT layers in place; returns the model."""
    layers = model.model
    if isinstance(layers[0], TerraMindStem):
        return model
    device, was_training = next(model.parameters()).device, model.training
    channels = {}
    hooks = [layers[i].register_forward_hook(lambda m, _in, out, i=i: channels.__setitem__(i, out.shape[1]))
             for i in FUSE_AT]
    model.eval()  # eval: the probe image must not touch BatchNorm statistics
    with torch.no_grad():
        model(torch.zeros(1, 3, 256, 256, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    stem = TerraMindStem(layers[0], pretrained).to(device)
    layers[0] = stem
    for k, i in enumerate(FUSE_AT):
        layers[i] = TerraMindFusion(layers[i], stem, channels[i], last=k == len(FUSE_AT) - 1).to(device)
    stem.train(was_training)
    return model


def trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable, frozen) parameter counts, TerraMind counted as frozen."""
    frozen = sum(p.numel() for n, p in model.named_parameters() if '.terramind.' in n)
    return sum(p.numel() for p in model.parameters()) - frozen, frozen


def _trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class TerraMindTrainer(DetectionTrainer):
        """Builds the detector as usual (YAML + weights), then adds the frozen TerraMind and the zero fusion layers."""

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = attach(super().get_model(cfg, weights, verbose))
            train_params, frozen = trainable(model)
            print(f'TerraMind fusion: {train_params:,} trainable detector parameters, {frozen:,} frozen TerraMind '
                  f'parameters, {train_params + frozen:,} in all', flush=True)

            def refreeze(trainer):  # the trainer re-enables gradients on every parameter it does not freeze itself
                for name, p in trainer.model.named_parameters():
                    if '.terramind.' in name:
                        p.requires_grad_(False)

            self.add_callback('on_pretrain_routine_end', refreeze)
            return model

    return TerraMindTrainer


def __getattr__(name):  # TerraMindTrainer imports Ultralytics' trainer only when asked for
    if name == 'TerraMindTrainer':
        return _trainer()
    raise AttributeError(name)
