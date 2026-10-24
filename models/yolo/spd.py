"""SPD (space-to-depth) first layer for YOLO: keep every pixel instead of striding over them, at nearly normal cost.

A stride-2 convolution reads only every other position, so detail between those positions is lost before any
feature is learned (Sunkara and Luo, "No More Strided Convolutions or Pooling", 2022). SPD first rearranges every
2x2 block of pixels into 4x the channels (half the width and height, nothing dropped), then applies a stride-1
convolution. The output has the same size as the strided layer's, so the rest of the network is unchanged.

Ultralytics builds models from YAML and treats 'Conv' specially when counting channels. SPDConv is a Conv subclass
that behaves exactly like Conv unless its stride is written as -2 in the YAML: then it is SPD + stride-1 Conv. While
an SPD model is built or rebuilt (the trainer rebuilds it from its YAML), install() makes the YAML name 'Conv' refer
to SPDConv; every other layer stays a plain convolution.

exact_copy() initialises the SPD layer from a pretrained stride-2 layer so that it computes EXACTLY the same output:
the pretrained 3x3 kernel over pixels is rewritten as a 3x3 kernel over the 2x2 blocks (each weight moved to the
slot of its pixel, zeros elsewhere), so training starts from the pretrained model and can only add detail.
"""
from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


class SPDConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.spd = s == -2
        super().__init__(c1 * 4 if self.spd else c1, c2, k, 1 if self.spd else s, p, g, d, act)
        if self.spd:
            self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        return super().forward(self.unshuffle(x) if self.spd else x)

    def forward_fuse(self, x):
        return super().forward_fuse(self.unshuffle(x) if self.spd else x)


def install() -> None:
    """Make the YAML layer name 'Conv' build SPDConv (identical to Conv unless the stride is -2)."""
    from ultralytics.nn import tasks

    tasks.Conv = SPDConv


@torch.no_grad()
def exact_copy(spd_layer: SPDConv, strided_layer: Conv) -> None:
    """Rewrite a pretrained 3x3, stride-2, padding-1 Conv as the equivalent kernel of an SPD layer (3x3 over 2x2 blocks).

    Output (y, x) of the strided conv reads pixel rows 2y-1, 2y, 2y+1. In block coordinates (row = 2Y + i) those are
    (Y=y-1, i=1), (Y=y, i=0), (Y=y, i=1), i.e. kernel rows KY=0,1,1 of a padding-1 3x3 kernel centred on block y.
    PixelUnshuffle puts pixel (i, j) of channel c in channel c*4 + i*2 + j."""
    w = strided_layer.conv.weight  # (out, c, 3, 3)
    if w.shape[2:] != (3, 3) or strided_layer.conv.stride != (2, 2) or strided_layer.conv.padding != (1, 1):
        raise ValueError('exact_copy needs a 3x3, stride-2, padding-1 convolution')
    v = torch.zeros_like(spd_layer.conv.weight)  # (out, 4c, 3, 3)
    if v.shape[1] != 4 * w.shape[1] or v.shape[2:] != (3, 3):
        raise ValueError(f'unexpected SPD kernel shape {tuple(v.shape)}')
    place = {0: (0, 1), 1: (1, 0), 2: (1, 1)}  # kernel offset -> (block kernel index, pixel offset inside the block)
    for ky, (KY, i) in place.items():
        for kx, (KX, j) in place.items():
            for c in range(w.shape[1]):
                v[:, c * 4 + i * 2 + j, KY, KX] = w[:, c, ky, kx]
    spd_layer.conv.weight.copy_(v)
    spd_layer.bn.load_state_dict(strided_layer.bn.state_dict())
