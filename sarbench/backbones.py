"""The two ViT-B/16 backbones, built for 512 px input: images [B, 1, 512, 512] -> patch tokens [B, 1024, 768]."""
import timm
import torch
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool, weights_path=None):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=1, num_classes=0)
        if weights_path is not None:
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.vit.load_state_dict(state, strict=True)
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.vit.blocks

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.terramind.encoder

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


def build_backbone(name, pretrained, weights_path=None):
    """name: 'vit' or 'terramind'. pretrained=False builds the same architecture with random weights."""
    if weights_path is not None:
        if name != 'vit' or pretrained:
            raise ValueError('Local backbone weights are supported only for ViT with pretrained=False.')
        return ViT(pretrained=False, weights_path=weights_path)
    return {'vit': ViT, 'terramind': TerraMind}[name](pretrained)
