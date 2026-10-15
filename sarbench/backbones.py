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


class ResNet34(nn.Module):
    """ImageNet-1k ResNet-34; native stride-4/8/16/32 maps, no classifier."""

    def __init__(self, pretrained, weights_path=None):
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34
        from torchvision.models._utils import IntermediateLayerGetter

        network = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        if weights_path is not None:
            network.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
        rgb_weight = network.conv1.weight.detach().sum(dim=1, keepdim=True)
        network.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            network.conv1.weight.copy_(rgb_weight)
        self.body = IntermediateLayerGetter(network, return_layers={
            'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'})
        self.feature_channels = [64, 128, 256, 512]
        self.embed_dim = 512

    @property
    def blocks(self):
        return [self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4]

    def forward(self, x):
        return self.body(x)


def build_backbone(name, pretrained, weights_path=None):
    """name: 'vit' or 'terramind'. pretrained=False builds the same architecture with random weights."""
    if weights_path is not None:
        if name == 'resnet34' and not pretrained:
            return ResNet34(pretrained=False, weights_path=weights_path)
        if name != 'vit' or pretrained:
            raise ValueError('Local backbone weights are supported only for ViT with pretrained=False.')
        return ViT(pretrained=False, weights_path=weights_path)
    return {'vit': ViT, 'terramind': TerraMind, 'resnet34': ResNet34}[name](pretrained)
