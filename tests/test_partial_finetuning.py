"""CPU checks for frozen weights, partial-backbone gradients, and optimizer learning rates."""
import pytest
import torch
from torch import nn

from sarbench.adapters import add_adapters, unfreeze_last_blocks
from sarbench.optim import parameter_groups


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(4, 12)
        self.attn.proj = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        x = self.attn.proj(q + k + v)
        return x + self.mlp(x)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(3))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def adapted_model():
    model = nn.Module()
    model.backbone = TinyBackbone()
    add_adapters(model.backbone, 'moelora')
    model.head = nn.Linear(4, 2)
    return model


def test_zero_blocks_keeps_original_weights_frozen():
    model = adapted_model()
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    unfreeze_last_blocks(model.backbone, 0)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def test_partial_training_updates_only_selected_original_weights():
    torch.manual_seed(0)
    model = adapted_model()
    unfreeze_last_blocks(model.backbone, 2)
    first = model.backbone.blocks[0].attn.qkv.base.weight
    last = model.backbone.blocks[-1].attn.qkv.base.weight
    first_before, last_before = first.detach().clone(), last.detach().clone()
    optimizer = torch.optim.SGD(parameter_groups(model, 1e-3, 1e-3, 1e-4))
    model.head(model.backbone(torch.randn(2, 3, 4))).square().sum().backward()
    assert first.grad is None
    assert last.grad is not None and last.grad.abs().sum() > 0
    optimizer.step()
    assert torch.equal(first, first_before)
    assert not torch.equal(last, last_before)


def test_parameter_groups_cover_all_trainable_weights_once_at_correct_rates():
    model = adapted_model()
    unfreeze_last_blocks(model.backbone, 2)
    groups = parameter_groups(model, 1e-4, 1e-4, 2e-5)
    grouped = [id(p) for group in groups for p in group['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    original = model.backbone.blocks[-1].attn.qkv.base.weight
    adapter = model.backbone.blocks[-1].attn.qkv.up.weight
    lr_by_id = {id(p): group['lr'] for group in groups for p in group['params']}
    assert lr_by_id[id(original)] == 2e-5
    assert lr_by_id[id(adapter)] == lr_by_id[id(model.head.weight)] == 1e-4


@pytest.mark.parametrize('count', [-1, 4])
def test_invalid_block_count_is_rejected(count):
    with pytest.raises(ValueError):
        unfreeze_last_blocks(adapted_model().backbone, count)
