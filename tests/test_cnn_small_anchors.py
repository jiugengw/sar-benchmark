"""Check native CNN features, uneven anchor counts, and checkpoint reconstruction."""
import torch
from torch import nn

from sarbench.backbones import ResNet34
from sarbench.model import MultiTaskModel, VariableAnchorRPNHead


def test_small_anchors_have_matching_prediction_counts():
    backbone = nn.Module()
    backbone.embed_dim = 768
    model = MultiTaskModel(backbone, False, anchor_profile='small_extreme')
    assert model.rpn.anchor_generator.sizes == ((6, 8, 11), (16, 23), (32, 45), (64, 91), (128, 181))
    assert model.rpn.anchor_generator.num_anchors_per_location() == [15, 10, 10, 10, 10]
    assert model.rpn.anchor_generator.aspect_ratios == ((0.25, 0.5, 1.0, 2.0, 4.0),) * 5
    assert isinstance(model.rpn.head, VariableAnchorRPNHead)
    shapes = [16, 8, 4, 2, 1]
    scores, boxes = model.rpn.head([torch.randn(2, 256, size, size) for size in shapes])
    for size, count, score, box in zip(shapes, [15, 10, 10, 10, 10], scores, boxes):
        assert score.shape == (2, count, size, size)
        assert box.shape == (2, count * 4, size, size)


def test_resnet_joint_training_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = MultiTaskModel(ResNet34(False), False, anchor_profile='small_extreme')
    assert sum(p.numel() for p in model.parameters()) < 40_000_000
    images = torch.randn(2, 1, 64, 64)
    targets = [{'boxes': torch.tensor([[12., 12., 30., 30.]]), 'labels': torch.tensor([1])} for _ in range(2)]
    model.train()
    logits, _, losses = model(images, targets)
    loss = nn.functional.cross_entropy(logits, torch.zeros(2, dtype=torch.long)) + sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.body.conv1.weight.grad is not None
    assert model.rpn.head.cls_logits['15'].weight.grad is not None
    assert model.rpn.head.cls_logits['10'].weight.grad is not None
    model.eval()
    with torch.no_grad():
        expected, detections, _ = model(images)
    assert len(detections) == 2
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = MultiTaskModel(ResNet34(False), False, anchor_profile='small_extreme').eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        actual, _, _ = restored(images)
    torch.testing.assert_close(actual, expected)
