"""Learned task scales must train both heads and avoid trivial zero weighting."""
import math

import torch

from sarbench.loss_weighting import TaskUncertaintyWeighting


def test_initial_objective_and_gradients():
    module = TaskUncertaintyWeighting()
    classification = torch.tensor(0.2, requires_grad=True)
    detection = torch.tensor(1.5, requires_grad=True)
    objective = module(classification, detection)
    torch.testing.assert_close(objective, classification + detection)
    objective.backward()
    torch.testing.assert_close(classification.grad, torch.tensor(1.))
    torch.testing.assert_close(detection.grad, torch.tensor(1.))
    torch.testing.assert_close(module.log_variances.grad, torch.tensor([0.3, -1.]))


def test_regularizer_and_checkpoint(tmp_path):
    module = TaskUncertaintyWeighting()
    # For fixed positive losses the finite optimum is s_i = log(2 L_i).
    with torch.no_grad():
        module.log_variances.copy_(torch.tensor([math.log(0.4), math.log(3.)]))
    objective = module(torch.tensor(0.2), torch.tensor(1.5))
    objective.backward()
    torch.testing.assert_close(module.log_variances.grad, torch.zeros(2), atol=1e-6, rtol=0)
    path = tmp_path / 'loss_weighting.pt'
    torch.save(module.state_dict(), path)
    restored = TaskUncertaintyWeighting()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.values() == module.values()
    assert sum(p.numel() for p in restored.parameters()) == 2
