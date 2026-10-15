"""Kendall-style learned weighting of two composite training objectives."""
import torch
from torch import nn


class TaskUncertaintyWeighting(nn.Module):
    """Sum exp(-s_i) L_i + s_i/2 for classification and composite detection.

    s_i is a learned log variance, initialized to zero (unit loss weights).
    Detection contains several likelihood types, so treating its sum as one
    objective is a practical adaptation, not an exact joint likelihood model.
    This module is used only in training, never in prediction.
    """

    def __init__(self):
        super().__init__()
        self.log_variances = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, classification, detection):
        tasks = torch.stack((classification.float(), detection.float()))
        return (self.log_variances.neg().exp() * tasks + 0.5 * self.log_variances).sum()

    @torch.no_grad()
    def values(self):
        variances = self.log_variances.detach().cpu().tolist()
        weights = self.log_variances.detach().neg().exp().cpu().tolist()
        return {'classification_log_variance': variances[0], 'detection_log_variance': variances[1],
                'classification_weight': weights[0], 'detection_weight': weights[1]}
