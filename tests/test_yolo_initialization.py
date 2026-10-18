"""Exercise the actual Ultralytics Model.train handoff without training or a GPU."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from models.yolo.train_yolo26s_p2 import parameter_fingerprint, preserve_initialization


class HandoffReached(Exception):
    pass


def test_transferred_values_survive_actual_wrapper_train(tmp_path):
    torch.set_num_threads(1)
    config = yaml.safe_load(Path('.venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26.yaml').read_text())
    config.update(scale='n', nc=9)
    path = tmp_path / 'model.yaml'
    path.write_text(yaml.safe_dump(config))
    model = YOLO(str(path))
    with torch.no_grad():
        model.model.model[0].conv.weight.fill_(.125)
    expected = parameter_fingerprint(model.model)
    audit = preserve_initialization(model, tmp_path / 'audit.json')

    class ProbeTrainer:
        def __init__(self, overrides, _callbacks):
            assert overrides['pretrained'] is True

        def get_model(self, cfg, weights):
            assert weights is not None, 'Wrapper discarded initialization'
            rebuilt = DetectionModel(cfg=deepcopy(cfg), nc=9, verbose=False)
            rebuilt.load(weights, verbose=False)
            return rebuilt

        def train(self):
            assert parameter_fingerprint(self.model) == expected
            for callback in model.callbacks['on_pretrain_routine_end']:
                callback(self)
            assert audit['training_handoff_verified'] is True
            with torch.no_grad():
                self.model.model[0].conv.weight.zero_()
            with pytest.raises(RuntimeError, match='discarded or changed initialization'):
                model.callbacks['on_pretrain_routine_end'][-1](self)
            raise HandoffReached

    with pytest.raises(HandoffReached):
        model.train(trainer=ProbeTrainer, pretrained=True, data='data/yolo11s-sar-512/dataset.yaml')
