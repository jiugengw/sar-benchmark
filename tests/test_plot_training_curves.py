"""tools/plot_training_curves.py: history loaders and run.json parent inheritance."""
import json

import pytest

from tools import plot_training_curves as curves


def test_mmengine_history_marks_missing_validation_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    (tmp_path / 'history.json').write_text(json.dumps([
        {'epoch': 48, 'train_loss': 0.37, 'mAP': 0.374, 'AP50': 0.623, 'macro_f1': 0.9375}]))
    history = curves.load({'type': 'mmengine', 'path': 'history.json'})
    assert history['epochs'] == [48]
    assert history['losses']['logged training loss'] == ([0.37], [None])
    assert history['metrics']['Δm'] == [pytest.approx(curves.delta_m(0.9375, 0.374))]
    assert 'validation loss was not recorded' in history['note']


def test_ultralytics_history_keeps_train_and_validation_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    (tmp_path / 'results.csv').write_text(
        'epoch,train/box_loss,val/box_loss,metrics/mAP50-95(B),metrics/mAP50(B)\n1,2.0,2.5,0.1,0.2\n2,1.5,2.4,0.2,0.4\n')
    history = curves.load({'type': 'ultralytics', 'path': 'results.csv'})
    assert history['epochs'] == [1, 2]
    assert history['losses']['box_loss'] == ([2.0, 1.5], [2.5, 2.4])
    assert history['metrics']['mAP50-95'] == [0.1, 0.2]


def test_terramind_history_computes_delta_m(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    (tmp_path / 'metrics.json').write_text(json.dumps({'val_history': [
        {'epoch': 4, 'mAP': 0.2613, 'AP50': 0.5, 'macro_f1': 0.9331}]}))
    history = curves.load({'type': 'terramind', 'path': 'metrics.json'})
    assert history['metrics']['Δm'] == [pytest.approx(0.0)]
    assert history['losses'] == {}


def test_package_inherits_parent_histories(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    (tmp_path / 'parent.run.json').write_text(json.dumps({'zip': 'a.zip', 'histories': [{'type': 'ultralytics', 'path': 'r.csv'}]}))
    (tmp_path / 'child.run.json').write_text(json.dumps({'zip': 'b.zip', 'parent': 'parent.run.json', 'histories': []}))
    assert curves.resolve(tmp_path / 'child.run.json')['histories'] == [{'type': 'ultralytics', 'path': 'r.csv'}]
