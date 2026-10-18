"""Context boxes (experiment A) and the water input filter (experiment B)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

from models.yolo.build_variant_dataset import enlarge_line  # noqa: E402
from models.yolo.context_boxes import shrink  # noqa: E402
from models.yolo.input_filters import active_filter, apply  # noqa: E402


def test_enlarge_keeps_centre_and_other_classes():
    assert enlarge_line('0 0.5 0.5 0.1 0.1', {2}, 3) == '0 0.5 0.5 0.1 0.1'
    cls, cx, cy, w, h = enlarge_line('2 0.5 0.4 0.02 0.04', {2}, 3).split()
    assert (cls, float(cx), float(cy)) == ('2', 0.5, 0.4)
    assert float(w) == pytest.approx(0.06) and float(h) == pytest.approx(0.12)


def test_enlarge_stops_at_the_border_without_moving_the_centre():
    _, cx, cy, w, h = map(float, enlarge_line('2 0.02 0.5 0.02 0.02', {2}, 3).split())
    assert cx == 0.02 and w == pytest.approx(0.04)  # 3x would cross the left edge; centre stays put
    assert cx - w / 2 >= 0 and h == pytest.approx(0.06)


def test_shrink_undoes_enlarge_for_the_chosen_class_only():
    predictions = [{'image_id': 1, 'category_id': 3, 'bbox': [100, 200, 30, 60], 'score': 0.9},
                   {'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 60], 'score': 0.8}]
    out = shrink(predictions, {3}, 3)
    assert out[0]['bbox'] == pytest.approx([110, 220, 10, 20])  # same centre (115, 230), a third of the size
    assert out[1]['bbox'] == [10, 20, 30, 60]


def test_water_filter_marks_dark_smooth_regions():
    gray = np.full((128, 128), 140, np.uint8)
    rng = np.random.default_rng(0)
    gray = np.clip(gray + rng.normal(0, 25, gray.shape), 0, 255).astype(np.uint8)
    gray[:, 60:70] = 15  # a dark river
    out = apply('water', gray)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    assert (out[..., 0] == gray).all()  # channel 0 is the untouched SAR image
    assert out[64, 65, 2] > 200 and out[64, 20, 2] < 60  # river marked, land not
    assert np.std(out[..., 1].astype(float)) < np.std(gray.astype(float))  # channel 1 is despeckled


def test_no_filter_is_three_copies_and_env_is_checked(monkeypatch):
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert (apply('none', gray) == gray[..., None]).all()
    monkeypatch.delenv('SAR_INPUT_FILTER', raising=False)
    assert active_filter() == 'none'
    monkeypatch.setenv('SAR_INPUT_FILTER', 'water')
    assert active_filter() == 'water'
    monkeypatch.setenv('SAR_INPUT_FILTER', 'typo')
    with pytest.raises(ValueError):
        active_filter()


def make_source(root: Path) -> Path:
    src = root / 'src'
    names = ['aircraft', 'airport', 'bridge']
    for split in ('train', 'val'):
        (src / 'images' / split).mkdir(parents=True)
        (src / 'labels' / split).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((32, 32), 80 + i, np.uint8), 'L').save(src / 'images' / split / f'{i}.png')
            (src / 'labels' / split / f'{i}.txt').write_text('2 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.2 0.2\n')
    (src / 'conversion_manifest.json').write_text(json.dumps({'classes': names}))
    (src / 'image_map.json').write_text('{}')
    return src


def test_build_context_and_filter_variants(tmp_path, monkeypatch):
    from models.yolo import build_variant_dataset as builder

    src = make_source(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--out', str(tmp_path / 'ctx'), '--source', str(src),
                                      '--context-classes', 'bridge', '--context-factor', '3'])
    builder.main()
    lines = (tmp_path / 'ctx/labels/train/0.txt').read_text().splitlines()
    assert lines[1] == '0 0.3 0.3 0.2 0.2' and float(lines[0].split()[3]) == pytest.approx(0.3)
    assert os.path.islink(tmp_path / 'ctx/images/train')  # images linked, not copied
    assert 'path: ' + str((tmp_path / 'ctx').resolve()) in (tmp_path / 'ctx/dataset.yaml').read_text()

    monkeypatch.setattr(sys, 'argv', ['x', '--out', str(tmp_path / 'wet'), '--source', str(src), '--filter', 'water'])
    builder.main()
    with Image.open(tmp_path / 'wet/images/val/1.png') as image:
        assert image.mode == 'RGB' and image.size == (32, 32)
    assert os.path.islink(tmp_path / 'wet/labels/val')  # labels linked
