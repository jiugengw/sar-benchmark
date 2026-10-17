"""sarbench/twins.py: twin detection on synthetic speckle images, and the clean holdout subset."""
import numpy as np
import pytest
from PIL import Image

from sarbench.twins import TWINS_PATH, clean_image_ids, find_twins, load_twins, thumbnails


def test_same_scene_with_new_speckle_is_a_twin_of_the_same_class_only(tmp_path):
    rng = np.random.default_rng(0)
    scene = np.kron(rng.uniform(40, 200, (16, 16)), np.ones((32, 32)))  # 512 x 512 blocky scene
    other = np.kron(rng.uniform(40, 200, (16, 16)), np.ones((32, 32)))
    images = [scene, scene, other, scene]  # the last copy is labeled with another class
    paths = []
    for k, pixels in enumerate(images):
        speckled = np.clip(pixels * rng.gamma(4.0, 0.25, pixels.shape), 0, 255).astype(np.uint8)
        paths.append(tmp_path / f'{k}.png')
        Image.fromarray(speckled).save(paths[-1])
    pairs = find_twins(thumbnails(paths), [1, 1, 1, 2])
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs[0][2] > 0.9


def test_holdout_image_is_clean_unless_its_twin_is_a_training_image():
    twins = {'a': {'b'}, 'b': {'a'}, 'c': {'d'}, 'd': {'c'}}
    holdout = [{'id': 1, 'file_name': 'a'}, {'id': 2, 'file_name': 'c'}, {'id': 3, 'file_name': 'd'},
               {'id': 4, 'file_name': 'e'}]
    assert clean_image_ids(holdout, twins) == [2, 3, 4]  # 'a' has its twin 'b' in training


def test_committed_twin_list_pairs_distinct_images_symmetrically():
    if not TWINS_PATH.is_file():
        pytest.skip('sarbench/train_twins.json not built')
    twins = load_twins()
    assert twins and all(name not in partners for name, partners in twins.items())
    assert all(name in twins[partner] for name, partners in twins.items() for partner in partners)
