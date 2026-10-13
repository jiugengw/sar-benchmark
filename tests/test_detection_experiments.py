"""CPU checks for experimental geometry, anchor wiring, and train-only sampling."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import tv_tensors

from make_submission import inference_transform, write_submission
from sarbench.data import ResizePad, SARMultiTask, to_original_xywh
from sarbench.model import MultiTaskModel
from train import make_loader


@pytest.mark.parametrize('height,width', [(100, 200), (200, 100), (73, 219), (512, 512)])
def test_padded_box_round_trip(height, width):
    image = tv_tensors.Image(torch.zeros(1, height, width, dtype=torch.uint8))
    original = torch.tensor([[width * .1, height * .2, width * .8, height * .9]])
    boxes = tv_tensors.BoundingBoxes(original, format='XYXY', canvas_size=(height, width))
    resized_image, resized_boxes = ResizePad()(image, boxes)
    assert resized_image.shape == (1, 512, 512)
    restored = to_original_xywh(resized_boxes, (height, width), geometry='pad')
    expected = original.clone()
    expected[:, 2:] -= expected[:, :2]
    torch.testing.assert_close(restored.as_subclass(torch.Tensor), expected)
    infer = inference_transform('pad')(Image.new('L', (width, height)))
    assert infer.shape == resized_image.shape


def test_padding_predictions_clipped_in_submission(tmp_path):
    path = tmp_path / 'submission.zip'
    images = [{'id': 1, 'file_name': 'a.png', 'width': 200, 'height': 100}]
    predictions = {'a.png': (0, [[0, 0, 512, 256], [0, 300, 20, 400]], [0, 0], [.9, .8])}
    write_submission(images, predictions, out=path, geometry='pad')
    import zipfile
    with zipfile.ZipFile(path) as archive:
        detections = json.loads(archive.read('detection.json'))
    assert len(detections) == 1
    assert detections[0]['bbox'] == [0., 0., 200., 100.]


@pytest.mark.parametrize('profile,count', [('default', 6), ('wide', 10), ('large', 6)])
def test_anchor_profile_matches_rpn_output(profile, count):
    backbone = nn.Module()
    backbone.embed_dim = 768
    model = MultiTaskModel(backbone, False, anchor_profile=profile)
    assert model.rpn.anchor_generator.num_anchors_per_location() == [count] * 5
    assert model.rpn.head.cls_logits.out_channels == count
    assert model.rpn.head.bbox_pred.out_channels == count * 4


def test_sampler_uses_only_training_subset_and_keeps_epoch_length(tmp_path):
    root = tmp_path / 'train'
    root.mkdir()
    with (root / 'labels.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['file_name', 'label'])
        writer.writeheader()
        writer.writerows([{'file_name': f'{i}.png', 'label': label}
                          for i, label in enumerate([0, 0, 0, 1, 8])])
    (root / 'instances.json').write_text(json.dumps({'images': [], 'annotations': []}))
    args = SimpleNamespace(data=tmp_path, limit=None, geometry='stretch',
                           class_sampling='sqrt-balanced', seed=0, batch_size=2, workers=0)
    train = make_loader(args, [0, 1, 2, 3], True)
    assert len(train.sampler) == 4
    assert set(int(row['label']) for row in train.dataset.rows) == {0, 1}
    assert train.sampler.weights[3] > train.sampler.weights[0]
    val = make_loader(args, [4], False)
    assert not isinstance(val.sampler, torch.utils.data.WeightedRandomSampler)


def test_training_and_submission_geometry_match(tmp_path):
    root = tmp_path / 'train'
    root.mkdir()
    image = Image.new('L', (200, 100), color=127)
    image.save(root / 'a.png')
    (root / 'labels.csv').write_text('file_name,label\na.png,0\n')
    (root / 'instances.json').write_text(json.dumps({
        'images': [{'id': 1, 'file_name': 'a.png', 'width': 200, 'height': 100}],
        'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40]}]}))
    for geometry in ('stretch', 'pad'):
        dataset = SARMultiTask(tmp_path, geometry=geometry)
        tensor, _, target = dataset[0]
        torch.testing.assert_close(tensor, inference_transform(geometry)(image))
        restored = to_original_xywh(target['boxes'], target['orig_size'], geometry=geometry)
        torch.testing.assert_close(restored.as_subclass(torch.Tensor), torch.tensor([[10., 20., 30., 40.]]))
