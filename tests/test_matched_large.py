"""Reject confounded architecture comparisons before spending GPU time."""
import pytest
import yaml

from models.yolo.matched_large import compare_arguments


@pytest.mark.parametrize('changed', [{'lr0': .002}, {'speckle_prob': .5}, {'seed': 1}, {'epochs': 80}])
def test_comparison_rejects_recipe_changes(tmp_path, changed):
    common = {'model': 'medium.pt', 'project': 'medium', 'save_dir': 'medium/train',
              'lr0': .001, 'speckle_prob': 0., 'seed': 0, 'epochs': 40}
    reference, candidate = tmp_path / 'medium.yaml', tmp_path / 'large.yaml'
    reference.write_text(yaml.safe_dump(common))
    candidate.write_text(yaml.safe_dump({**common, 'model': 'large.pt', **changed}))
    with pytest.raises(ValueError, match='Training settings differ'):
        compare_arguments(reference, candidate)


def test_comparison_allows_architecture_and_output_paths(tmp_path):
    reference, candidate = tmp_path / 'medium.yaml', tmp_path / 'large.yaml'
    for path, scale in [(reference, 'medium'), (candidate, 'large')]:
        path.write_text(yaml.safe_dump({'model': f'{scale}.pt', 'project': scale, 'name': f'{scale}_stage',
                                       'save_dir': f'{scale}/train', 'seed': 0, 'lr0': .001}))
    compare_arguments(reference, candidate)
