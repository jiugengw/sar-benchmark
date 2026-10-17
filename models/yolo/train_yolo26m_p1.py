"""Add a stride-2 (P1) detection level to a trained YOLO26m-P2 detector (default: continue_1) and train 40 epochs.

The settings match the continue_2 control (40 epochs from continue_1, MuSGD, learning rate 0.001, no mosaic), so a
gain over continue_2 (holdout mAP 44.98% full / 41.89% clean) comes from the new level."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16, help='16 matches the control; use 8 if memory runs out')
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'runs/deadline-yolo26m-329644/continue_1/weights/best.pt',
                        help='detector to extend; default the current best (continue_1) so gains stack')
    parser.add_argument('--lr0', type=float, default=0.001, help='0.001 matches the control (continue_2)')
    args = parser.parse_args()
    import torch
    import yaml
    from ultralytics import YOLO
    from models.yolo.deadline_recovery import candidate, write_json
    from models.yolo.train_image_head import ImageLabels

    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = YOLO(str(source_path))
    config = copy.deepcopy(source.model.yaml)
    old_detect = config['head'].pop()
    old_levels = list(old_detect[0])
    first_new_layer = len(config['backbone']) + len(config['head'])
    config['head'] += [
        [old_levels[0], 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 0], 1, 'Concat', [1]],
        [-1, 1, 'C3k2', [128, True]],
        [[first_new_layer + 2, *old_levels], old_detect[1], old_detect[2], old_detect[3]],
    ]
    yaml_path = args.output / 'yolo26m_p1.yaml'
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    target = YOLO(str(yaml_path), task='detect')
    original_layers = len(source.model.model) - 1
    for index in range(original_layers):
        target.model.model[index].load_state_dict(source.model.model[index].state_dict(), strict=True)
    old_head = source.model.model[-1]
    new_head = target.model.model[-1]
    for name in ('cv2', 'cv3', 'one2one_cv2', 'one2one_cv3'):
        old_towers, new_towers = getattr(old_head, name), getattr(new_head, name)
        for index, tower in enumerate(old_towers):
            new_towers[index + 1].load_state_dict(tower.state_dict(), strict=True)
        new_towers[0].load_state_dict(old_towers[0].state_dict(), strict=True)
    if new_head.stride.tolist() != [2., 4., 8., 16., 32.]:
        raise RuntimeError(f'Unexpected P1 pyramid strides: {new_head.stride}')
    total = sum(parameter.numel() for parameter in target.model.parameters())
    if total > 34_000_000:
        raise RuntimeError('Detector leaves insufficient room for the spatial classifier under 40M')
    # Verify on a real image that the inherited four pyramid maps are unchanged.
    import csv
    with (ROOT / 'data/course-data/train/labels.csv').open(newline='') as stream:
        labels = {int(row['image_id']): int(row['label']) for row in csv.DictReader(stream)}
    split = json.loads((ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json').read_text())
    image_map = json.loads((ROOT / 'data/yolo11s-sar-512/image_map.json').read_text())
    dataset = ImageLabels(split['train_image_ids'][:1], labels, image_map, ROOT / 'data/yolo11s-sar-512')
    image = dataset[0][0].unsqueeze(0).cuda()
    captured = {}
    def capture(name):
        def hook(_module, inputs):
            captured[name] = [value.detach().clone() for value in inputs[0]]
        return hook
    source.model.cuda().eval()
    target.model.cuda().eval()
    handles = [old_head.register_forward_pre_hook(capture('source')),
               new_head.register_forward_pre_hook(capture('target'))]
    with torch.inference_mode():
        source.model(image)
        target.model(image)
    for original, inherited in zip(captured['source'], captured['target'][1:]):
        torch.testing.assert_close(original, inherited, rtol=1e-5, atol=1e-5)
    for handle in handles:
        handle.remove()
    write_json(args.output / 'transfer.json', {
        'source': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'copied_backbone_neck_layers': original_layers,
        'copied_detection_levels': 'source P2-P5 copied exactly; new P1 towers initialized from the source P2 towers',
        'new_neck': 'stride-2 upsample plus first backbone feature; C3k2 initialized randomly',
        'strides': new_head.stride.tolist(), 'detector_parameters': total,
        'inherited_feature_equivalence_on_real_training_image': True,
        'pretraining': 'course-trained source detector; original generic COCO pretraining, no external SAR weights',
    })
    del captured, source, old_head, new_head, image
    torch.cuda.empty_cache()
    initial_weights = args.output / 'p1_initialized.pt'
    target.save(str(initial_weights))
    del target
    detector = YOLO(str(initial_weights))
    started = time.monotonic()
    detector.train(
        data=str(ROOT / 'data/yolo11s-sar-512/dataset.yaml'), pretrained=True,
        epochs=args.epochs, imgsz=512, batch=args.batch, workers=4, device=0,
        optimizer='MuSGD', lr0=args.lr0, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True, seed=0, deterministic=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        translate=0.05, scale=0.2, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, multi_scale=0.0,
        amp=True, patience=args.epochs + 1, project=str(args.output), name='train',
        plots=True, save_period=10,
    )
    write_json(args.output / 'compute.json', {
        'wall_seconds': time.monotonic() - started, 'epochs': args.epochs,
        'detector_parameters': total, 'gpu': torch.cuda.get_device_name(0),
    })
    del detector
    torch.cuda.empty_cache()
    result = candidate(args.output / 'train/weights/best.pt', 'p1', args.output, args.output.name)
    write_json(args.output / 'complete.json', result)


if __name__ == '__main__':
    main()
