"""Cascade R-CNN (Detectron2) on our SAR split: a different detector family, built for high-IoU boxes.

Why this family: our largest loss is at the strict IoU thresholds. Cascade R-CNN refines every box three times, each
stage trained at a stricter IoU (0.5, 0.6, 0.7), so later stages specialise in tight boxes. Its FPN includes the
stride-4 level P2 for proposals and box pooling, which our tiny objects need.

Setup (within the rules: <= 40M trainable, <= 130M total, input <= 512, generic optical pretraining):
- Detectron2 "Misc/cascade_mask_rcnn_R_50_FPN_3x" COCO weights, masks off; ResNet-50 + FPN + RPN transfer.
- Box heads 256 wide instead of 1024 (37.0M trainable instead of 68.9M); they and the 9-class outputs start fresh.
- Every image resized to 512 on its shorter side and at most 512 (256 px sources are upscaled to 512, which the TA
  allowed); horizontal and vertical flips while training; then `--clean-iters` final iterations without flips (the
  "memorise" lesson: exact geometry at the end).
- Train on the 8,453 training images, evaluate the 939-image holdout, and write results/<tag>/predictions_val.json
  plus instances_val_holdout.json in the layout of the YOLO runs (sarbench.twins.score, official_mix_score.py).

Detectron2 lives outside the project venv (PYTHONPATH=~/scratchDirectory/cascade-libs), so it cannot change the
packages other jobs use.

    python models/cascade/train.py --tag cascade-r50 --iters 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / 'data/course-data/train'
SPLIT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json'
HOLDOUT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json'
CONFIG = 'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'


def write_split_jsons(out: Path) -> tuple[Path, Path]:
    """COCO files for the training split and the holdout, both pointing at data/course-data/train/images."""
    coco = json.loads((DATA / 'instances.json').read_text())
    holdout_ids = {im['id'] for im in json.loads(HOLDOUT.read_text())['images']}
    train_ids = set(map(int, json.loads(SPLIT.read_text())['train_image_ids']))
    if train_ids & holdout_ids or len(train_ids) != 8453 or len(holdout_ids) != 939:
        raise ValueError('unexpected split')
    paths = []
    for name, ids in (('train', train_ids), ('holdout', holdout_ids)):
        subset = {'images': [im for im in coco['images'] if im['id'] in ids],
                  'annotations': [a for a in coco['annotations'] if a['image_id'] in ids],
                  'categories': coco['categories']}
        path = out / f'instances_{name}.json'
        path.write_text(json.dumps(subset))
        paths.append(path)
    return paths[0], paths[1]


def build_cfg(args, out: Path):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
    cfg.MODEL.WEIGHTS = str(args.weights) if args.weights else model_zoo.get_checkpoint_url(CONFIG)
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = args.fc_dim
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.INPUT.RANDOM_FLIP = 'none'  # flips are added by the trainer below (both directions, until the clean phase)
    cfg.DATASETS.TRAIN = ('sar_train',)
    cfg.DATASETS.TEST = ('sar_holdout',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = 0.02 * args.batch / 16
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.STEPS = (int(args.iters * 0.75), int(args.iters * 0.9))
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = args.iters  # last checkpoint only (cluster file limit)
    cfg.SOLVER.AMP.ENABLED = True
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SEED = 0
    cfg.OUTPUT_DIR = str(out)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='runs/<tag>, results/<tag>')
    parser.add_argument('--iters', type=int, default=20000, help='about 528 iterations per epoch at batch 16')
    parser.add_argument('--clean-iters', type=int, default=3000, help='final iterations without flips')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--fc-dim', type=int, default=256, help='box-head width; 256 keeps trainable parameters < 40M')
    parser.add_argument('--eval-period', type=int, default=2000)
    parser.add_argument('--weights', type=Path, help='start from these weights instead of the COCO model-zoo ones')
    parser.add_argument('--smoke', action='store_true', help='50 iterations and a 50-image holdout, to test the setup')
    args = parser.parse_args()
    if args.smoke:
        args.iters, args.clean_iters, args.eval_period = 50, 20, 0

    import torch
    from detectron2.data import DatasetMapper, build_detection_train_loader, transforms as T
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, HookBase
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.utils.logger import setup_logger

    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required')
    run = ROOT / 'runs' / args.tag
    out = ROOT / 'results' / args.tag
    run.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    setup_logger(str(run / 'log.txt'))
    train_json, holdout_json = write_split_jsons(run)
    if args.smoke:
        small = json.loads(holdout_json.read_text())
        keep = {im['id'] for im in small['images'][:50]}
        small['images'] = [im for im in small['images'] if im['id'] in keep]
        small['annotations'] = [a for a in small['annotations'] if a['image_id'] in keep]
        holdout_json.write_text(json.dumps(small))
    register_coco_instances('sar_train', {}, str(train_json), str(DATA))
    register_coco_instances('sar_holdout', {}, str(holdout_json), str(DATA))
    cfg = build_cfg(args, run)
    clean_from = args.iters - args.clean_iters

    def augmentations(flips: bool):
        augs = [T.ResizeShortestEdge(512, 512, 'choice')]
        if flips:
            augs += [T.RandomFlip(horizontal=True, vertical=False), T.RandomFlip(horizontal=False, vertical=True)]
        return augs

    class Trainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                          augmentations=augmentations(True)))

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, output_dir=str(run / 'eval'))

    class CleanPhase(HookBase):
        """Switch the training loader to no flips for the last `clean_iters` iterations."""
        def before_step(self):
            if self.trainer.iter == clean_from and args.clean_iters > 0:
                loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                                augmentations=augmentations(False)))
                self.trainer._trainer._data_loader_iter_obj = iter(loader)
                self.trainer._trainer.data_loader = loader
                print(f'CLEAN_PHASE from iteration {self.trainer.iter}: flips off', flush=True)

    trainer = Trainer(cfg)
    trainer.register_hooks([CleanPhase()])
    trainer.resume_or_load(resume=False)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f'PARAMETERS trainable {trainable / 1e6:.1f}M total {total / 1e6:.1f}M', flush=True)
    if trainable > 40e6:
        raise RuntimeError('over the 40M trainable-parameter limit')
    started = time.time()
    trainer.train()

    evaluator = COCOEvaluator('sar_holdout', output_dir=str(run / 'eval_final'))
    metrics = inference_on_dataset(trainer.model, build_detection_test_loader(cfg, 'sar_holdout'), evaluator)
    predictions = json.loads((run / 'eval_final' / 'coco_instances_results.json').read_text())
    (out / 'predictions_val.json').write_text(json.dumps(predictions))
    (out / 'instances_val_holdout.json').write_text(holdout_json.read_text())
    (out / 'metrics.json').write_text(json.dumps({'detectron2': metrics, 'wall_seconds': time.time() - started,
                                                  'trainable_parameters': trainable, 'total_parameters': total,
                                                  'settings': vars(args) | {'weights': str(args.weights)}},
                                                 indent=2, default=str) + '\n')
    print(f'CASCADE_COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
