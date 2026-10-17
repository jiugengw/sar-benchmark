"""Build the course config from the pinned authors' GFL + FFTResNet config."""
import copy
import runpy
from pathlib import Path


def build_config(root, dataset, metadata, output, weights, epochs, batch, workers):
    from mmengine.config import Config
    source = root / "third_party/GrokSAR/configs/DenoDetV2/DenoDetV2_1x_SARDet_100k.py"
    model = copy.deepcopy(runpy.run_path(str(source))["model"])
    model.update(type="MultiTaskDenoDet", num_image_classes=9, classification_weight=1.0)
    model["backbone"].update(norm_cfg=dict(type="BN", requires_grad=True),
                              init_cfg=dict(type="Pretrained", checkpoint=str(weights.resolve())))
    model["bbox_head"]["num_classes"] = 9
    model["data_preprocessor"].update(mean=[metadata["mean"]] * 3, std=[metadata["std"]] * 3)
    model["test_cfg"]["score_thr"] = 0.001
    train_pipeline = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True),
                      dict(type="RandomFlip", prob=0.5), dict(type="PackDetInputs")]
    test_pipeline = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True),
                     dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]
    # The reused PNGs are already 512x512; Resize sets scale_factor for postprocessing.
    for pipeline in [train_pipeline, test_pipeline]:
        pipeline.insert(2, dict(type="Resize", scale=(512, 512), keep_ratio=False))
    def loader(name, train):
        return dict(batch_size=batch, num_workers=workers, persistent_workers=workers > 0,
                    drop_last=False, sampler=dict(type="DefaultSampler", shuffle=train),
                    dataset=dict(type="CocoDataset", data_root=metadata["image_root"],
                                 ann_file=str((dataset / f"{name}.json").resolve()),
                                 data_prefix=dict(img=""), metainfo=dict(classes=tuple(metadata["classes"])),
                                 filter_cfg=dict(filter_empty_gt=True, min_size=1) if train else None,
                                 test_mode=not train, pipeline=train_pipeline if train else test_pipeline))
    evaluator = [dict(type="CocoMetric", ann_file=str((dataset / "val.json").resolve()), metric="bbox",
                      classwise=True, outfile_prefix=str(output / "holdout_512")),
                 dict(type="ImageClassificationMetric")]
    return Config(dict(
        default_scope="mmdet", model=model, work_dir=str(output.resolve()),
        custom_imports=dict(imports=["groksar.models", "denodet_baseline.model"], allow_failed_imports=False),
        train_dataloader=loader("train", True), val_dataloader=loader("val", False),
        test_dataloader=loader("val", False), val_evaluator=evaluator, test_evaluator=evaluator,
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=epochs, val_interval=1),
        val_cfg=dict(type="ValLoop"), test_cfg=dict(type="TestLoop"),
        optim_wrapper=dict(type="OptimWrapper", optimizer=dict(type="DAdaptAdam", lr=1.0, weight_decay=0.05, decouple=True),
                           paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
                           clip_grad=dict(max_norm=35, norm_type=2)),
        param_scheduler=[dict(type="LinearLR", start_factor=1/3, by_epoch=False, begin=0, end=500),
                         dict(type="MultiStepLR", by_epoch=True, begin=0, end=epochs,
                              milestones=sorted(set([max(1, epochs*2//3), max(2, epochs*11//12)])), gamma=0.1)],
        default_hooks=dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50),
                           param_scheduler=dict(type="ParamSchedulerHook"), sampler_seed=dict(type="DistSamplerSeedHook"),
                           checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2,
                                           save_best="coco/bbox_mAP", rule="greater")),
        custom_hooks=[dict(type="FiniteLossHook")],
        env_cfg=dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
                     dist_cfg=dict(backend="nccl")),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        log_level="INFO", launcher="none", randomness=dict(seed=0, deterministic=False),
        visualizer=dict(type="DetLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")], name="visualizer"),
        load_from=None, resume=False))
