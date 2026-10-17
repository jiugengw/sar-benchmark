custom_hooks = [
    dict(type='FiniteLossHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'groksar.models',
        'denodet_baseline.model',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        img_scale=(
            512,
            512,
        ),
        init_cfg=dict(
            checkpoint=
            '/common/scratch/users/j/jgwong.2023/sar-benchmark/pretrained/resnet50-0676ba61.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        partition_stride=(
            8,
            8,
        ),
        style='pytorch',
        type='groksar.FFTResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_dfl=dict(loss_weight=0.25, type='DistributionFocalLoss'),
        num_classes=9,
        reg_max=16,
        stacked_convs=4,
        type='GFLHead'),
    classification_weight=1.0,
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            64.49589435410869,
            64.49589435410869,
            64.49589435410869,
        ],
        pad_size_divisor=32,
        std=[
            54.51271241987604,
            54.51271241987604,
            54.51271241987604,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    num_image_classes=9,
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='MultiTaskDenoDet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        decouple=True, lr=1.0, type='DAdaptAdam', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=48,
        gamma=0.1,
        milestones=[
            32,
            44,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=0)
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/dataset/val.json',
        data_prefix=dict(img=''),
        data_root=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/data/yolo11s-sar-512',
        filter_cfg=None,
        metainfo=dict(
            classes=(
                'aircraft',
                'airport',
                'bridge',
                'car',
                'harbor',
                'oil_tank',
                'playground',
                'ship',
                'wind_turbine',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/dataset/val.json',
        classwise=True,
        metric='bbox',
        outfile_prefix=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/holdout_512',
        type='CocoMetric'),
    dict(type='ImageClassificationMetric'),
]
train_cfg = dict(max_epochs=48, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/dataset/train.json',
        data_prefix=dict(img=''),
        data_root=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/data/yolo11s-sar-512',
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        metainfo=dict(
            classes=(
                'aircraft',
                'airport',
                'bridge',
                'car',
                'harbor',
                'oil_tank',
                'playground',
                'ship',
                'wind_turbine',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        test_mode=False,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/dataset/val.json',
        data_prefix=dict(img=''),
        data_root=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/data/yolo11s-sar-512',
        filter_cfg=None,
        metainfo=dict(
            classes=(
                'aircraft',
                'airport',
                'bridge',
                'car',
                'harbor',
                'oil_tank',
                'playground',
                'ship',
                'wind_turbine',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/dataset/val.json',
        classwise=True,
        metric='bbox',
        outfile_prefix=
        '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477/holdout_512',
        type='CocoMetric'),
    dict(type='ImageClassificationMetric'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/common/scratch/users/j/jgwong.2023/sar-benchmark/runs/denodet-v2-329477'
