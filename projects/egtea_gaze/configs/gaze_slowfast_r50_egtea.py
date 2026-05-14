import os

_base_ = [
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.egtea_gaze.egtea_gaze.datasets.transforms.load_gaze',
        'projects.egtea_gaze.egtea_gaze.models.losses.gaze_losses',
        'projects.egtea_gaze.egtea_gaze.models.heads.gaze_slowfast_head',
        'projects.egtea_gaze.egtea_gaze.models.recognizers.gaze_recognizer',
    ],
    allow_failed_imports=False)

gaze_map_root = os.getenv('EGTEA_GAZE_MAP_ROOT', '/root/data/egtea/gaze_maps')
gaze_metadata_file = os.path.join(gaze_map_root, 'metadata.json')
slowfast_init_ckpt = os.getenv(
    'EGTEA_SLOWFAST_CKPT',
    '/root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep/best_acc_top1_epoch_24.pth')

model = dict(
    type='GazeRecognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='GazeSlowFastHead',
        num_classes=106,
        in_channels=(2048, 256),
        mid_channels=256,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        attention_size=(14, 14),
        gaze_loss=dict(type='GazeKLLoss', loss_weight=0.1),
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

dataset_type = 'VideoDataset'
data_root = '/root/data/egtea/videos'
data_root_val = '/root/data/egtea/videos'
ann_file_train = '/root/data/egtea/action_annotation/train.txt'
ann_file_val = '/root/data/egtea/action_annotation/val.txt'
ann_file_test = '/root/data/egtea/action_annotation/test.txt'
file_client_args = dict(io_backend='disk')
algorithm_keys = ('gaze_maps', 'gaze_valid', 'gaze_xy', 'gaze_source')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='LoadGazeMap',
        gaze_map_root=gaze_map_root,
        metadata_file=gaze_metadata_file,
        heatmap_size=(14, 14),
        sigma=1.5,
        missing_policy='zeros',
        gaze_mode='real',
        cache_size=256),
    dict(type='PackActionInputs', algorithm_keys=algorithm_keys),
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(
    type='AccMetric',
    metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='acc/top1',
        max_keep_ckpts=3),
    logger=dict(interval=20))

env_cfg = dict(cudnn_benchmark=True)
load_from = slowfast_init_ckpt

