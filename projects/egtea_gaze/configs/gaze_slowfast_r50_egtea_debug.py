import os

_base_ = ['./gaze_slowfast_r50_egtea.py']

gaze_map_root = os.getenv('EGTEA_GAZE_MAP_ROOT', '/root/data/egtea/gaze_maps')
gaze_metadata_file = os.path.join(gaze_map_root, 'metadata.json')
algorithm_keys = ('gaze_maps', 'gaze_valid', 'gaze_xy', 'gaze_source')

train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='LoadGazeMap',
        gaze_map_root=gaze_map_root,
        metadata_file=gaze_metadata_file,
        heatmap_size=(14, 14),
        sigma=1.5,
        missing_policy='zeros',
        gaze_mode='real',
        cache_size=64),
    dict(type='PackActionInputs', algorithm_keys=algorithm_keys),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='acc/top1', max_keep_ckpts=1),
    logger=dict(interval=5))
