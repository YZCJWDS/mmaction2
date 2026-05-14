_base_ = ['./gaze_slowfast_r50_egtea.py']

gaze_map_root = '/root/data/egtea/gaze_maps'
gaze_metadata_file = '/root/data/egtea/gaze_maps/metadata.json'

train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
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
                gaze_mode='shuffle',
                cache_size=256),
            dict(
                type='PackActionInputs',
                algorithm_keys=('gaze_maps', 'gaze_valid', 'gaze_xy', 'gaze_source')),
        ]))
