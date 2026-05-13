"""
最小烟雾测试：验证 config 可加载、dataset 可构建、能取到第一个样本。

Usage (云端):
    cd /root/mmaction2
    python projects/egtea_gaze/tools/smoke_test.py \
        --config projects/egtea_gaze/configs/tsm_r50_egtea.py

    python projects/egtea_gaze/tools/smoke_test.py \
        --config projects/egtea_gaze/configs/slowfast_r50_egtea.py
"""
import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Smoke test for EGTEA config')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--skip-model', action='store_true',
                        help='Skip model build (only test dataset)')
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []

    print('=' * 60)
    print('EGTEA Gaze+ Smoke Test')
    print('=' * 60)
    print(f'Config: {args.config}')
    print()

    # ---- Step 1: Load config ----
    print('[1/5] Loading config...')
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(args.config)
        print(f'  [OK] Config loaded successfully')
        print(f'  dataset_type: {cfg.dataset_type}')
        print(f'  num_classes: {cfg.model.cls_head.num_classes}')
        print(f'  data_root: {cfg.data_root}')
        print(f'  ann_file_train: {cfg.ann_file_train}')
    except Exception as e:
        print(f'  [FAIL] {e}')
        errors.append(f'Config load failed: {e}')
        print('\nCannot proceed without config. Exiting.')
        sys.exit(1)

    # ---- Step 2: Check paths ----
    print('\n[2/5] Checking file paths...')
    import os
    paths_to_check = [
        ('data_root', cfg.data_root),
        ('ann_file_train', cfg.ann_file_train),
        ('ann_file_val', cfg.ann_file_val),
    ]
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = '[OK]' if exists else '[MISSING]'
        print(f'  {status} {name}: {path}')
        if not exists:
            errors.append(f'Path not found: {name} = {path}')

    # Check first video file from annotation
    if os.path.exists(cfg.ann_file_train):
        with open(cfg.ann_file_train) as f:
            first_line = f.readline().strip()
        video_rel = first_line.split()[0]
        video_full = os.path.join(cfg.data_root, video_rel)
        exists = os.path.exists(video_full)
        status = '[OK]' if exists else '[MISSING]'
        print(f'  {status} First video: {video_full}')
        if not exists:
            errors.append(f'First video not found: {video_full}')

    # ---- Step 3: Check decord ----
    print('\n[3/5] Checking decord...')
    try:
        import decord
        print(f'  [OK] decord version: {decord.__version__}')

        # Try to read the first video
        if os.path.exists(cfg.ann_file_train):
            with open(cfg.ann_file_train) as f:
                first_line = f.readline().strip()
            video_rel = first_line.split()[0]
            video_full = os.path.join(cfg.data_root, video_rel)
            if os.path.exists(video_full):
                vr = decord.VideoReader(video_full)
                print(f'  [OK] Video readable: {len(vr)} frames, '
                      f'fps={vr.get_avg_fps():.1f}')
    except ImportError:
        print('  [FAIL] decord not installed')
        errors.append('decord not installed')
    except Exception as e:
        print(f'  [FAIL] decord error: {e}')
        errors.append(f'decord error: {e}')

    # ---- Step 4: Build dataset and get first sample ----
    print('\n[4/5] Building dataset and loading first sample...')
    try:
        from mmaction.registry import DATASETS, TRANSFORMS
        from mmengine.registry import init_default_scope
        init_default_scope('mmaction')

        # Build train dataset
        dataset_cfg = cfg.train_dataloader.dataset
        dataset = DATASETS.build(dataset_cfg)
        print(f'  [OK] Dataset built: {len(dataset)} samples')

        # Get first sample
        t0 = time.time()
        sample = dataset[0]
        t1 = time.time()
        print(f'  [OK] First sample loaded in {t1-t0:.2f}s')

        # Print tensor shapes
        inputs = sample['inputs']
        label = sample['data_samples'].gt_label
        print(f'  Input tensor shape: {inputs.shape}')
        print(f'  Input dtype: {inputs.dtype}')
        print(f'  Label: {label}')

    except Exception as e:
        import traceback
        print(f'  [FAIL] {e}')
        traceback.print_exc()
        errors.append(f'Dataset build/load failed: {e}')

    # ---- Step 5: Build model (optional) ----
    if not args.skip_model:
        print('\n[5/5] Building model...')
        try:
            from mmaction.registry import MODELS
            import torch

            model = MODELS.build(cfg.model)
            print(f'  [OK] Model built: {type(model).__name__}')

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters()
                                   if p.requires_grad)
            print(f'  Total params: {total_params/1e6:.1f}M')
            print(f'  Trainable params: {trainable_params/1e6:.1f}M')

            # Try a forward pass with dummy data
            if 'sample' in dir() and sample is not None:
                print('  Attempting forward pass with real sample...')
                model.eval()
                with torch.no_grad():
                    # Wrap in batch format
                    from mmaction.structures import ActionDataSample
                    data = dict(
                        inputs=[inputs.unsqueeze(0)],
                        data_samples=[sample['data_samples']]
                    )
                    # Use model's data_preprocessor
                    data = model.data_preprocessor(data, training=False)
                    result = model._run_forward(data, mode='predict')
                    print(f'  [OK] Forward pass successful')
                    if hasattr(result[0], 'pred_score'):
                        pred = result[0].pred_score
                        print(f'  Prediction shape: {pred.shape}')
                        print(f'  Top-5 predicted classes: '
                              f'{pred.topk(5).indices.tolist()}')

        except Exception as e:
            import traceback
            print(f'  [FAIL] {e}')
            traceback.print_exc()
            errors.append(f'Model build/forward failed: {e}')
    else:
        print('\n[5/5] Model build skipped (--skip-model)')

    # ---- Summary ----
    print('\n' + '=' * 60)
    if errors:
        print(f'SMOKE TEST: {len(errors)} ERROR(S) FOUND')
        for i, err in enumerate(errors, 1):
            print(f'  {i}. {err}')
        print('\nPlease fix these issues before starting training.')
        sys.exit(1)
    else:
        print('SMOKE TEST: ALL PASSED')
        print('\nYou can now start training with:')
        print(f'  python tools/train.py {args.config} --work-dir /root/outputs/egtea_gaze/xxx')
    print('=' * 60)


if __name__ == '__main__':
    main()
