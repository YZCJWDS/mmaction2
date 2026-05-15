"""Visualize model attention vs GT gaze for thesis figures and defense slides.

This script generates side-by-side comparison images showing:
  1. RGB original frame
  2. GT gaze heatmap overlay (from offline cache, for comparison only)
  3. Model predicted attention overlay (from the gaze attention branch)
  4. Combined comparison strip

IMPORTANT: GT gaze is used ONLY for visualization comparison. The model
inference is RGB-only — no gaze is fed as input at test time.

Outputs are organized into correct/ and wrong/ subdirectories based on whether
the model's top-1 prediction matches the ground-truth label.

File naming convention:
    {rank:03d}_{video_stem}_gt{gt_label}_pred{pred_label}_{correct|wrong}_*.png

Typical usage::

    python projects/egtea_gaze/tools/visualize_gaze_attention.py
    python projects/egtea_gaze/tools/visualize_gaze_attention.py \
        --checkpoint /root/outputs/egtea_gaze/gaze_slowfast_r50/epoch_25.pth \
        --num-samples 50 --overwrite
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Defaults (cloud paths)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = 'projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py'
DEFAULT_CHECKPOINT_PATTERN = '/root/outputs/egtea_gaze/gaze_slowfast_r50/best_acc_top1_*.pth'
DEFAULT_VIDEO_ROOT = '/root/data/egtea/videos/cropped_clips'
DEFAULT_GAZE_MAP_ROOT = '/root/data/egtea/gaze_maps'
DEFAULT_ANN_FILE = '/root/data/egtea/action_annotation/val.txt'
DEFAULT_OUT_DIR = '/root/outputs/egtea_gaze/attention_visualization'
DEFAULT_NUM_SAMPLES = 20
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _resolve_checkpoint(pattern: str) -> Optional[str]:
    """Resolve a checkpoint path that may contain glob wildcards.

    Priority: newest file matching the glob pattern.
    """
    if os.path.isfile(pattern):
        return pattern

    matches = glob.glob(pattern)
    if not matches:
        return None

    # Sort by modification time, newest first.
    matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                 reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def _load_label_names(ann_root: str) -> Dict[int, str]:
    """Try to load action class names. Returns {label_id: name}."""
    mapping: Dict[int, str] = {}

    # Try action_idx.txt (format: "action_name action_id")
    candidates = [
        os.path.join(ann_root, 'action_idx.txt'),
        os.path.join(os.path.dirname(ann_root), 'action_idx.txt'),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    parts = line.strip().rsplit(maxsplit=1)
                    if len(parts) == 2:
                        try:
                            mapping[int(parts[1])] = parts[0]
                        except ValueError:
                            pass
        except OSError:
            pass
        if mapping:
            return mapping

    return mapping


# ---------------------------------------------------------------------------
# Safe filename helper
# ---------------------------------------------------------------------------

def _safe_stem(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe stem from arbitrary text."""
    safe = re.sub(r'[^\w\-.]', '_', text)
    return safe[:max_len]


# ---------------------------------------------------------------------------
# Core visualization logic
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Visualize model attention vs GT gaze for thesis figures.')
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG,
        help=f'Model config path (relative to repo root). Default: {DEFAULT_CONFIG}')
    parser.add_argument(
        '--checkpoint', default=DEFAULT_CHECKPOINT_PATTERN,
        help='Checkpoint path or glob pattern. Default: best_acc_top1_*.pth')
    parser.add_argument(
        '--video-root', default=DEFAULT_VIDEO_ROOT,
        help=f'Video root directory. Default: {DEFAULT_VIDEO_ROOT}')
    parser.add_argument(
        '--gaze-map-root', default=DEFAULT_GAZE_MAP_ROOT,
        help=f'Gaze cache root. Default: {DEFAULT_GAZE_MAP_ROOT}')
    parser.add_argument(
        '--ann-file', default=DEFAULT_ANN_FILE,
        help=f'Annotation file for sampling. Default: {DEFAULT_ANN_FILE}')
    parser.add_argument(
        '--out-dir', default=DEFAULT_OUT_DIR,
        help=f'Output directory. Default: {DEFAULT_OUT_DIR}')
    parser.add_argument(
        '--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
        help=f'Number of samples to visualize. Default: {DEFAULT_NUM_SAMPLES}')
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help=f'Random seed. Default: {DEFAULT_SEED}')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow writing into a non-empty output directory.')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device for inference. Default: cuda:0')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # -----------------------------------------------------------------------
    # Resolve checkpoint (with glob support and graceful missing handling)
    # -----------------------------------------------------------------------
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    if checkpoint_path is None:
        print(f'[WARN] checkpoint not found: {args.checkpoint}', file=sys.stderr)
        print('[WARN] training may not be finished yet.', file=sys.stderr)
        print('[HINT] run this script after training completes and '
              'best_acc_top1_*.pth exists.', file=sys.stderr)
        return 0

    print(f'[INFO] using checkpoint: {checkpoint_path}')

    # -----------------------------------------------------------------------
    # Output directory safety
    # -----------------------------------------------------------------------
    out_dir = args.out_dir
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not args.overwrite:
            print(f'[ERROR] output directory not empty: {out_dir}', file=sys.stderr)
            print('[HINT] pass --overwrite to allow.', file=sys.stderr)
            return 2
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'wrong'), exist_ok=True)

    # -----------------------------------------------------------------------
    # Lazy imports (torch / mmengine / mmaction) — only after checkpoint check
    # -----------------------------------------------------------------------
    import torch
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint

    from mmaction.registry import DATASETS, MODELS, TRANSFORMS
    from projects.egtea_gaze.egtea_gaze.datasets.transforms import LoadGazeMap
    from projects.egtea_gaze.egtea_gaze.visualization import (
        blend_heatmap_on_rgb, draw_text_box, save_image, write_simple_gallery)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'[INFO] model loaded on {device}')

    # -----------------------------------------------------------------------
    # Build dataset and transforms
    # -----------------------------------------------------------------------
    dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    dataset_cfg['ann_file'] = args.ann_file
    dataset_cfg['test_mode'] = True
    dataset = DATASETS.build(dataset_cfg)
    num_classes = cfg.model.cls_head.num_classes

    # Build pipeline stages: pre-format (for RGB extraction) and post (for model input).
    pipeline_cfg = copy.deepcopy(cfg.val_dataloader.dataset.pipeline)
    # Split at FormatShape: everything before is spatial transforms, after is packing.
    format_idx = None
    for i, stage in enumerate(pipeline_cfg):
        if stage.get('type') == 'FormatShape':
            format_idx = i
            break
    if format_idx is None:
        format_idx = len(pipeline_cfg) - 2

    pre_format_cfgs = pipeline_cfg[:format_idx]
    post_format_cfgs = pipeline_cfg[format_idx:]
    pre_format_transforms = [TRANSFORMS.build(item) for item in pre_format_cfgs]
    post_format_transforms = [TRANSFORMS.build(item) for item in post_format_cfgs]

    # Gaze loader for GT visualization only (NOT fed to model).
    gaze_loader = LoadGazeMap(
        gaze_map_root=args.gaze_map_root,
        metadata_file=os.path.join(args.gaze_map_root, 'metadata.json'),
        missing_policy='zeros',
        gaze_mode='real')

    # Label names.
    ann_root = os.path.dirname(args.ann_file)
    label_names = _load_label_names(ann_root)

    # -----------------------------------------------------------------------
    # Sample indices
    # -----------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    print(f'[INFO] visualizing {num_samples} samples from {args.ann_file}')

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    summary: List[Dict[str, Any]] = []
    gallery_items: List[dict] = []
    correct_count = 0
    wrong_count = 0

    for rank, idx in enumerate(indices):
        data_info = dataset.get_data_info(idx)
        gt_label = int(data_info.get('label', -1))
        filename = data_info.get('filename', '')
        video_stem = _safe_stem(Path(filename).stem)

        # Run pre-format transforms to get decoded RGB frames.
        results = copy.deepcopy(data_info)
        for transform in pre_format_transforms:
            results = transform(results)

        # Extract a representative RGB frame (middle of clip).
        imgs = results.get('imgs')
        if imgs is None or len(imgs) == 0:
            warnings.warn(f'no frames decoded for index {idx}, skipping')
            continue
        mid_idx = len(imgs) // 2
        rgb_frame = np.asarray(imgs[mid_idx])

        # Load GT gaze for visualization comparison (NOT model input).
        gaze_results = copy.deepcopy(results)
        gaze_results = gaze_loader(gaze_results)
        gt_gaze_maps = gaze_results.get('gaze_maps')
        gt_gaze_valid = gaze_results.get('gaze_valid')
        gt_map = None
        if gt_gaze_maps is not None and len(gt_gaze_maps) > mid_idx:
            gt_map = gt_gaze_maps[mid_idx]
            # Check if this frame's gaze is actually valid.
            if gt_gaze_valid is not None and len(gt_gaze_valid) > mid_idx:
                if gt_gaze_valid[mid_idx] < 0.5:
                    gt_map = None  # Invalid gaze, don't show misleading overlay.

        # Run post-format transforms to get model-ready tensor.
        packed_results = copy.deepcopy(results)
        for transform in post_format_transforms:
            packed_results = transform(packed_results)

        # Model inference (RGB only, no gaze input).
        inputs = packed_results['inputs']
        data_samples = packed_results['data_samples']
        batch = dict(inputs=[inputs], data_samples=[data_samples])

        with torch.no_grad():
            batch = model.data_preprocessor(batch, training=False)
            feats = model.extract_feat(batch['inputs'])
            if isinstance(feats, tuple) and len(feats) == 2:
                feats, _ = feats

            # Get attention map from gaze branch.
            cls_score, attn_logits = model.cls_head.forward_with_attention(feats)

            # Predicted class.
            pred_label = int(cls_score.argmax(dim=-1).item())

            # Attention probability map (spatial softmax).
            attn_flat = attn_logits.flatten(2)
            attn_prob = torch.softmax(attn_flat, dim=-1).view_as(attn_logits)
            # Take middle temporal slice.
            t_mid = attn_prob.shape[2] // 2 if attn_prob.dim() == 4 else 0
            if attn_prob.dim() == 4:
                attn_map = attn_prob[0, t_mid].cpu().numpy()
            elif attn_prob.dim() == 3:
                attn_map = attn_prob[0, t_mid].cpu().numpy() if attn_prob.shape[1] > 1 else attn_prob[0, 0].cpu().numpy()
            else:
                attn_map = attn_prob[0].cpu().numpy()

        # Determine correct/wrong.
        is_correct = (pred_label == gt_label)
        status = 'correct' if is_correct else 'wrong'
        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        # Build descriptive filename prefix.
        gt_name = label_names.get(gt_label, str(gt_label))
        pred_name = label_names.get(pred_label, str(pred_label))
        prefix = f'{rank:03d}_{video_stem}_gt{gt_label}_pred{pred_label}_{status}'
        subdir = os.path.join(out_dir, status)

        # Generate images.
        rgb_path = os.path.join(subdir, f'{prefix}_rgb.png')
        attn_path = os.path.join(subdir, f'{prefix}_attn.png')
        gt_path = os.path.join(subdir, f'{prefix}_gt.png')
        comp_path = os.path.join(subdir, f'{prefix}_comparison.png')

        # Attention overlay (blue-ish to distinguish from GT gaze red).
        attn_overlay = blend_heatmap_on_rgb(rgb_frame, attn_map,
                                            alpha=0.5, color=(64, 128, 255))
        attn_overlay = draw_text_box(attn_overlay,
                                     [f'Pred: {pred_name} ({pred_label})',
                                      f'Status: {status.upper()}'])

        save_image(rgb_frame, rgb_path)
        save_image(attn_overlay, attn_path)

        # GT gaze overlay (red).
        panels = [rgb_frame]
        if gt_map is not None:
            gt_overlay = blend_heatmap_on_rgb(rgb_frame, gt_map,
                                             alpha=0.5, color=(255, 64, 64))
            gt_overlay = draw_text_box(gt_overlay, ['GT Gaze (train-time only)'])
            save_image(gt_overlay, gt_path)
            panels.append(gt_overlay)
        else:
            gt_path = None

        panels.append(attn_overlay)

        # Side-by-side comparison.
        # Ensure all panels have same height.
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for panel in panels:
            if panel.shape[0] < max_h or panel.shape[1] < max_w:
                pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                pad[:panel.shape[0], :panel.shape[1]] = panel
                padded.append(pad)
            else:
                padded.append(panel)
        comparison = np.concatenate(padded, axis=1)

        # Add header text.
        header_lines = [
            f'Video: {Path(filename).name}',
            f'GT: {gt_name} ({gt_label})  |  Pred: {pred_name} ({pred_label})  |  {status.upper()}',
        ]
        comparison = draw_text_box(comparison, header_lines)
        save_image(comparison, comp_path)

        # Gallery and summary.
        gallery_items.append(dict(
            image=comp_path,
            title=f'{status.upper()} | {Path(filename).name}',
            meta=f'GT: {gt_name} ({gt_label})\nPred: {pred_name} ({pred_label})\n'
                 f'Index: {idx}'))

        summary.append(dict(
            rank=rank,
            index=idx,
            filename=filename,
            video_stem=video_stem,
            gt_label=gt_label,
            gt_name=gt_name,
            pred_label=pred_label,
            pred_name=pred_name,
            is_correct=is_correct,
            rgb=rgb_path,
            gt_gaze=gt_path,
            attention=attn_path,
            comparison=comp_path,
        ))

        if (rank + 1) % 5 == 0:
            print(f'  [{rank + 1}/{num_samples}] processed')

    # -----------------------------------------------------------------------
    # Write summary
    # -----------------------------------------------------------------------
    summary_data = dict(
        config=args.config,
        checkpoint=checkpoint_path,
        ann_file=args.ann_file,
        gaze_map_root=args.gaze_map_root,
        num_samples=num_samples,
        seed=args.seed,
        correct_count=correct_count,
        wrong_count=wrong_count,
        accuracy=correct_count / max(correct_count + wrong_count, 1),
        items=summary,
    )
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary_data, handle, ensure_ascii=False, indent=2, default=str)

    # HTML gallery.
    gallery_path = os.path.join(out_dir, 'index.html')
    write_simple_gallery(
        gallery_items, gallery_path,
        title=f'Gaze Attention Visualization (correct={correct_count}, wrong={wrong_count})')

    print(f'\n[OK] visualization complete: {out_dir}')
    print(f'     correct: {correct_count}, wrong: {wrong_count}')
    print(f'     summary: {summary_path}')
    print(f'     gallery: {gallery_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
