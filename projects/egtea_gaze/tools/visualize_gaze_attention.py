"""Visualize model attention against offline gaze supervision."""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmaction.registry import MODELS, TRANSFORMS
from projects.egtea_gaze.egtea_gaze.datasets.transforms import LoadGazeMap
from projects.egtea_gaze.egtea_gaze.utils import dump_json
from projects.egtea_gaze.egtea_gaze.visualization import (blend_heatmap_on_rgb,
                                                          save_image,
                                                          write_simple_gallery)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize gaze attention maps')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-clips', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gaze-map-root', default='/root/data/egtea/gaze_maps')
    return parser.parse_args()


def build_vis_sample(cfg: Config, ann_file: str, idx: int):
    pipeline_cfg = copy.deepcopy(cfg.val_dataloader.dataset.pipeline)
    pre_vis_cfgs = pipeline_cfg[:-2]
    post_vis_cfgs = pipeline_cfg[-2:]
    pre_vis = [TRANSFORMS.build(item) for item in pre_vis_cfgs]
    post_vis = [TRANSFORMS.build(item) for item in post_vis_cfgs]

    dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    dataset_cfg['ann_file'] = ann_file
    dataset_cfg['test_mode'] = True
    from mmaction.registry import DATASETS
    dataset = DATASETS.build(dataset_cfg)
    data_info = dataset.get_data_info(idx)
    results = copy.deepcopy(data_info)
    for transform in pre_vis:
        results = transform(results)

    vis_results = copy.deepcopy(results)
    gaze_loader = LoadGazeMap(
        gaze_map_root=os.getenv('EGTEA_GAZE_MAP_ROOT', '/root/data/egtea/gaze_maps'),
        metadata_file=os.path.join(
            os.getenv('EGTEA_GAZE_MAP_ROOT', '/root/data/egtea/gaze_maps'),
            'metadata.json'),
        missing_policy='zeros',
        gaze_mode='real')
    vis_results = gaze_loader(vis_results)

    packed_results = copy.deepcopy(results)
    for transform in post_vis:
        packed_results = transform(packed_results)
    return dataset, vis_results, packed_results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['EGTEA_GAZE_MAP_ROOT'] = args.gaze_map_root

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.dataset.ann_file = args.ann_file
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    dataset, _, _ = build_vis_sample(cfg, args.ann_file, 0)
    num_samples = min(args.num_clips, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    gallery_items = []
    summary = []

    for rank, idx in enumerate(indices):
        dataset, vis_results, packed_results = build_vis_sample(cfg, args.ann_file, idx)
        frame_list = vis_results['imgs']
        frame_mid = len(frame_list) // 2
        rgb = frame_list[frame_mid]
        gt_gaze = vis_results.get('gaze_maps')
        gt_map = gt_gaze[frame_mid] if gt_gaze is not None and len(gt_gaze) > frame_mid else None

        data = dict(
            inputs=[packed_results['inputs']],
            data_samples=[packed_results['data_samples']])
        with torch.no_grad():
            data = model.data_preprocessor(data, training=False)
            feats, _ = model.extract_feat(data['inputs'], test_mode=True)
            _, attn_logits = model.cls_head.forward_with_attention(feats)
            attn_prob = torch.softmax(attn_logits.flatten(2), dim=-1).view_as(attn_logits)
            attn_map = attn_prob[0, attn_prob.shape[1] // 2].cpu().numpy()

        rgb_path = os.path.join(args.out_dir, f'{rank:03d}_rgb.png')
        attn_path = os.path.join(args.out_dir, f'{rank:03d}_attn.png')
        gt_path = os.path.join(args.out_dir, f'{rank:03d}_gt.png')
        comp_path = os.path.join(args.out_dir, f'{rank:03d}_comparison.png')

        rgb_img = np.asarray(rgb)
        attn_img = blend_heatmap_on_rgb(rgb_img, attn_map)
        save_image(rgb_img, rgb_path)
        save_image(attn_img, attn_path)
        if gt_map is not None:
            gt_img = blend_heatmap_on_rgb(rgb_img, gt_map)
            save_image(gt_img, gt_path)
            comparison = np.concatenate([rgb_img, gt_img, attn_img], axis=1)
        else:
            gt_img = None
            comparison = np.concatenate([rgb_img, attn_img], axis=1)
        save_image(comparison, comp_path)

        title = os.path.basename(dataset.get_data_info(idx)['filename'])
        gallery_items.append(
            dict(
                image=comp_path,
                title=title,
                meta=f'index={idx}\ncomparison={comp_path}'))
        summary.append(
            dict(
                index=idx,
                filename=dataset.get_data_info(idx)['filename'],
                rgb=rgb_path,
                gt_gaze=gt_path if gt_img is not None else None,
                attention=attn_path,
                comparison=comp_path))

    write_simple_gallery(gallery_items, os.path.join(args.out_dir, 'index.html'),
                         title='EGTEA gaze attention')
    dump_json(dict(items=summary), os.path.join(args.out_dir, 'summary.json'))
    print(f'[OK] Saved attention visualization to: {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
