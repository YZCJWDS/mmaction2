"""Validate offline gaze caches."""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.egtea_gaze.egtea_gaze.utils import dump_json


def parse_args():
    parser = argparse.ArgumentParser(description='Check EGTEA gaze cache consistency')
    parser.add_argument('--gaze-map-root', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    if not os.path.exists(args.metadata):
        print(f'[ERROR] Missing metadata file: {args.metadata}')
        return 1
    metadata = np.load if args.metadata.endswith('.npz') else None
    import json
    with open(args.metadata, 'r', encoding='utf-8') as handle:
        meta = json.load(handle)
    sample_items = list(meta.get('samples', {}).items())
    if not sample_items:
        print('[ERROR] No samples listed in metadata.')
        return 1
    random.shuffle(sample_items)
    sample_items = sample_items[: min(args.sample, len(sample_items))]

    failures = []
    checked = 0
    valid_heatmaps = 0
    invalid_heatmaps = 0
    expected_hw = tuple(meta.get('heatmap_size', [14, 14]))
    for video_relpath, info in sample_items:
        npz_path = os.path.join(args.gaze_map_root, info['npz_path'])
        if not os.path.exists(npz_path):
            failures.append(f'missing file: {npz_path}')
            continue
        payload = np.load(npz_path, allow_pickle=True)
        gaze_maps = np.asarray(payload['gaze_maps'])
        gaze_valid = np.asarray(payload['gaze_valid'])
        if gaze_maps.ndim != 3:
            failures.append(f'bad gaze_maps rank: {npz_path} -> {gaze_maps.shape}')
            continue
        if tuple(gaze_maps.shape[1:][::-1]) != expected_hw:
            failures.append(f'bad heatmap size: {npz_path} -> {gaze_maps.shape}')
        if gaze_maps.shape[0] != gaze_valid.shape[0]:
            failures.append(f'length mismatch: {npz_path}')
        if not np.isfinite(gaze_maps).all():
            failures.append(f'nan/inf heatmap: {npz_path}')
        heatmap_sums = gaze_maps.reshape(gaze_maps.shape[0], -1).sum(axis=1)
        valid_mask = gaze_valid.astype(bool)
        if valid_mask.any():
            valid_heatmaps += int(valid_mask.sum())
            if not np.allclose(heatmap_sums[valid_mask], 1.0, atol=1e-2):
                failures.append(f'valid heatmap sum mismatch: {npz_path}')
        if (~valid_mask).any():
            invalid_heatmaps += int((~valid_mask).sum())
            if not np.allclose(heatmap_sums[~valid_mask], 0.0, atol=1e-4):
                failures.append(f'invalid heatmap not zero: {npz_path}')
        checked += 1

    summary = dict(
        checked=checked,
        valid_heatmaps=valid_heatmaps,
        invalid_heatmaps=invalid_heatmaps,
        num_failures=len(failures),
        failures=failures[:200],
    )
    dump_json(summary, os.path.join(args.gaze_map_root, 'check_summary.json'))
    print(summary)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
