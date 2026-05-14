"""Build offline gaze caches for EGTEA clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.egtea_gaze.egtea_gaze.utils import (
    align_gaze_to_clip, build_gaze_file_index, compare_annotation_lists,
    dump_json, ensure_dir, gaze_xy_to_heatmaps, get_video_stats,
    match_gaze_file, normalize_xy_array, parse_clip_frame_range,
    parse_clip_start_frame, parse_gaze_file_full, resolve_source_resolution,
    resolve_video_path, safe_path_id, stable_hash)


def parse_args():
    parser = argparse.ArgumentParser(description='Create offline EGTEA gaze map caches')
    parser.add_argument('--data-root', default='/root/data/egtea')
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--gaze-root', required=True)
    parser.add_argument('--processed-root', default='')
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--splits', nargs='*', default=['1', '2', '3'])
    parser.add_argument('--heatmap-size', nargs=2, type=int, default=[14, 14])
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--only-fixation', action='store_true')
    parser.add_argument('--fixation-values', nargs='*', default=None)
    parser.add_argument('--gaze-source-width', type=float, default=None)
    parser.add_argument('--gaze-source-height', type=float, default=None)
    parser.add_argument('--out-of-source-policy', choices=['clip', 'invalid'],
                        default='clip')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def collect_annotation_files(data_root: str,
                             processed_root: str,
                             splits: List[str]) -> Dict[str, str]:
    ann_root = os.path.join(data_root, 'action_annotation')
    ann_files: Dict[str, str] = {}
    for split_name in ('train', 'val', 'test'):
        path = os.path.join(ann_root, f'{split_name}.txt')
        if os.path.exists(path):
            ann_files[split_name] = path
    if processed_root:
        for split in splits:
            for split_name in ('train', 'val', 'test'):
                path = os.path.join(processed_root, f'{split_name}_s{split}.txt')
                if os.path.exists(path):
                    ann_files[f'{split_name}_s{split}'] = path
    return ann_files


def process_one_sample(video_relpath: str,
                       video_root: str,
                       gaze_index,
                       out_dir: str,
                       heatmap_size: Tuple[int, int],
                       sigma: float,
                       only_fixation: bool,
                       fixation_values,
                       gaze_source_width: float | None,
                       gaze_source_height: float | None,
                       out_of_source_policy: str,
                       overwrite: bool) -> dict:
    video_path = resolve_video_path(video_root, video_relpath)
    sample_id = f'{safe_path_id(Path(video_relpath).stem)}_{stable_hash(video_relpath)}'
    npz_path = os.path.join(out_dir, f'{sample_id}.npz')
    if os.path.exists(npz_path) and not overwrite:
        return dict(
            status='skipped_existing',
            video_relpath=video_relpath,
            npz_path=npz_path)

    if not os.path.exists(video_path):
        return dict(status='missing_video', video_relpath=video_relpath, npz_path=npz_path)

    gaze_file, match_warnings = match_gaze_file(video_relpath, gaze_index)
    if gaze_file is None:
        return dict(status='missing_gaze', video_relpath=video_relpath, npz_path=npz_path,
                    warnings=match_warnings)

    parsed = parse_gaze_file_full(gaze_file)
    total_frames, fps = get_video_stats(video_path)
    start_frame = parse_clip_start_frame(video_path)
    frame_range = parse_clip_frame_range(video_path)
    aligned = align_gaze_to_clip(
        parsed,
        total_frames=total_frames,
        fps=fps,
        start_frame=start_frame,
        frame_range=frame_range,
        only_fixation=only_fixation,
        fixation_values=fixation_values)

    coordinate_mode = parsed.gaze_format.coordinate_mode
    image_size = (320, 240)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                image_size = (width, height)
        cap.release()
    except Exception:
        pass

    fallback_resolution = image_size if coordinate_mode == 'pixel' else None
    source_resolution, source_resolution_mode = resolve_source_resolution(
        parsed.gaze_format,
        override_width=gaze_source_width,
        override_height=gaze_source_height,
        fallback_resolution=fallback_resolution)
    gaze_xy, source_in_bounds, coordinate_scale_mode = normalize_xy_array(
        aligned['gaze_xy'],
        coordinate_mode,
        image_size,
        source_resolution=source_resolution,
        clip=(out_of_source_policy == 'clip'),
        return_info=True)
    gaze_valid = np.asarray(aligned['gaze_valid'], dtype=np.uint8)
    frames_out_of_source_bounds = int(
        np.sum((gaze_valid > 0) & (~source_in_bounds.astype(bool))))
    if out_of_source_policy == 'invalid':
        gaze_valid = gaze_valid * source_in_bounds.astype(np.uint8)
    gaze_maps = gaze_xy_to_heatmaps(gaze_xy, gaze_valid, heatmap_size, sigma).astype(np.float16)
    no_valid_gaze = int(gaze_valid.sum()) == 0

    ensure_dir(out_dir)
    source_resolution_array = np.asarray([
        int(source_resolution[0]) if source_resolution[0] is not None else -1,
        int(source_resolution[1]) if source_resolution[1] is not None else -1,
    ], dtype=np.int32)
    np.savez_compressed(
        npz_path,
        gaze_xy=gaze_xy.astype(np.float16),
        gaze_valid=gaze_valid.astype(np.uint8),
        frame_indices=np.asarray(aligned['frame_indices'], dtype=np.int32),
        gaze_maps=gaze_maps,
        gaze_type=np.asarray(aligned['gaze_type']),
        original_video=np.asarray(video_relpath),
        source_gaze_file=np.asarray(gaze_file),
        coordinate_mode=np.asarray(coordinate_mode),
        coordinate_scale_mode=np.asarray(coordinate_scale_mode),
        source_resolution=source_resolution_array,
        warning_codes=np.asarray(aligned['warning_codes'], dtype=object),
    )
    return dict(
        status='no_valid_gaze' if no_valid_gaze else 'success',
        video_relpath=video_relpath,
        npz_path=npz_path,
        source_gaze_file=gaze_file,
        coordinate_mode=coordinate_mode,
        coordinate_scale_mode=coordinate_scale_mode,
        source_resolution=list(source_resolution),
        source_resolution_mode=source_resolution_mode,
        frames_out_of_source_bounds=frames_out_of_source_bounds,
        warnings=match_warnings + aligned['warning_codes'],
        valid_frames=int(gaze_valid.sum()),
    )


def main():
    args = parse_args()
    ensure_dir(args.out_root)
    ann_files = collect_annotation_files(args.data_root, args.processed_root, args.splits)
    if not ann_files:
        print('[ERROR] No annotation files found for gaze cache generation.')
        return 1

    gaze_index = build_gaze_file_index(args.gaze_root)
    metadata = dict(
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        only_fixation=args.only_fixation,
        fixation_values=args.fixation_values,
        coordinate_mode='normalized_per_clip',
        out_of_source_policy=args.out_of_source_policy,
        created_at=np.datetime64('now').astype(str),
        num_samples=0,
        num_success=0,
        num_missing_gaze=0,
        num_no_valid_gaze=0,
        num_missing_video=0,
        num_generated=0,
        num_skipped_existing=0,
        num_failed=0,
        warnings=[],
        splits={},
        samples={},
        split_differences=[],
    )

    if args.processed_root:
        for base_name in ('train', 'val', 'test'):
            baseline_ann = ann_files.get(base_name)
            if not baseline_ann:
                continue
            for split in args.splits:
                candidate = ann_files.get(f'{base_name}_s{split}')
                if candidate:
                    metadata['split_differences'].append(
                        compare_annotation_lists(baseline_ann, candidate))

    for split_name, ann_path in ann_files.items():
        out_dir = os.path.join(args.out_root, split_name)
        ensure_dir(out_dir)
        with open(ann_path, 'r', encoding='utf-8') as handle:
            samples = [line.strip().split()[0] for line in handle if line.strip()]
        metadata['splits'][split_name] = dict(
            ann_file=ann_path,
            out_dir=out_dir,
            num_samples=len(samples),
        )
        metadata['num_samples'] += len(samples)
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            for video_relpath in samples:
                futures.append(executor.submit(
                    process_one_sample,
                    video_relpath,
                    args.video_root,
                    gaze_index,
                    out_dir,
                    tuple(args.heatmap_size),
                    args.sigma,
                    args.only_fixation,
                    args.fixation_values,
                    args.gaze_source_width,
                    args.gaze_source_height,
                    args.out_of_source_policy,
                    args.overwrite))
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(
                        status='failed_exception',
                        video_relpath='unknown',
                        npz_path=os.path.join(out_dir, 'unknown.npz'),
                        warnings=[f'worker_exception: {exc}'])
                results.append(result)
        for result in results:
            video_relpath = result['video_relpath']
            metadata['samples'][video_relpath] = dict(
                npz_path=os.path.relpath(result['npz_path'], args.out_root),
                split=split_name,
                status=result['status'],
                source_gaze_file=result.get('source_gaze_file'),
                coordinate_mode=result.get('coordinate_mode'),
                coordinate_scale_mode=result.get('coordinate_scale_mode'),
                source_resolution=result.get('source_resolution'),
                frames_out_of_source_bounds=result.get('frames_out_of_source_bounds', 0),
                valid_frames=result.get('valid_frames', 0),
                warnings=result.get('warnings', []),
            )
            if result['status'] == 'success':
                metadata['num_success'] += 1
                metadata['num_generated'] += 1
            elif result['status'] == 'missing_gaze':
                metadata['num_missing_gaze'] += 1
                metadata['num_failed'] += 1
            elif result['status'] == 'missing_video':
                metadata['num_missing_video'] += 1
                metadata['num_failed'] += 1
            elif result['status'] == 'no_valid_gaze':
                metadata['num_no_valid_gaze'] += 1
                metadata['num_success'] += 1
                metadata['num_generated'] += 1
            elif result['status'] == 'skipped_existing':
                metadata['num_skipped_existing'] += 1
                metadata['num_success'] += 1
            else:
                metadata['num_failed'] += 1
            metadata['warnings'].extend(result.get('warnings', []))

    # Random spot checks.
    sample_items = list(metadata['samples'].items())[: min(10, len(metadata['samples']))]
    spot_checks = []
    for video_relpath, info in sample_items:
        npz_path = os.path.join(args.out_root, info['npz_path'])
        if not os.path.exists(npz_path):
            continue
        payload = np.load(npz_path, allow_pickle=True)
        spot_checks.append(
            dict(
                video_relpath=video_relpath,
                shape=payload['gaze_maps'].shape,
                valid_ratio=float(payload['gaze_valid'].mean())
                if payload['gaze_valid'].size else 0.0,
                finite=bool(np.isfinite(payload['gaze_maps']).all()),
            ))
    metadata['spot_checks'] = spot_checks
    metadata['warnings'] = sorted(set(metadata['warnings']))
    dump_json(metadata, os.path.join(args.out_root, 'metadata.json'))
    print(json.dumps({
        'num_samples': metadata['num_samples'],
        'num_generated': metadata['num_generated'],
        'num_skipped_existing': metadata['num_skipped_existing'],
        'num_success': metadata['num_success'],
        'num_failed': metadata['num_failed'],
        'num_missing_gaze': metadata['num_missing_gaze'],
        'num_no_valid_gaze': metadata['num_no_valid_gaze'],
        'num_missing_video': metadata['num_missing_video'],
        'metadata': os.path.join(args.out_root, 'metadata.json'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
