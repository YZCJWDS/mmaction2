"""Visualize gaze points on sampled RGB clip frames."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.egtea_gaze.egtea_gaze.utils import (
    align_gaze_to_clip, build_gaze_file_index, dump_json, get_video_stats,
    match_gaze_file, normalize_xy_array, parse_clip_start_frame,
    parse_clip_frame_info, parse_clip_frame_range, parse_gaze_file_full,
    read_video_frame, resolve_source_resolution, resolve_video_path)
from projects.egtea_gaze.egtea_gaze.visualization import (
    draw_gaze_point, draw_text_box, save_image, write_simple_gallery)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize gaze overlays on EGTEA clips')
    parser.add_argument('--data-root', default='/root/data/egtea')
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--gaze-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-clips', type=int, default=30)
    parser.add_argument('--frames-per-clip', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only-fixation', action='store_true')
    parser.add_argument('--gaze-source-width', type=float, default=None)
    parser.add_argument('--gaze-source-height', type=float, default=None)
    parser.add_argument('--auto-scale-gaze', dest='auto_scale_gaze', action='store_true')
    parser.add_argument('--no-auto-scale-gaze', dest='auto_scale_gaze', action='store_false')
    parser.set_defaults(auto_scale_gaze=True)
    return parser.parse_args()


def sample_frame_indices(total_frames: int, frames_per_clip: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int32)
    if total_frames <= frames_per_clip:
        return np.arange(total_frames, dtype=np.int32)
    return np.linspace(0, total_frames - 1, frames_per_clip, dtype=np.int32)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ann_file, 'r', encoding='utf-8') as handle:
        entries = [line.strip().split() for line in handle if line.strip()]
    if not entries:
        print(f'[ERROR] Empty annotation file: {args.ann_file}')
        return 1

    sampled_entries = random.sample(entries, min(args.num_clips, len(entries)))
    gaze_index = build_gaze_file_index(args.gaze_root)
    gallery_items = []
    warnings_list = []
    frames_saved = 0
    clips_missing_gaze = 0
    frames_with_valid_gaze = 0
    frames_with_visible_gaze = 0
    frames_out_of_bounds = 0
    frames_invalid = 0
    frames_fixation = 0
    frames_saccade = 0
    frames_scaled_into_view = 0
    frames_out_of_source_bounds = 0
    used_source_resolution = None
    used_coordinate_scale_mode = None
    invalid_reason_counts = Counter()
    clip_debug = []

    print('=' * 80)
    print('Visualize EGTEA Gaze Overlay')
    print('=' * 80)
    for clip_idx, parts in enumerate(sampled_entries):
        video_relpath = parts[0]
        try:
            video_path = resolve_video_path(args.video_root, video_relpath)
            if not os.path.exists(video_path):
                warnings_list.append(f'Video missing: {video_path}')
                continue
            total_frames, fps = get_video_stats(video_path)
            if total_frames <= 0:
                warnings_list.append(f'Empty or unreadable video: {video_path}')
                continue
            gaze_file, match_warnings = match_gaze_file(video_relpath, gaze_index)
            warnings_list.extend(match_warnings)
            if gaze_file is None:
                clips_missing_gaze += 1
                invalid_reason_counts['no_gaze_file'] += int(args.frames_per_clip)
                continue

            parsed = parse_gaze_file_full(gaze_file)
            frame_info = parse_clip_frame_info(video_path)
            start_frame = frame_info['start_frame'] if frame_info else None
            end_frame = frame_info['end_frame'] if frame_info else None
            parse_source = frame_info['parse_source'] if frame_info else 'fallback_local'
            frame_range = (start_frame, end_frame) if frame_info else parse_clip_frame_range(video_path)
            gaze_frame_ids = [record['frame_id'] for record in parsed.records if record.get('frame_id') is not None]
            clip_debug_item = dict(
                video_name=Path(video_path).name,
                video_relpath=video_relpath,
                parsed_start_frame=start_frame,
                parsed_end_frame=end_frame,
                parse_source=parse_source,
                num_gaze_rows_total=len(parsed.records),
                num_gaze_frames_total=len(set(gaze_frame_ids)),
                gaze_frame_min=min(gaze_frame_ids) if gaze_frame_ids else None,
                gaze_frame_max=max(gaze_frame_ids) if gaze_frame_ids else None,
                sampled_frames=[],
            )
            aligned = align_gaze_to_clip(
                parsed,
                total_frames=total_frames,
                fps=fps,
                start_frame=start_frame,
                frame_range=frame_range,
                only_fixation=args.only_fixation,
                prefer_fixation=True)
            try:
                first_frame = read_video_frame(video_path, 0)
            except Exception as exc:
                warnings_list.append(f'First frame decode failed: {video_path} ({exc})')
                continue
            image_size = (first_frame.shape[1], first_frame.shape[0])
            fallback_resolution = image_size if parsed.gaze_format.coordinate_mode == 'pixel' else None
            source_resolution, source_resolution_mode = resolve_source_resolution(
                parsed.gaze_format,
                override_width=args.gaze_source_width,
                override_height=args.gaze_source_height,
                fallback_resolution=fallback_resolution if args.auto_scale_gaze else None)
            if not args.auto_scale_gaze and parsed.gaze_format.coordinate_mode == 'pixel':
                source_resolution = (None, None)
                source_resolution_mode = 'disabled'
            gaze_xy, source_in_bounds, coordinate_scale_mode = normalize_xy_array(
                aligned['gaze_xy'],
                parsed.gaze_format.coordinate_mode,
                image_size,
                source_resolution=source_resolution,
                clip=False,
                return_info=True)
            if used_source_resolution is None:
                used_source_resolution = list(source_resolution)
            if used_coordinate_scale_mode is None:
                used_coordinate_scale_mode = coordinate_scale_mode
            frame_ids = sample_frame_indices(total_frames, args.frames_per_clip)
            for frame_id in frame_ids:
                try:
                    frame = read_video_frame(video_path, int(frame_id))
                except Exception as exc:
                    warnings_list.append(
                        f'Frame decode failed: {video_path} frame={int(frame_id)} ({exc})')
                    continue
                if aligned['gaze_valid'][frame_id]:
                    frames_with_valid_gaze += 1
                    xy = gaze_xy[frame_id]
                    in_bounds = 0.0 <= float(xy[0]) <= 1.0 and 0.0 <= float(xy[1]) <= 1.0
                    if not bool(source_in_bounds[frame_id]):
                        frames_out_of_source_bounds += 1
                        invalid_reason_counts['out_of_source_bounds'] += 1
                    event_name = str(aligned['gaze_type'][frame_id])
                    if event_name == 'fixation':
                        frames_fixation += 1
                    elif event_name == 'saccade':
                        frames_saccade += 1
                    if in_bounds:
                        frame = draw_gaze_point(frame, xy, radius=10)
                        frames_with_visible_gaze += 1
                        frames_scaled_into_view += 1
                    else:
                        frames_out_of_bounds += 1
                        invalid_reason_counts['out_of_image_bounds'] += 1
                else:
                    frames_invalid += 1
                    reason = str(aligned['invalid_reasons'][frame_id])
                    invalid_reason_counts[reason] += 1
                xy = gaze_xy[frame_id]
                raw_xy = aligned['gaze_xy'][frame_id]
                visible = int(
                    aligned['gaze_valid'][frame_id] and
                    0.0 <= float(xy[0]) <= 1.0 and
                    0.0 <= float(xy[1]) <= 1.0)
                frame = draw_text_box(
                    frame,
                    lines=[
                        f'frame: {int(frame_id)}',
                        f'valid: {int(aligned["gaze_valid"][frame_id])}',
                        f'visible: {visible}',
                        f'type: {aligned["gaze_type"][frame_id]}',
                        f'raw_x: {float(raw_xy[0]):.2f}',
                        f'raw_y: {float(raw_xy[1]):.2f}',
                        f'scaled_x: {float(xy[0]):.4f}',
                        f'scaled_y: {float(xy[1]):.4f}',
                        f'global: {int(aligned["global_frame_ids"][frame_id])}',
                        f'matched: {int(aligned["matched_frame_ids"][frame_id])}',
                        f'rows: {int(aligned["rows_found"][frame_id])}',
                        f'invalid: {aligned["invalid_reasons"][frame_id]}',
                        f'start: {start_frame}',
                        f'range: {frame_range}',
                    ],
                )
                clip_debug_item['sampled_frames'].append(dict(
                    clip_idx=int(frame_id),
                    computed_global_frame=int(aligned['global_frame_ids'][frame_id]),
                    matched_frame=int(aligned['matched_frame_ids'][frame_id]),
                    rows_found=int(aligned['rows_found'][frame_id]),
                    valid=int(aligned['gaze_valid'][frame_id]),
                    visible=int(visible),
                    invalid_reason=str(aligned['invalid_reasons'][frame_id]),
                ))
                output_name = f'{clip_idx:03d}_{Path(video_relpath).stem}_f{int(frame_id):04d}.png'
                output_path = os.path.join(args.out_dir, output_name)
                save_image(frame, output_path)
                frames_saved += 1
                gallery_items.append(
                    dict(
                        image=output_path,
                        title=output_name,
                        meta=(
                            f'video_relpath: {video_relpath}\n'
                            f'gaze_file: {gaze_file}\n'
                            f'frame_id: {int(frame_id)}\n'
                            f'valid: {int(aligned["gaze_valid"][frame_id])}\n'
                            f'gaze_type: {aligned["gaze_type"][frame_id]}\n'
                            f'start_frame: {start_frame}\n'
                            f'frame_range: {frame_range}\n'
                            f'global_frame: {int(aligned["global_frame_ids"][frame_id])}\n'
                            f'matched_frame: {int(aligned["matched_frame_ids"][frame_id])}\n'
                            f'rows_found: {int(aligned["rows_found"][frame_id])}\n'
                            f'invalid_reason: {aligned["invalid_reasons"][frame_id]}'
                        )))
            if frames_invalid and all(int(item['valid']) == 0 for item in clip_debug_item['sampled_frames']):
                if start_frame is None:
                    invalid_reason_counts['no_start_frame'] += 1
            clip_debug.append(clip_debug_item)
        except Exception as exc:
            warnings_list.append(f'Clip skipped due to unexpected error: {video_relpath} ({exc})')
            continue

    html_path = os.path.join(args.out_dir, 'index.html')
    write_simple_gallery(gallery_items, html_path, title='EGTEA gaze overlay')
    summary = dict(
        sampled_clips=len(sampled_entries),
        frames_saved=frames_saved,
        clips_missing_gaze=clips_missing_gaze,
        frames_with_valid_gaze=frames_with_valid_gaze,
        frames_with_visible_gaze=frames_with_visible_gaze,
        frames_out_of_bounds=frames_out_of_bounds,
        frames_invalid=frames_invalid,
        frames_fixation=frames_fixation,
        frames_saccade=frames_saccade,
        gaze_source_resolution=used_source_resolution,
        frames_scaled_into_view=frames_scaled_into_view,
        frames_out_of_source_bounds=frames_out_of_source_bounds,
        coordinate_scale_mode=used_coordinate_scale_mode,
        invalid_reason_counts=dict(invalid_reason_counts),
        clip_debug=clip_debug,
        valid_ratio=frames_with_valid_gaze / max(1, frames_saved),
        visible_ratio=frames_with_visible_gaze / max(1, frames_with_valid_gaze),
        warnings=sorted(set(warnings_list)),
    )
    dump_json(summary, os.path.join(args.out_dir, 'summary.json'))
    print(f'[OK] Saved overlays to: {args.out_dir}')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
