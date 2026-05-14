"""Minimal check for EGTEA clip frame-range parsing."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_parse_fn():
    module_path = Path(__file__).resolve().parents[1] / 'egtea_gaze' / 'utils' / 'gaze_utils.py'
    spec = importlib.util.spec_from_file_location('egtea_gaze_gaze_utils_check',
                                                  str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.parse_clip_frame_info


def parse_args():
    parser = argparse.ArgumentParser(description='Check EGTEA clip frame parsing')
    parser.add_argument(
        '--video-name',
        default='OP01-R06-GreekSalad-465120-471590-F011147-F011334.mp4')
    parser.add_argument('--expect-start', type=int, default=11147)
    parser.add_argument('--expect-end', type=int, default=11334)
    parser.add_argument('--expect-source', default='F_range')
    return parser.parse_args()


def main():
    args = parse_args()
    parse_clip_frame_info = load_parse_fn()
    info = parse_clip_frame_info(args.video_name)
    print('input:', args.video_name)
    print('parsed:', info)
    if info is None:
        print('[ERROR] failed to parse frame info')
        return 1
    if info['start_frame'] != args.expect_start:
        print(f'[ERROR] start_frame mismatch: {info["start_frame"]} != {args.expect_start}')
        return 1
    if info['end_frame'] != args.expect_end:
        print(f'[ERROR] end_frame mismatch: {info["end_frame"]} != {args.expect_end}')
        return 1
    if info['parse_source'] != args.expect_source:
        print(f'[ERROR] parse_source mismatch: {info["parse_source"]} != {args.expect_source}')
        return 1
    print('[OK] frame parsing matches expectation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
