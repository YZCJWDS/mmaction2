"""Inspect EGTEA gaze files and emit a robust format report."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.egtea_gaze.egtea_gaze.utils import (build_gaze_file_index,
                                                   dump_json, find_gaze_files,
                                                   parse_gaze_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect EGTEA gaze data layout')
    parser.add_argument('--gaze-root', required=True, help='Root directory with raw gaze files')
    parser.add_argument('--video-root', default='', help='Optional video root for context')
    parser.add_argument('--ann-root', default='', help='Optional annotation root for context')
    parser.add_argument('--max-files', type=int, default=20, help='Maximum files to inspect deeply')
    parser.add_argument('--focus-file', default='',
                        help='Optional specific gaze file to inspect first')
    parser.add_argument('--output', required=True, help='Output JSON report path')
    return parser.parse_args()


def main():
    args = parse_args()
    gaze_files = find_gaze_files(args.gaze_root)
    if not gaze_files:
        print(f'[ERROR] No gaze files found under: {args.gaze_root}')
        return 1
    if args.focus_file:
        normalized_focus = os.path.normpath(args.focus_file)
        if normalized_focus in gaze_files:
            gaze_files = [normalized_focus] + [path for path in gaze_files if path != normalized_focus]
        else:
            print(f'[WARN] focus file not found under gaze_root: {args.focus_file}')

    gaze_index = build_gaze_file_index(args.gaze_root)
    inspected = []
    delimiter_counts = Counter()
    coord_counts = Counter()
    filetype_counts = Counter()
    gaze_type_counts = Counter()
    warnings_list = []

    print('=' * 80)
    print('EGTEA Gaze Inspection')
    print('=' * 80)
    print(f'gaze_root: {args.gaze_root}')
    print(f'num_files:  {len(gaze_files)}')
    print(f'index_keys: {len(gaze_index.key_to_files)}')
    print('-' * 80)

    for path in gaze_files[:args.max_files]:
        parsed = parse_gaze_file(path, max_records=2000)
        fmt = parsed.gaze_format
        inspected.append(
            dict(
                path=path,
                kind=fmt.kind,
                delimiter=fmt.delimiter,
                has_header=fmt.has_header,
                columns=fmt.columns,
                encoding=fmt.encoding,
                header=fmt.header,
                preview_rows=fmt.preview_rows,
                coordinate_mode=fmt.coordinate_mode,
                source_resolution=list(fmt.source_resolution),
                sampled_rows=fmt.sampled_rows,
                skipped_rows=fmt.skipped_rows,
                x_range=fmt.x_range,
                y_range=fmt.y_range,
                out_of_source_bounds_ratio=fmt.out_of_source_bounds_ratio,
                gaze_type_counts=fmt.gaze_type_counts,
                validity_counts=fmt.validity_counts,
                timestamp_unit=parsed.timestamp_unit,
                warnings=fmt.warnings,
            ))
        delimiter_counts[fmt.delimiter] += 1
        coord_counts[fmt.coordinate_mode] += 1
        filetype_counts[fmt.kind] += 1
        gaze_type_counts.update(fmt.gaze_type_counts)
        warnings_list.extend(fmt.warnings)
        if fmt.coordinate_mode == 'unknown':
            warnings_list.append(f'Coordinate mode unknown: {path}')
        print(f'[FILE] {path}')
        print(f'  kind={fmt.kind} delimiter={fmt.delimiter} header={fmt.has_header} encoding={fmt.encoding}')
        if fmt.header:
            print(f'  header={fmt.header}')
        if fmt.preview_rows:
            print(f'  preview_rows={fmt.preview_rows[:5]}')
        print(f'  columns={fmt.columns}')
        print(f'  sampled_rows={fmt.sampled_rows} skipped_rows={fmt.skipped_rows}')
        print(
            f'  source_resolution={list(fmt.source_resolution)} '
            f'coordinate_mode={fmt.coordinate_mode} '
            f'x_range={fmt.x_range} y_range={fmt.y_range} '
            f'out_of_source_bounds_ratio={fmt.out_of_source_bounds_ratio}')
        if fmt.gaze_type_counts:
            print(f'  gaze_types={dict(fmt.gaze_type_counts)}')
        if fmt.validity_counts:
            print(f'  validity={dict(fmt.validity_counts)}')

    report = dict(
        gaze_root=args.gaze_root,
        video_root=args.video_root,
        ann_root=args.ann_root,
        num_files=len(gaze_files),
        sample_files=gaze_files[:min(20, len(gaze_files))],
        detected_format_summary=dict(
            filetype_counts=dict(filetype_counts),
            delimiter_counts=dict(delimiter_counts),
            coordinate_modes=dict(coord_counts),
        ),
        coordinate_summary=dict(
            coordinate_modes=dict(coord_counts),
            inspected_files=len(inspected),
        ),
        gaze_type_summary=dict(gaze_type_counts),
        focus_file=args.focus_file,
        inspected=inspected,
        warnings=sorted(set(warnings_list)),
    )
    dump_json(report, args.output)
    print('-' * 80)
    print(f'[OK] Report written to: {args.output}')
    print(f'[OK] Unique warnings: {len(report["warnings"])}')
    print('=' * 80)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
