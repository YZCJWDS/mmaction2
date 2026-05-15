"""Analyze offline gaze cache quality for thesis data-preprocessing chapter.

This script is **read-only**: it never modifies any cache file. It scans the
gaze_maps directory produced by make_gaze_maps.py and reports per-split and
per-class statistics that support the "gaze supervision validity" argument in
the paper.

Outputs (all written to --out-dir):
  - gaze_cache_stats.csv   (per-split summary)
  - gaze_cache_stats.json  (full detail including per-class breakdown)
  - gaze_cache_stats.md    (paper-friendly markdown)

Typical usage::

    python projects/egtea_gaze/tools/analyze_gaze_cache_stats.py --pretty-print

Override paths::

    python projects/egtea_gaze/tools/analyze_gaze_cache_stats.py \
        --gaze-map-root /cloud/egtea/gaze_maps \
        --ann-root /cloud/egtea/action_annotation \
        --out-dir /root/outputs/egtea_gaze/gaze_cache_stats \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (cloud paths)
# ---------------------------------------------------------------------------
DEFAULT_GAZE_MAP_ROOT = '/root/data/egtea/gaze_maps'
DEFAULT_ANN_ROOT = '/root/data/egtea/action_annotation'
DEFAULT_OUT_DIR = '/root/outputs/egtea_gaze/gaze_cache_stats'

NA = 'NA'

SPLIT_FILES = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt',
}

CSV_FIELDS = [
    'split',
    'num_samples',
    'valid_gaze_samples',
    'invalid_gaze_samples',
    'valid_sample_ratio',
    'total_gaze_frames',
    'valid_gaze_frames',
    'valid_frame_ratio',
    'heatmap_shape',
    'gaze_valid_shape',
    'nan_count',
    'inf_count',
    'bad_files',
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    split: str = ''
    num_samples: int = 0
    valid_gaze_samples: int = 0
    invalid_gaze_samples: int = 0
    valid_sample_ratio: float = 0.0
    total_gaze_frames: int = 0
    valid_gaze_frames: int = 0
    valid_frame_ratio: float = 0.0
    heatmap_shape: str = NA
    gaze_valid_shape: str = NA
    nan_count: int = 0
    inf_count: int = 0
    bad_files: int = 0
    bad_file_list: List[str] = field(default_factory=list)
    # Per-class breakdown: class_label -> {total_samples, valid_samples, valid_ratio}
    per_class: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_ann_file(path: str) -> List[Dict[str, str]]:
    """Parse an MMAction2-style annotation file (video_path label).

    Returns list of dicts with keys: video_path, label.
    """
    entries: List[Dict[str, str]] = []
    if not os.path.isfile(path):
        return entries
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) == 2:
                    entries.append(dict(video_path=parts[0], label=parts[1]))
                elif len(parts) == 1:
                    entries.append(dict(video_path=parts[0], label='unknown'))
    except OSError:
        pass
    return entries


def _load_label_mapping(ann_root: str) -> Dict[str, str]:
    """Try to load a label index file for human-readable class names.

    Looks for: action_idx.txt, label_mapping_generated.txt, or cls_label_index.csv.
    Returns {label_id_str: class_name}.
    """
    mapping: Dict[str, str] = {}

    # Try action_idx.txt (format: "action_name action_id")
    action_idx = os.path.join(ann_root, 'action_idx.txt')
    if os.path.isfile(action_idx):
        try:
            with open(action_idx, 'r', encoding='utf-8') as handle:
                for line in handle:
                    parts = line.strip().rsplit(maxsplit=1)
                    if len(parts) == 2:
                        mapping[parts[1]] = parts[0]
        except OSError:
            pass
        if mapping:
            return mapping

    # Try label_mapping_generated.txt (same format)
    label_map = os.path.join(ann_root, 'label_mapping_generated.txt')
    if os.path.isfile(label_map):
        try:
            with open(label_map, 'r', encoding='utf-8') as handle:
                for line in handle:
                    parts = line.strip().rsplit(maxsplit=1)
                    if len(parts) == 2:
                        mapping[parts[1]] = parts[0]
        except OSError:
            pass
        if mapping:
            return mapping

    # Try cls_label_index.csv
    cls_csv = os.path.join(ann_root, 'raw_annotations', 'cls_label_index.csv')
    if os.path.isfile(cls_csv):
        try:
            with open(cls_csv, 'r', encoding='utf-8') as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) >= 2:
                        mapping[row[0].strip()] = row[1].strip()
        except (OSError, csv.Error):
            pass

    return mapping


# ---------------------------------------------------------------------------
# Cache analysis
# ---------------------------------------------------------------------------

def _resolve_npz_for_video(video_path: str, metadata_samples: Dict[str, Any],
                           gaze_map_root: str) -> Optional[str]:
    """Resolve the .npz cache file for a given video_path using metadata."""
    # Normalize path separators.
    normalized = video_path.replace('\\', '/')

    # Try direct lookup.
    if normalized in metadata_samples:
        rel = metadata_samples[normalized].get('npz_path', '')
        full = os.path.join(gaze_map_root, rel)
        if os.path.isfile(full):
            return full

    # Try common suffixes.
    for candidate in [normalized]:
        # Strip leading prefixes that metadata might not have.
        for prefix in ('cropped_clips/', 'videos/cropped_clips/', 'videos/'):
            if candidate.startswith(prefix):
                stripped = candidate[len(prefix):]
                if stripped in metadata_samples:
                    rel = metadata_samples[stripped].get('npz_path', '')
                    full = os.path.join(gaze_map_root, rel)
                    if os.path.isfile(full):
                        return full

    return None


def _analyze_npz(npz_path: str) -> Dict[str, Any]:
    """Analyze a single .npz cache file. Returns stats dict. Never raises."""
    result: Dict[str, Any] = dict(
        valid=False,
        total_frames=0,
        valid_frames=0,
        nan_count=0,
        inf_count=0,
        heatmap_shape=None,
        gaze_valid_shape=None,
        error=None,
    )
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        result['error'] = f'load failed: {exc}'
        return result

    gaze_valid = data.get('gaze_valid')
    gaze_maps = data.get('gaze_maps')
    gaze_xy = data.get('gaze_xy')

    if gaze_valid is None:
        result['error'] = 'missing gaze_valid key'
        return result

    gaze_valid_arr = np.asarray(gaze_valid)
    result['gaze_valid_shape'] = str(tuple(gaze_valid_arr.shape))
    result['total_frames'] = int(gaze_valid_arr.size)
    result['valid_frames'] = int((gaze_valid_arr > 0.5).sum())
    result['valid'] = result['valid_frames'] > 0

    if gaze_maps is not None:
        maps_arr = np.asarray(gaze_maps, dtype=np.float32)
        result['heatmap_shape'] = str(tuple(maps_arr.shape))
        result['nan_count'] = int(np.isnan(maps_arr).sum())
        result['inf_count'] = int(np.isinf(maps_arr).sum())
    elif gaze_xy is not None:
        xy_arr = np.asarray(gaze_xy, dtype=np.float32)
        result['heatmap_shape'] = f'xy:{tuple(xy_arr.shape)}'
        result['nan_count'] = int(np.isnan(xy_arr).sum())
        result['inf_count'] = int(np.isinf(xy_arr).sum())

    return result


def analyze_split(split_name: str,
                  ann_entries: List[Dict[str, str]],
                  metadata_samples: Dict[str, Any],
                  gaze_map_root: str,
                  label_mapping: Dict[str, str]) -> SplitStats:
    """Analyze all samples in one split."""
    stats = SplitStats(split=split_name)
    stats.num_samples = len(ann_entries)

    # Per-class accumulators.
    class_total: Dict[str, int] = defaultdict(int)
    class_valid: Dict[str, int] = defaultdict(int)

    representative_heatmap_shape: Optional[str] = None
    representative_gaze_valid_shape: Optional[str] = None

    for entry in ann_entries:
        video_path = entry['video_path']
        label = entry.get('label', 'unknown')
        class_total[label] += 1

        npz_path = _resolve_npz_for_video(video_path, metadata_samples, gaze_map_root)
        if npz_path is None:
            stats.invalid_gaze_samples += 1
            stats.bad_files += 1
            stats.bad_file_list.append(f'not_found:{video_path}')
            continue

        info = _analyze_npz(npz_path)
        if info.get('error'):
            stats.invalid_gaze_samples += 1
            stats.bad_files += 1
            stats.bad_file_list.append(f'{info["error"]}:{npz_path}')
            continue

        stats.total_gaze_frames += info['total_frames']
        stats.valid_gaze_frames += info['valid_frames']
        stats.nan_count += info['nan_count']
        stats.inf_count += info['inf_count']

        if info['valid']:
            stats.valid_gaze_samples += 1
            class_valid[label] += 1
        else:
            stats.invalid_gaze_samples += 1

        if representative_heatmap_shape is None and info['heatmap_shape']:
            representative_heatmap_shape = info['heatmap_shape']
        if representative_gaze_valid_shape is None and info['gaze_valid_shape']:
            representative_gaze_valid_shape = info['gaze_valid_shape']

    stats.heatmap_shape = representative_heatmap_shape or NA
    stats.gaze_valid_shape = representative_gaze_valid_shape or NA

    if stats.num_samples > 0:
        stats.valid_sample_ratio = stats.valid_gaze_samples / stats.num_samples
    if stats.total_gaze_frames > 0:
        stats.valid_frame_ratio = stats.valid_gaze_frames / stats.total_gaze_frames

    # Build per-class breakdown.
    for label_id in sorted(class_total.keys(), key=lambda x: int(x) if x.isdigit() else x):
        total = class_total[label_id]
        valid = class_valid.get(label_id, 0)
        class_name = label_mapping.get(label_id, f'class_{label_id}')
        stats.per_class[label_id] = dict(
            class_name=class_name,
            total_samples=total,
            valid_samples=valid,
            valid_ratio=valid / total if total > 0 else 0.0,
        )

    # Cap bad_file_list for JSON sanity.
    if len(stats.bad_file_list) > 50:
        stats.bad_file_list = stats.bad_file_list[:50] + [
            f'... and {len(stats.bad_file_list) - 50} more']

    return stats


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _fmt_pct(value: float) -> str:
    return f'{value * 100:.2f}%'


def write_csv(all_stats: List[SplitStats], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for stats in all_stats:
            writer.writerow({
                'split': stats.split,
                'num_samples': stats.num_samples,
                'valid_gaze_samples': stats.valid_gaze_samples,
                'invalid_gaze_samples': stats.invalid_gaze_samples,
                'valid_sample_ratio': f'{stats.valid_sample_ratio:.4f}',
                'total_gaze_frames': stats.total_gaze_frames,
                'valid_gaze_frames': stats.valid_gaze_frames,
                'valid_frame_ratio': f'{stats.valid_frame_ratio:.4f}',
                'heatmap_shape': stats.heatmap_shape,
                'gaze_valid_shape': stats.gaze_valid_shape,
                'nan_count': stats.nan_count,
                'inf_count': stats.inf_count,
                'bad_files': stats.bad_files,
            })


def write_json(all_stats: List[SplitStats], path: str) -> None:
    payload = dict(
        splits=[asdict(s) for s in all_stats],
        fields=CSV_FIELDS,
    )
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _build_markdown(all_stats: List[SplitStats]) -> str:
    parts: List[str] = []
    parts.append('# Gaze Cache Quality Report\n\n')
    parts.append('## Per-Split Summary\n\n')
    parts.append('| Split | Samples | Valid Samples | Valid Ratio | '
                 'Total Frames | Valid Frames | Frame Valid Ratio | '
                 'NaN | Inf | Bad Files |\n')
    parts.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for s in all_stats:
        parts.append(
            f'| {s.split} | {s.num_samples} | {s.valid_gaze_samples} | '
            f'{_fmt_pct(s.valid_sample_ratio)} | {s.total_gaze_frames} | '
            f'{s.valid_gaze_frames} | {_fmt_pct(s.valid_frame_ratio)} | '
            f'{s.nan_count} | {s.inf_count} | {s.bad_files} |\n')

    parts.append('\n## Interpretation for Thesis\n\n')
    parts.append('- **Valid Sample Ratio**: fraction of clips with at least one valid gaze frame.\n')
    parts.append('- **Frame Valid Ratio**: fraction of individual frames with usable gaze coordinates.\n')
    parts.append('- A high valid sample ratio supports the claim that gaze supervision is broadly applicable.\n')
    parts.append('- NaN/Inf counts should be 0; non-zero indicates cache corruption.\n')

    # Per-class table for the first split that has per_class data.
    for s in all_stats:
        if not s.per_class:
            continue
        parts.append(f'\n## Per-Class Valid Gaze Ratio ({s.split} split)\n\n')
        parts.append('| Label | Class Name | Total | Valid | Valid Ratio |\n')
        parts.append('|---:|---|---:|---:|---:|\n')
        for label_id, info in s.per_class.items():
            parts.append(
                f'| {label_id} | {info["class_name"]} | {info["total_samples"]} | '
                f'{info["valid_samples"]} | {_fmt_pct(info["valid_ratio"])} |\n')
        break  # Only show one split's per-class table to keep markdown concise.

    return ''.join(parts)


def write_markdown(all_stats: List[SplitStats], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(_build_markdown(all_stats))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Analyze gaze cache quality (read-only). '
                    'Produces per-split and per-class statistics for the thesis.')
    parser.add_argument(
        '--gaze-map-root', default=DEFAULT_GAZE_MAP_ROOT,
        help=f'Root of gaze cache directory. Default: {DEFAULT_GAZE_MAP_ROOT}')
    parser.add_argument(
        '--ann-root', default=DEFAULT_ANN_ROOT,
        help=f'Annotation directory with train/val/test.txt. Default: {DEFAULT_ANN_ROOT}')
    parser.add_argument(
        '--out-dir', default=DEFAULT_OUT_DIR,
        help=f'Output directory. Default: {DEFAULT_OUT_DIR}')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow overwriting existing output files.')
    parser.add_argument(
        '--pretty-print', action='store_true',
        help='Print markdown summary to stdout.')
    parser.add_argument(
        '--splits', nargs='+', default=None,
        help='Splits to analyze. Default: train val test.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gaze_map_root = args.gaze_map_root
    ann_root = args.ann_root
    out_dir = args.out_dir

    # Load metadata.json from gaze cache.
    metadata_path = os.path.join(gaze_map_root, 'metadata.json')
    metadata_samples: Dict[str, Any] = {}
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)
            metadata_samples = metadata.get('samples', {})
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f'Failed to read metadata.json: {exc}')
    else:
        warnings.warn(f'metadata.json not found at {metadata_path}. '
                      'NPZ resolution will rely on filesystem search.')

    if not os.path.isdir(gaze_map_root):
        print(f'[WARN] gaze-map-root not found: {gaze_map_root}', file=sys.stderr)
        print('[WARN] gaze cache may not have been generated yet.', file=sys.stderr)
        # Still produce empty outputs.

    # Load label mapping for per-class names.
    label_mapping = _load_label_mapping(ann_root)

    # Determine splits.
    splits_to_run = args.splits or list(SPLIT_FILES.keys())

    all_stats: List[SplitStats] = []
    for split_name in splits_to_run:
        ann_filename = SPLIT_FILES.get(split_name, f'{split_name}.txt')
        ann_path = os.path.join(ann_root, ann_filename)
        if not os.path.isfile(ann_path):
            warnings.warn(f'Annotation file not found: {ann_path}')
            s = SplitStats(split=split_name)
            s.heatmap_shape = NA
            s.gaze_valid_shape = NA
            all_stats.append(s)
            continue

        ann_entries = _parse_ann_file(ann_path)
        if not ann_entries:
            warnings.warn(f'No entries parsed from {ann_path}')

        print(f'[INFO] analyzing split={split_name}, samples={len(ann_entries)} ...')
        stats = analyze_split(
            split_name, ann_entries, metadata_samples, gaze_map_root, label_mapping)
        all_stats.append(stats)
        print(f'       valid_samples={stats.valid_gaze_samples}/{stats.num_samples} '
              f'({_fmt_pct(stats.valid_sample_ratio)}), '
              f'valid_frames={stats.valid_gaze_frames}/{stats.total_gaze_frames} '
              f'({_fmt_pct(stats.valid_frame_ratio)}), '
              f'bad_files={stats.bad_files}')

    # Write outputs.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f'[ERROR] cannot create output dir: {exc}', file=sys.stderr)
        return 1

    csv_path = os.path.join(out_dir, 'gaze_cache_stats.csv')
    json_path = os.path.join(out_dir, 'gaze_cache_stats.json')
    md_path = os.path.join(out_dir, 'gaze_cache_stats.md')

    existing = [p for p in (csv_path, json_path, md_path) if os.path.exists(p)]
    if existing and not args.overwrite:
        print('[ERROR] refusing to overwrite existing files (pass --overwrite):',
              file=sys.stderr)
        for p in existing:
            print(f'  - {p}', file=sys.stderr)
        return 2

    try:
        write_csv(all_stats, csv_path)
        write_json(all_stats, json_path)
        write_markdown(all_stats, md_path)
    except OSError as exc:
        print(f'[ERROR] write failed: {exc}', file=sys.stderr)
        return 3

    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {json_path}')
    print(f'[OK] wrote {md_path}')

    if args.pretty_print:
        print()
        print(_build_markdown(all_stats))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
