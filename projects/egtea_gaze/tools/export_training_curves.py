"""Export training curves from MMEngine work directories for thesis figures.

Reads JSONL logs (vis_data/scalars.json or timestamped .json files) and
produces:

1. One CSV per experiment with all parsed metrics per epoch/step.
2. One PNG per key metric, overlaying all experiments on the same axes.

Uses matplotlib only (no seaborn). Each figure is saved individually.

Typical usage::

    python projects/egtea_gaze/tools/export_training_curves.py --pretty-print
    python projects/egtea_gaze/tools/export_training_curves.py \
        --work-dir /root/outputs/egtea_gaze/gaze_slowfast_r50 \
        --out-dir /root/outputs/egtea_gaze/training_curves --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (cloud paths)
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = '/root/outputs/egtea_gaze/training_curves'

DEFAULT_WORK_DIRS: List[str] = [
    '/root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep',
    '/root/outputs/egtea_gaze/gaze_slowfast_r50',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_random',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_center',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_shuffle',
]

# Metrics to extract and plot.
TRAIN_METRICS = ['loss', 'loss_cls', 'loss_gaze', 'lr', 'data_time', 'time']
VAL_METRICS = ['acc/top1', 'acc/top5', 'acc/mean1']

# All metrics we care about (union).
ALL_METRICS = TRAIN_METRICS + VAL_METRICS

# Pretty labels for plots.
METRIC_LABELS: Dict[str, Dict[str, str]] = {
    'loss': dict(title='Total Loss', ylabel='Loss'),
    'loss_cls': dict(title='Classification Loss', ylabel='Loss'),
    'loss_gaze': dict(title='Gaze KL Loss', ylabel='Loss'),
    'lr': dict(title='Learning Rate', ylabel='LR'),
    'data_time': dict(title='Data Loading Time', ylabel='Seconds'),
    'time': dict(title='Iteration Time', ylabel='Seconds'),
    'acc/top1': dict(title='Top-1 Accuracy (Val)', ylabel='Accuracy'),
    'acc/top5': dict(title='Top-5 Accuracy (Val)', ylabel='Accuracy'),
    'acc/mean1': dict(title='Mean Class Accuracy (Val)', ylabel='Accuracy'),
}

# Display names for experiments (by work_dir basename).
METHOD_LABELS: Dict[str, str] = {
    'slowfast_r50_bs24_amp_25ep': 'SlowFast-R50 (Baseline)',
    'gaze_slowfast_r50': 'Gaze-SlowFast (Ours)',
    'gaze_slowfast_ablation_random': 'Abl. Random Gaze',
    'gaze_slowfast_ablation_center': 'Abl. Center Gaze',
    'gaze_slowfast_ablation_shuffle': 'Abl. Shuffle Gaze',
}

# CSV column order.
CSV_COLUMNS = ['experiment', 'epoch', 'step'] + ALL_METRICS


# ---------------------------------------------------------------------------
# Log discovery & parsing (reused logic from collect_experiment_results.py)
# ---------------------------------------------------------------------------

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _list_timestamp_subdirs(work_dir: str) -> List[str]:
    if not os.path.isdir(work_dir):
        return []
    out = []
    for name in _safe_listdir(work_dir):
        full = os.path.join(work_dir, name)
        if os.path.isdir(full) and re.match(r'^\d{4,}', name):
            out.append(full)
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
             reverse=True)
    return out


def _discover_jsonl_files(work_dir: str) -> List[str]:
    """Find all JSONL log files in a work directory."""
    jsonl_files: List[str] = []
    if not os.path.isdir(work_dir):
        return jsonl_files

    for ts_dir in _list_timestamp_subdirs(work_dir):
        vis_dir = os.path.join(ts_dir, 'vis_data')
        if os.path.isdir(vis_dir):
            scalars = os.path.join(vis_dir, 'scalars.json')
            if os.path.isfile(scalars):
                jsonl_files.append(scalars)
            for name in _safe_listdir(vis_dir):
                if name.endswith('.json') and name != 'scalars.json':
                    jsonl_files.append(os.path.join(vis_dir, name))
        for name in _safe_listdir(ts_dir):
            full = os.path.join(ts_dir, name)
            if os.path.isfile(full) and name.endswith('.json'):
                jsonl_files.append(full)

    # Direct files at root.
    for name in _safe_listdir(work_dir):
        full = os.path.join(work_dir, name)
        if os.path.isfile(full) and name.endswith('.json') and name != 'metadata.json':
            jsonl_files.append(full)

    # Deduplicate.
    seen = set()
    unique = []
    for path in jsonl_files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_epoch(entry: dict) -> Optional[float]:
    """Get epoch from an entry. May be fractional for per-iter logs."""
    epoch = entry.get('epoch')
    if isinstance(epoch, (int, float)):
        return float(epoch)
    return None


def _extract_step(entry: dict) -> Optional[int]:
    step = entry.get('step') or entry.get('iter')
    if isinstance(step, (int, float)):
        return int(step)
    return None


def parse_work_dir(work_dir: str) -> List[dict]:
    """Parse all JSONL entries from a work directory into a flat list of records.

    Each record has: epoch, step, and any metric keys present.
    """
    jsonl_files = _discover_jsonl_files(work_dir)
    if not jsonl_files:
        return []

    all_entries: List[dict] = []
    for path in jsonl_files:
        rows = _read_jsonl(path)
        all_entries.extend(rows)

    # Deduplicate by (epoch, step, metric presence).
    records: List[dict] = []
    for entry in all_entries:
        epoch = _extract_epoch(entry)
        step = _extract_step(entry)
        record: dict = {'epoch': epoch, 'step': step}
        has_metric = False
        for metric in ALL_METRICS:
            value = entry.get(metric)
            if isinstance(value, (int, float)):
                record[metric] = float(value)
                has_metric = True
        if has_metric:
            records.append(record)

    # Sort by epoch then step.
    records.sort(key=lambda r: (r.get('epoch') or 0, r.get('step') or 0))
    return records


def aggregate_by_epoch(records: List[dict]) -> Dict[str, List[Tuple[float, float]]]:
    """Aggregate records into per-metric series: metric -> [(epoch, value), ...].

    For train metrics (multiple entries per epoch), take the mean per epoch.
    For val metrics (one entry per epoch), take as-is.
    """
    # Group by epoch.
    epoch_groups: Dict[float, List[dict]] = defaultdict(list)
    for record in records:
        epoch = record.get('epoch')
        if epoch is None:
            continue
        epoch_groups[epoch].append(record)

    series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for epoch in sorted(epoch_groups.keys()):
        group = epoch_groups[epoch]
        for metric in ALL_METRICS:
            values = [r[metric] for r in group if metric in r]
            if not values:
                continue
            # Val metrics: take the single value (or last if duplicated).
            if metric in VAL_METRICS:
                series[metric].append((epoch, values[-1]))
            else:
                # Train metrics: average over the epoch.
                series[metric].append((epoch, float(np.mean(values))))

    return dict(series)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_experiment_csv(experiment_name: str, records: List[dict],
                         out_dir: str) -> str:
    """Write one CSV for a single experiment. Returns the output path."""
    safe_name = re.sub(r'[^\w\-]', '_', experiment_name)
    csv_path = os.path.join(out_dir, f'{safe_name}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for record in records:
            row = {'experiment': experiment_name}
            row['epoch'] = record.get('epoch', '')
            row['step'] = record.get('step', '')
            for metric in ALL_METRICS:
                row[metric] = record.get(metric, '')
            writer.writerow(row)
    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _import_matplotlib():
    """Import matplotlib with Agg backend (no display needed on cloud)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_metric(metric: str,
                all_series: Dict[str, List[Tuple[float, float]]],
                out_dir: str,
                dpi: int = 150) -> Optional[str]:
    """Plot a single metric across all experiments. Returns output path or None."""
    # Filter experiments that have this metric.
    available = {name: series for name, series in all_series.items() if series}
    if not available:
        return None

    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    meta = METRIC_LABELS.get(metric, dict(title=metric, ylabel=metric))

    for exp_name, series in available.items():
        epochs = [p[0] for p in series]
        values = [p[1] for p in series]
        label = METHOD_LABELS.get(exp_name, exp_name)
        ax.plot(epochs, values, marker='o', markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(meta['ylabel'], fontsize=11)
    ax.set_title(meta['title'], fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_metric = metric.replace('/', '_')
    png_path = os.path.join(out_dir, f'curve_{safe_metric}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return png_path


def plot_combined_loss(all_series_by_exp: Dict[str, Dict[str, List[Tuple[float, float]]]],
                       out_dir: str, dpi: int = 150) -> Optional[str]:
    """Plot loss_cls and loss_gaze on the same axes for the gaze model."""
    plt = _import_matplotlib()

    # Find experiments that have both loss_cls and loss_gaze.
    candidates = {}
    for exp_name, series_dict in all_series_by_exp.items():
        if 'loss_cls' in series_dict and 'loss_gaze' in series_dict:
            candidates[exp_name] = series_dict

    if not candidates:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0

    for exp_name, series_dict in candidates.items():
        label = METHOD_LABELS.get(exp_name, exp_name)
        color = colors[color_idx % len(colors)]
        color_idx += 1

        cls_series = series_dict['loss_cls']
        gaze_series = series_dict['loss_gaze']

        epochs_cls = [p[0] for p in cls_series]
        values_cls = [p[1] for p in cls_series]
        epochs_gaze = [p[0] for p in gaze_series]
        values_gaze = [p[1] for p in gaze_series]

        ax.plot(epochs_cls, values_cls, linestyle='-', color=color,
                marker='o', markersize=3, linewidth=1.5,
                label=f'{label} - loss_cls')
        ax.plot(epochs_gaze, values_gaze, linestyle='--', color=color,
                marker='s', markersize=3, linewidth=1.5,
                label=f'{label} - loss_gaze')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Classification Loss vs Gaze Loss', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(out_dir, 'curve_loss_cls_vs_gaze.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Export training curves from MMEngine work directories.')
    parser.add_argument(
        '--work-dir', action='append', default=None,
        help='Work directory to scan. Pass multiple times. '
             'Defaults to the five EGTEA cloud paths.')
    parser.add_argument(
        '--out-dir', default=DEFAULT_OUT_DIR,
        help=f'Output directory for CSVs and PNGs. Default: {DEFAULT_OUT_DIR}')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow overwriting existing output directory contents.')
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for saved PNG figures. Default: 150.')
    parser.add_argument(
        '--pretty-print', action='store_true',
        help='Print a summary of exported files to stdout.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    work_dirs = args.work_dir if args.work_dir else list(DEFAULT_WORK_DIRS)
    out_dir = args.out_dir

    # Check output directory.
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not args.overwrite:
            print(f'[ERROR] output directory not empty: {out_dir}', file=sys.stderr)
            print('[HINT] pass --overwrite to allow.', file=sys.stderr)
            return 2
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f'[ERROR] cannot create output dir: {exc}', file=sys.stderr)
        return 1

    # Parse all experiments.
    all_records: Dict[str, List[dict]] = {}
    all_series: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for work_dir in work_dirs:
        exp_name = os.path.basename(os.path.normpath(work_dir))
        if not os.path.isdir(work_dir):
            warnings.warn(f'work_dir not found, skipping: {work_dir}')
            continue

        records = parse_work_dir(work_dir)
        if not records:
            warnings.warn(f'no log entries found in: {work_dir}')
            continue

        all_records[exp_name] = records
        all_series[exp_name] = aggregate_by_epoch(records)
        print(f'[INFO] {exp_name}: {len(records)} log entries, '
              f'{len(all_series[exp_name])} metrics with data')

    if not all_records:
        print('[WARN] no data found in any work directory. '
              'Training may not have started yet.', file=sys.stderr)
        # Still exit cleanly.
        return 0

    # Write per-experiment CSVs.
    csv_paths: List[str] = []
    for exp_name, records in all_records.items():
        path = write_experiment_csv(exp_name, records, out_dir)
        csv_paths.append(path)

    # Plot each metric across all experiments.
    png_paths: List[str] = []
    for metric in ALL_METRICS:
        # Gather this metric's series from all experiments.
        metric_series: Dict[str, List[Tuple[float, float]]] = {}
        for exp_name, series_dict in all_series.items():
            if metric in series_dict and series_dict[metric]:
                metric_series[exp_name] = series_dict[metric]

        if not metric_series:
            warnings.warn(f'metric "{metric}" not found in any experiment, skipping plot.')
            continue

        path = plot_metric(metric, metric_series, out_dir, dpi=args.dpi)
        if path:
            png_paths.append(path)

    # Bonus: combined loss_cls vs loss_gaze plot.
    combined_path = plot_combined_loss(all_series, out_dir, dpi=args.dpi)
    if combined_path:
        png_paths.append(combined_path)

    # Summary.
    print(f'\n[OK] exported {len(csv_paths)} CSVs and {len(png_paths)} PNGs to: {out_dir}')

    if args.pretty_print:
        print('\n===== Exported Files =====')
        print('\nCSVs:')
        for p in csv_paths:
            print(f'  {p}')
        print('\nPNGs:')
        for p in png_paths:
            print(f'  {p}')
        print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
