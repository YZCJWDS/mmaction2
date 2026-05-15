"""Collect EGTEA Gaze+ training / test results into a unified summary table.

Reads several MMEngine work directories on the cloud (default paths point to
the EGTEA Gaze+ project on /root/outputs/egtea_gaze) and produces three files:

- experiment_summary.csv   (machine-friendly)
- experiment_summary.md    (paper-friendly markdown)
- experiment_summary.json  (lossless full record)

Design rules:

1. Robust to missing logs / metrics / checkpoints. Missing fields become "NA".
2. Never overwrites by accident: writes new files into a fresh output dir or
   uses --overwrite; refuses to clobber otherwise.
3. Read-only with respect to every scanned work directory.
4. Supports MMEngine JSONL outputs:
   - <work_dir>/<ts>/vis_data/scalars.json
   - <work_dir>/<ts>/vis_data/<ts>.json
   - <work_dir>/<ts>/<ts>.json
   - <work_dir>/<ts>/<ts>.log   (regex fallback)

Typical usage on the cloud, after training has finished::

    python projects/egtea_gaze/tools/collect_experiment_results.py --pretty-print

To override the scan list::

    python projects/egtea_gaze/tools/collect_experiment_results.py \
        --root /root/outputs/egtea_gaze/gaze_slowfast_r50 \
        --root /root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults aligned with cloud paths under /root/outputs/egtea_gaze.
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = '/root/outputs/egtea_gaze'

DEFAULT_SCAN_DIRS: List[str] = [
    '/root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep',
    '/root/outputs/egtea_gaze/gaze_slowfast_r50',
    '/root/outputs/egtea_gaze/gaze_slowfast_r50_test_best',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_random',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_center',
    '/root/outputs/egtea_gaze/gaze_slowfast_ablation_shuffle',
]

# Pretty method labels for the paper table. Falls back to dir basename.
METHOD_LABELS: Dict[str, str] = {
    'slowfast_r50_bs24_amp_25ep': 'SlowFast-R50 (Baseline)',
    'gaze_slowfast_r50': 'Gaze-SlowFast (Ours)',
    'gaze_slowfast_r50_test_best': 'Gaze-SlowFast (Best-Ckpt Eval)',
    'gaze_slowfast_ablation_random': 'Gaze-SlowFast Abl. Random',
    'gaze_slowfast_ablation_center': 'Gaze-SlowFast Abl. Center',
    'gaze_slowfast_ablation_shuffle': 'Gaze-SlowFast Abl. Shuffle',
}

NA = 'NA'

CSV_FIELDS = [
    'method_name',
    'work_dir',
    'best_checkpoint_path',
    'latest_checkpoint_path',
    'best_epoch',
    'top1_acc',
    'top5_acc',
    'mean_class_accuracy',
    'loss_cls',
    'loss_gaze',
    'train_status',
]


@dataclass
class ExperimentRecord:
    method_name: Any = NA
    work_dir: Any = NA
    best_checkpoint_path: Any = NA
    latest_checkpoint_path: Any = NA
    best_epoch: Any = NA
    top1_acc: Any = NA  # stored as float in [0, 1] when available
    top5_acc: Any = NA
    mean_class_accuracy: Any = NA
    loss_cls: Any = NA
    loss_gaze: Any = NA
    train_status: Any = NA
    # Free-form diagnostics, not in CSV but kept in JSON.
    notes: List[str] = field(default_factory=list)
    sources: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _newest_match(directory: str, pattern: re.Pattern) -> Optional[str]:
    """Return the newest file in ``directory`` whose name matches ``pattern``."""
    if not os.path.isdir(directory):
        return None
    candidates = []
    for name in _safe_listdir(directory):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        if pattern.match(name):
            try:
                candidates.append((os.path.getmtime(full), full))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _list_timestamp_subdirs(work_dir: str) -> List[str]:
    """Find timestamped MMEngine subdirs (heuristic: name starts with digits)."""
    if not os.path.isdir(work_dir):
        return []
    out = []
    for name in _safe_listdir(work_dir):
        full = os.path.join(work_dir, name)
        if not os.path.isdir(full):
            continue
        if re.match(r'^\d{4,}', name):
            out.append(full)
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
             reverse=True)
    return out


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_BEST_TOP1_RE = re.compile(r'^best_acc_top1_.*\.pth$')
_BEST_GENERIC_RE = re.compile(r'^best_.*\.pth$')
_EPOCH_FROM_NAME_RE = re.compile(r'epoch_(\d+)')


def _discover_checkpoints(work_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (best_ckpt_path, latest_ckpt_path), each may be None."""
    if not os.path.isdir(work_dir):
        return None, None
    best = _newest_match(work_dir, _BEST_TOP1_RE)
    if best is None:
        best = _newest_match(work_dir, _BEST_GENERIC_RE)
    latest_path = os.path.join(work_dir, 'latest.pth')
    latest = latest_path if os.path.isfile(latest_path) else None
    if latest is None:
        # Fall back to newest epoch_*.pth so users still see something useful.
        latest = _newest_match(work_dir, re.compile(r'^epoch_\d+\.pth$'))
    return best, latest


def _epoch_from_ckpt_name(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    match = _EPOCH_FROM_NAME_RE.search(os.path.basename(path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Log discovery & parsing
# ---------------------------------------------------------------------------

def _discover_log_files(work_dir: str) -> Dict[str, List[str]]:
    """Return a dict with keys 'jsonl' and 'log' pointing at candidate files."""
    jsonl_files: List[str] = []
    log_files: List[str] = []
    if not os.path.isdir(work_dir):
        return dict(jsonl=jsonl_files, log=log_files)

    # 1) timestamped subdirs (most common for mmengine)
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
            if not os.path.isfile(full):
                continue
            if name.endswith('.json'):
                jsonl_files.append(full)
            elif name.endswith('.log'):
                log_files.append(full)

    # 2) Direct files at work_dir root (older layouts or test out dirs).
    for name in _safe_listdir(work_dir):
        full = os.path.join(work_dir, name)
        if not os.path.isfile(full):
            continue
        if name.endswith('.json') and name != 'metadata.json':
            jsonl_files.append(full)
        elif name.endswith('.log'):
            log_files.append(full)

    # Deduplicate while preserving order.
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return dict(jsonl=_dedupe(jsonl_files), log=_dedupe(log_files))


def _read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file. Returns [] on any error; never raises."""
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
        return rows
    return rows


def _is_val_entry(entry: dict) -> bool:
    return any(key in entry for key in ('acc/top1', 'acc/top5', 'acc/mean1'))


def _is_train_entry(entry: dict) -> bool:
    if 'loss_cls' in entry or 'loss_gaze' in entry:
        return True
    return 'loss' in entry and ('lr' in entry or 'data_time' in entry)


def _entry_epoch(entry: dict) -> Optional[int]:
    for key in ('epoch', 'step'):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _extract_metrics_from_jsonl(entries: List[dict]) -> Dict[str, Any]:
    """Extract best val metrics + last train losses + best epoch."""
    out: Dict[str, Any] = dict(
        best_top1=None, best_top5=None, best_mean=None, best_epoch=None,
        last_loss_cls=None, last_loss_gaze=None,
        n_val_entries=0, n_train_entries=0)

    best_idx = None
    best_top1 = -float('inf')
    val_entries: List[dict] = []
    train_entries: List[dict] = []

    for entry in entries:
        if _is_val_entry(entry):
            val_entries.append(entry)
            top1 = entry.get('acc/top1')
            if isinstance(top1, (int, float)) and top1 > best_top1:
                best_top1 = top1
                best_idx = len(val_entries) - 1
        elif _is_train_entry(entry):
            train_entries.append(entry)

    out['n_val_entries'] = len(val_entries)
    out['n_train_entries'] = len(train_entries)

    if best_idx is not None:
        best = val_entries[best_idx]
        out['best_top1'] = best.get('acc/top1')
        out['best_top5'] = best.get('acc/top5')
        out['best_mean'] = best.get('acc/mean1')
        out['best_epoch'] = _entry_epoch(best)

    if train_entries:
        last = train_entries[-1]
        out['last_loss_cls'] = last.get('loss_cls')
        out['last_loss_gaze'] = last.get('loss_gaze')

    return out


# ---------------------------------------------------------------------------
# Plain-text log fallback (used when JSON parsing yields nothing).
# ---------------------------------------------------------------------------

# mmengine logger lines look like:
#   Epoch(val) [2][20/20]    acc/top1: 0.6543  acc/top5: 0.8912  acc/mean1: 0.6021
# We do NOT try to capture all three metrics in one regex because each is
# optional in older / partial logs. Instead we identify val lines and search
# for each metric independently on that line.
_VAL_LINE_HINT_RE = re.compile(r'Epoch\(val\)\s*\[(\d+)\]')
_TEST_LINE_HINT_RE = re.compile(r'Epoch\(test\)|Test set|Eval')
_TOP1_RE = re.compile(r'\bacc/top1:\s*([0-9.eE+-]+)')
_TOP5_RE = re.compile(r'\bacc/top5:\s*([0-9.eE+-]+)')
_MEAN1_RE = re.compile(r'\bacc/mean1:\s*([0-9.eE+-]+)')
_LOSS_CLS_RE = re.compile(r'\bloss_cls:\s*([0-9.eE+-]+)')
_LOSS_GAZE_RE = re.compile(r'\bloss_gaze:\s*([0-9.eE+-]+)')
_TRACEBACK_RE = re.compile(r'^Traceback \(most recent call last\):', re.MULTILINE)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics_from_log_text(text: str) -> Dict[str, Any]:
    """Best-effort regex parse of a plain text training log."""
    out: Dict[str, Any] = dict(
        best_top1=None, best_top5=None, best_mean=None, best_epoch=None,
        last_loss_cls=None, last_loss_gaze=None, has_traceback=False)

    # Scan each line: keep top1 with the best value, and the corresponding
    # top5 / mean1 / epoch for that very line.
    best_top1 = -float('inf')
    for line in text.splitlines():
        # Only treat lines that actually carry a top1 reading.
        if 'acc/top1' not in line:
            continue
        top1_match = _TOP1_RE.search(line)
        if not top1_match:
            continue
        top1 = _safe_float(top1_match.group(1))
        if top1 is None:
            continue
        top5_match = _TOP5_RE.search(line)
        mean1_match = _MEAN1_RE.search(line)
        epoch_match = _VAL_LINE_HINT_RE.search(line)
        if top1 > best_top1:
            best_top1 = top1
            out['best_top1'] = top1
            out['best_top5'] = _safe_float(top5_match.group(1)) if top5_match else None
            out['best_mean'] = _safe_float(mean1_match.group(1)) if mean1_match else None
            if epoch_match:
                try:
                    out['best_epoch'] = int(epoch_match.group(1))
                except (TypeError, ValueError):
                    pass

    # Last loss values (scan whole text; finditer is cheap on bounded reads).
    for match in _LOSS_CLS_RE.finditer(text):
        out['last_loss_cls'] = _safe_float(match.group(1))
    for match in _LOSS_GAZE_RE.finditer(text):
        out['last_loss_gaze'] = _safe_float(match.group(1))

    out['has_traceback'] = bool(_TRACEBACK_RE.search(text))
    return out


def _read_text_file(path: str, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Read up to ``max_bytes`` from a log file, never raising."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ''
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            if size > max_bytes:
                # Read head + tail to keep eval lines and traceback while
                # avoiding loading huge per-iter logs entirely.
                head = handle.read(max_bytes // 2)
                handle.seek(max(0, size - max_bytes // 2))
                tail = handle.read(max_bytes // 2)
                return head + '\n' + tail
            return handle.read()
    except OSError:
        return ''


# ---------------------------------------------------------------------------
# Per-work-dir orchestration
# ---------------------------------------------------------------------------

def _merge_metric(primary: Any, fallback: Any) -> Any:
    """Prefer primary if not None, else fallback, else None."""
    return primary if primary is not None else fallback


def _classify_status(work_dir: str,
                     best_ckpt: Optional[str],
                     latest_ckpt: Optional[str],
                     metrics: Dict[str, Any],
                     has_log_or_json: bool,
                     has_traceback: bool) -> str:
    if not os.path.isdir(work_dir):
        return 'missing'
    if not has_log_or_json and best_ckpt is None and latest_ckpt is None:
        return 'empty'
    if best_ckpt is None and latest_ckpt is None and metrics.get('best_top1') is not None:
        # Test-only directory: no training ckpts here, only eval results.
        return 'test_only'
    if best_ckpt is not None:
        return 'completed'
    if latest_ckpt is not None or metrics.get('n_train_entries', 0) > 0:
        return 'failed' if has_traceback else 'in_progress'
    if has_traceback:
        return 'failed'
    return 'empty'


def collect_one(work_dir: str) -> ExperimentRecord:
    """Collect a single ExperimentRecord for ``work_dir``. Never raises."""
    record = ExperimentRecord()
    record.work_dir = work_dir
    base = os.path.basename(os.path.normpath(work_dir)) if work_dir else ''
    record.method_name = METHOD_LABELS.get(base, base or NA)

    if not work_dir or not os.path.isdir(work_dir):
        record.train_status = 'missing'
        record.notes.append(f'work_dir not found: {work_dir!r}')
        return record

    best_ckpt, latest_ckpt = _discover_checkpoints(work_dir)
    record.best_checkpoint_path = best_ckpt or NA
    record.latest_checkpoint_path = latest_ckpt or NA
    epoch_from_name = _epoch_from_ckpt_name(best_ckpt)

    log_paths = _discover_log_files(work_dir)
    record.sources = dict(jsonl=log_paths['jsonl'][:8], log=log_paths['log'][:8])

    # Try JSONL first (most reliable).
    json_metrics: Dict[str, Any] = dict()
    for path in log_paths['jsonl']:
        rows = _read_jsonl(path)
        if not rows:
            continue
        m = _extract_metrics_from_jsonl(rows)
        for key, value in m.items():
            if json_metrics.get(key) in (None, 0) and value is not None:
                json_metrics[key] = value
        # If we already saw a best top1, no need to keep merging deeper logs.
        if json_metrics.get('best_top1') is not None and json_metrics.get('last_loss_cls') is not None:
            break

    # Plain text log fallback (also used for traceback detection regardless).
    log_metrics: Dict[str, Any] = dict()
    has_traceback = False
    for path in log_paths['log']:
        text = _read_text_file(path)
        if not text:
            continue
        m = _extract_metrics_from_log_text(text)
        if m.pop('has_traceback', False):
            has_traceback = True
        for key, value in m.items():
            if log_metrics.get(key) in (None, 0) and value is not None:
                log_metrics[key] = value

    metrics: Dict[str, Any] = {
        'best_top1': _merge_metric(json_metrics.get('best_top1'),
                                   log_metrics.get('best_top1')),
        'best_top5': _merge_metric(json_metrics.get('best_top5'),
                                   log_metrics.get('best_top5')),
        'best_mean': _merge_metric(json_metrics.get('best_mean'),
                                   log_metrics.get('best_mean')),
        'best_epoch': _merge_metric(json_metrics.get('best_epoch'),
                                    log_metrics.get('best_epoch')),
        'last_loss_cls': _merge_metric(json_metrics.get('last_loss_cls'),
                                       log_metrics.get('last_loss_cls')),
        'last_loss_gaze': _merge_metric(json_metrics.get('last_loss_gaze'),
                                        log_metrics.get('last_loss_gaze')),
        'n_train_entries': json_metrics.get('n_train_entries', 0),
        'n_val_entries': json_metrics.get('n_val_entries', 0),
    }

    # Best epoch priority: ckpt filename -> json/log entry epoch -> NA.
    best_epoch = epoch_from_name if epoch_from_name is not None else metrics['best_epoch']
    record.best_epoch = best_epoch if best_epoch is not None else NA

    record.top1_acc = metrics['best_top1'] if metrics['best_top1'] is not None else NA
    record.top5_acc = metrics['best_top5'] if metrics['best_top5'] is not None else NA
    record.mean_class_accuracy = (
        metrics['best_mean'] if metrics['best_mean'] is not None else NA)
    record.loss_cls = (
        metrics['last_loss_cls'] if metrics['last_loss_cls'] is not None else NA)
    record.loss_gaze = (
        metrics['last_loss_gaze'] if metrics['last_loss_gaze'] is not None else NA)

    has_log_or_json = bool(log_paths['jsonl']) or bool(log_paths['log'])
    record.train_status = _classify_status(
        work_dir, best_ckpt, latest_ckpt, metrics, has_log_or_json, has_traceback)

    if metrics['best_top1'] is None:
        record.notes.append('no validation top1 found in logs')
    if has_traceback and best_ckpt is None:
        record.notes.append('traceback detected in log')
    if record.train_status == 'in_progress':
        record.notes.append('training may not be finished yet')

    return record


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f'{value * 100:.2f}%'
    return str(value) if value not in (None, '') else NA


def _fmt_loss(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f'{value:.4f}'
    return str(value) if value not in (None, '') else NA


def _record_csv_row(record: ExperimentRecord) -> Dict[str, Any]:
    row = {field: getattr(record, field) for field in CSV_FIELDS}
    # Numeric fields kept as floats in CSV for reproducibility.
    return row


def _record_json_dict(record: ExperimentRecord) -> Dict[str, Any]:
    payload = asdict(record)
    return payload


def _write_csv(records: List[ExperimentRecord], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_csv_row(record))


def _write_json(records: List[ExperimentRecord], path: str) -> None:
    payload = dict(
        records=[_record_json_dict(r) for r in records],
        fields=CSV_FIELDS,
    )
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_paper_table(records: List[ExperimentRecord]) -> str:
    """Compact markdown table designed to be pasted into a thesis chapter."""
    header = (
        '| Method | Top-1 | Top-5 | Mean-Class | Best Epoch | Status |\n'
        '|---|---:|---:|---:|---:|---|\n'
    )
    rows = []
    for record in records:
        rows.append(
            f'| {record.method_name} '
            f'| {_fmt_pct(record.top1_acc)} '
            f'| {_fmt_pct(record.top5_acc)} '
            f'| {_fmt_pct(record.mean_class_accuracy)} '
            f'| {record.best_epoch} '
            f'| {record.train_status} |'
        )
    return header + '\n'.join(rows) + '\n'


def _build_diagnostics_table(records: List[ExperimentRecord]) -> str:
    """Wider table including loss diagnostics and checkpoint paths."""
    header = (
        '| Method | Best Epoch | Top-1 | Top-5 | Mean-Class | loss_cls | loss_gaze | Status | Best Ckpt | Latest Ckpt | Work Dir |\n'
        '|---|---:|---:|---:|---:|---:|---:|---|---|---|---|\n'
    )
    rows = []
    for record in records:
        rows.append(
            '| ' + ' | '.join([
                str(record.method_name),
                str(record.best_epoch),
                _fmt_pct(record.top1_acc),
                _fmt_pct(record.top5_acc),
                _fmt_pct(record.mean_class_accuracy),
                _fmt_loss(record.loss_cls),
                _fmt_loss(record.loss_gaze),
                str(record.train_status),
                f'`{record.best_checkpoint_path}`',
                f'`{record.latest_checkpoint_path}`',
                f'`{record.work_dir}`',
            ]) + ' |'
        )
    return header + '\n'.join(rows) + '\n'


def _write_markdown(records: List[ExperimentRecord], path: str) -> None:
    parts = [
        '# EGTEA Gaze+ Experiment Summary\n',
        '## Main Results (paper table)\n',
        'Top-1 / Top-5 / Mean-Class are reported on the EGTEA Gaze+ test split.\n',
        _build_paper_table(records),
        '\n',
        '## Detailed Diagnostics\n',
        'For sanity-checking only: training-side loss values, checkpoint paths,\n',
        'and work-dir locations. Not intended for direct paste into the thesis.\n',
        _build_diagnostics_table(records),
        '\n',
        '## Notes per Experiment\n',
    ]
    for record in records:
        notes = record.notes or ['(none)']
        bullet_notes = '\n'.join(f'  - {n}' for n in notes)
        parts.append(f'- **{record.method_name}** (`{record.work_dir}`)\n{bullet_notes}\n')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(''.join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect EGTEA Gaze+ training/test results into a unified '
                    'summary table (CSV + Markdown + JSON).')
    parser.add_argument(
        '--root',
        action='append',
        default=None,
        help='Work directory to scan. Pass multiple times. Defaults to the '
             'six EGTEA cloud paths.')
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_ROOT,
        help='Where to write experiment_summary.{csv,md,json}. '
             f'Default: {DEFAULT_OUTPUT_ROOT}')
    parser.add_argument(
        '--prefix',
        default='experiment_summary',
        help='Output filename prefix. Default: experiment_summary')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing summary files. Off by default.')
    parser.add_argument(
        '--pretty-print',
        action='store_true',
        help='Print the paper-friendly markdown table to stdout.')
    return parser.parse_args(argv)


def _resolve_scan_dirs(arg_roots: Optional[List[str]]) -> List[str]:
    if not arg_roots:
        return list(DEFAULT_SCAN_DIRS)
    seen = set()
    out = []
    for item in arg_roots:
        norm = os.path.normpath(item)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    scan_dirs = _resolve_scan_dirs(args.root)

    records: List[ExperimentRecord] = []
    for work_dir in scan_dirs:
        try:
            record = collect_one(work_dir)
        except Exception as exc:  # noqa: BLE001 - we explicitly want to be safe.
            warnings.warn(f'collect_one failed for {work_dir!r}: {exc}')
            record = ExperimentRecord(
                method_name=METHOD_LABELS.get(
                    os.path.basename(os.path.normpath(work_dir)),
                    os.path.basename(os.path.normpath(work_dir)) or NA),
                work_dir=work_dir,
                train_status='error',
                notes=[f'unhandled exception: {exc!r}'])
        records.append(record)

    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(f'[ERROR] cannot create output dir {output_dir!r}: {exc}',
              file=sys.stderr)
        return 1

    csv_path = os.path.join(output_dir, f'{args.prefix}.csv')
    md_path = os.path.join(output_dir, f'{args.prefix}.md')
    json_path = os.path.join(output_dir, f'{args.prefix}.json')

    existing = [p for p in (csv_path, md_path, json_path) if os.path.exists(p)]
    if existing and not args.overwrite:
        print('[ERROR] refusing to overwrite existing files (pass --overwrite '
              'to force):', file=sys.stderr)
        for path in existing:
            print(f'  - {path}', file=sys.stderr)
        return 2

    try:
        _write_csv(records, csv_path)
        _write_markdown(records, md_path)
        _write_json(records, json_path)
    except OSError as exc:
        print(f'[ERROR] failed to write summary files: {exc}', file=sys.stderr)
        return 3

    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {md_path}')
    print(f'[OK] wrote {json_path}')

    if args.pretty_print:
        print()
        print('===== Paper Results Table =====')
        print(_build_paper_table(records))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
