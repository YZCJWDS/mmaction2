"""Generate the main paper comparison table: SlowFast baseline vs Gaze-SlowFast.

Reads the unified experiment_summary.csv produced by collect_experiment_results.py
and outputs three files:

- baseline_vs_gaze.csv       (machine-friendly)
- baseline_vs_gaze.md        (markdown for thesis chapter)
- baseline_vs_gaze_latex.txt (LaTeX tabular, ready to paste into paper)

Key design points:

1. Test input is ALWAYS "RGB" for every row. Gaze is training-time supervision
   only and is never used at inference.
2. Delta columns are computed relative to the SlowFast baseline row.
3. Missing metrics are shown as "NA"; deltas become "NA" when either side is
   missing.
4. Never overwrites existing files unless --overwrite is passed.
5. Reads from experiment_summary.csv by default; path overridable via --input.

Usage::

    python projects/egtea_gaze/tools/compare_baseline_gaze.py --pretty-print
    python projects/egtea_gaze/tools/compare_baseline_gaze.py --input /path/to/summary.csv --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Defaults (cloud paths)
# ---------------------------------------------------------------------------
DEFAULT_INPUT = '/root/outputs/egtea_gaze/experiment_summary.csv'
DEFAULT_OUTPUT_DIR = '/root/outputs/egtea_gaze'
DEFAULT_PREFIX = 'baseline_vs_gaze'

NA = 'NA'

# ---------------------------------------------------------------------------
# Method metadata: maps work_dir basename -> display info.
# "supervision" describes what the model sees during TRAINING.
# "test_input" is always RGB because gaze is never used at inference.
# ---------------------------------------------------------------------------
METHOD_META: Dict[str, Dict[str, str]] = {
    'slowfast_r50_bs24_amp_25ep': dict(
        method='SlowFast-R50',
        test_input='RGB',
        supervision='Cross-Entropy (cls only)',
        is_baseline=True,
    ),
    'gaze_slowfast_r50': dict(
        method='Gaze-SlowFast-R50 (Ours)',
        test_input='RGB',
        supervision='Cross-Entropy + Real Gaze KL',
        is_baseline=False,
    ),
    'gaze_slowfast_r50_test_best': dict(
        method='Gaze-SlowFast-R50 (Best Ckpt)',
        test_input='RGB',
        supervision='Cross-Entropy + Real Gaze KL',
        is_baseline=False,
    ),
    'gaze_slowfast_ablation_random': dict(
        method='Gaze-SlowFast Abl. Random',
        test_input='RGB',
        supervision='Cross-Entropy + Random Gaze KL',
        is_baseline=False,
    ),
    'gaze_slowfast_ablation_center': dict(
        method='Gaze-SlowFast Abl. Center',
        test_input='RGB',
        supervision='Cross-Entropy + Center Gaze KL',
        is_baseline=False,
    ),
    'gaze_slowfast_ablation_shuffle': dict(
        method='Gaze-SlowFast Abl. Shuffle',
        test_input='RGB',
        supervision='Cross-Entropy + Shuffled Gaze KL',
        is_baseline=False,
    ),
}

# Ordered list of basenames controlling table row order.
ROW_ORDER = [
    'slowfast_r50_bs24_amp_25ep',
    'gaze_slowfast_r50',
    'gaze_slowfast_r50_test_best',
    'gaze_slowfast_ablation_random',
    'gaze_slowfast_ablation_center',
    'gaze_slowfast_ablation_shuffle',
]

OUTPUT_FIELDS = [
    'Method',
    'Test Input',
    'Training Supervision',
    'Top-1 Acc',
    'Top-5 Acc',
    'Mean Class Acc',
    'Delta Top-1 vs SlowFast',
    'Delta Mean Class Acc vs SlowFast',
    'Checkpoint',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float; return None on failure or NA."""
    if value is None or str(value).strip().upper() == 'NA' or str(value).strip() == '':
        return None
    try:
        v = float(value)
        # Heuristic: if value looks like a raw 0-1 fraction, keep it.
        # If it looks like a percentage string (e.g. "64.50%"), strip %.
        return v
    except (TypeError, ValueError):
        cleaned = str(value).strip().rstrip('%')
        try:
            return float(cleaned) / 100.0
        except (TypeError, ValueError):
            return None


def _parse_metric(value: Any) -> Optional[float]:
    """Parse a metric that may be 0-1 float or percentage string."""
    if value is None or str(value).strip().upper() == 'NA' or str(value).strip() == '':
        return None
    s = str(value).strip().rstrip('%')
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    # If original CSV stores as 0-1 float (e.g. 0.6450), keep as-is.
    # If it was already a percentage (>1), convert back to fraction.
    if v > 1.0:
        return v / 100.0
    return v


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return NA
    return f'{value * 100:.2f}%'


def _fmt_delta(value: Optional[float]) -> str:
    if value is None:
        return NA
    sign = '+' if value >= 0 else ''
    return f'{sign}{value * 100:.2f}%'


def _basename_from_workdir(work_dir: str) -> str:
    return os.path.basename(os.path.normpath(work_dir))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    method: str = NA
    test_input: str = 'RGB'
    supervision: str = NA
    top1: Optional[float] = None
    top5: Optional[float] = None
    mean_class: Optional[float] = None
    delta_top1: Optional[float] = None
    delta_mean_class: Optional[float] = None
    checkpoint: str = NA


def _read_summary_csv(path: str) -> List[Dict[str, str]]:
    """Read experiment_summary.csv. Returns [] on error."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    except (OSError, csv.Error) as exc:
        print(f'[WARN] failed to read {path}: {exc}', file=sys.stderr)
        return []


def build_comparison(rows: List[Dict[str, str]]) -> List[ComparisonRow]:
    """Build ordered comparison rows with delta computation."""
    # Index by work_dir basename.
    by_base: Dict[str, Dict[str, str]] = {}
    for row in rows:
        work_dir = row.get('work_dir', '')
        base = _basename_from_workdir(work_dir)
        if base:
            by_base[base] = row

    # Find baseline metrics for delta computation.
    baseline_row = by_base.get('slowfast_r50_bs24_amp_25ep', {})
    baseline_top1 = _parse_metric(baseline_row.get('top1_acc'))
    baseline_mean = _parse_metric(baseline_row.get('mean_class_accuracy'))

    results: List[ComparisonRow] = []
    for base_key in ROW_ORDER:
        meta = METHOD_META.get(base_key, {})
        csv_row = by_base.get(base_key)
        if csv_row is None:
            # Directory was not in the summary; still show it as NA.
            results.append(ComparisonRow(
                method=meta.get('method', base_key),
                test_input=meta.get('test_input', 'RGB'),
                supervision=meta.get('supervision', NA),
            ))
            continue

        top1 = _parse_metric(csv_row.get('top1_acc'))
        top5 = _parse_metric(csv_row.get('top5_acc'))
        mean_class = _parse_metric(csv_row.get('mean_class_accuracy'))

        # Delta vs baseline.
        delta_top1: Optional[float] = None
        delta_mean: Optional[float] = None
        if not meta.get('is_baseline', False):
            if top1 is not None and baseline_top1 is not None:
                delta_top1 = top1 - baseline_top1
            if mean_class is not None and baseline_mean is not None:
                delta_mean = mean_class - baseline_mean

        # Checkpoint: prefer best, fall back to latest.
        ckpt = csv_row.get('best_checkpoint_path', NA)
        if ckpt in (None, '', NA, 'NA'):
            ckpt = csv_row.get('latest_checkpoint_path', NA)
        if ckpt in (None, ''):
            ckpt = NA

        results.append(ComparisonRow(
            method=meta.get('method', base_key),
            test_input=meta.get('test_input', 'RGB'),
            supervision=meta.get('supervision', NA),
            top1=top1,
            top5=top5,
            mean_class=mean_class,
            delta_top1=delta_top1,
            delta_mean_class=delta_mean,
            checkpoint=ckpt,
        ))

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _row_to_dict(row: ComparisonRow) -> Dict[str, str]:
    return {
        'Method': row.method,
        'Test Input': row.test_input,
        'Training Supervision': row.supervision,
        'Top-1 Acc': _fmt_pct(row.top1),
        'Top-5 Acc': _fmt_pct(row.top5),
        'Mean Class Acc': _fmt_pct(row.mean_class),
        'Delta Top-1 vs SlowFast': _fmt_delta(row.delta_top1) if row.delta_top1 is not None else NA,
        'Delta Mean Class Acc vs SlowFast': _fmt_delta(row.delta_mean_class) if row.delta_mean_class is not None else NA,
        'Checkpoint': row.checkpoint,
    }


def write_csv(comparison: List[ComparisonRow], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in comparison:
            writer.writerow(_row_to_dict(row))


def _build_markdown(comparison: List[ComparisonRow]) -> str:
    """Build a paper-friendly markdown table (without Checkpoint column for brevity)."""
    # Paper table: omit Checkpoint column (too long for thesis body).
    paper_fields = [
        'Method', 'Test Input', 'Training Supervision',
        'Top-1 Acc', 'Top-5 Acc', 'Mean Class Acc',
        'Delta Top-1 vs SlowFast', 'Delta Mean Class Acc vs SlowFast',
    ]
    header = '| ' + ' | '.join(paper_fields) + ' |\n'
    sep = '|' + '|'.join(['---'] * len(paper_fields)) + '|\n'
    body_lines = []
    for row in comparison:
        d = _row_to_dict(row)
        cells = [d[f] for f in paper_fields]
        body_lines.append('| ' + ' | '.join(cells) + ' |')

    parts = [
        '# Baseline vs Gaze-Supervised SlowFast — Main Results\n\n',
        'All methods use **RGB-only** input at test time.\n',
        'Gaze supervision is applied **only during training** via an auxiliary KL loss.\n\n',
        header, sep, '\n'.join(body_lines), '\n',
        '\n---\n',
        '\n**Notes:**\n',
        '- Delta columns show improvement over the SlowFast-R50 baseline.\n',
        '- "Random / Center / Shuffle" ablations replace real gaze with synthetic targets.\n',
        '- All models share the same SlowFast-R50 backbone and test protocol (10-clip × 3-crop).\n',
    ]
    return ''.join(parts)


def write_markdown(comparison: List[ComparisonRow], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(_build_markdown(comparison))


def _build_latex(comparison: List[ComparisonRow]) -> str:
    """Build a LaTeX tabular environment ready to paste into a thesis."""
    # Compact columns for LaTeX: Method, Supervision, Top-1, Top-5, Mean, Delta Top-1
    lines = [
        r'% Auto-generated by compare_baseline_gaze.py',
        r'% Paste inside a table environment in your thesis.',
        r'\begin{tabular}{l l c c c c}',
        r'\toprule',
        r'Method & Training Supervision & Top-1 (\%) & Top-5 (\%) & Mean Class (\%) & $\Delta$ Top-1 \\',
        r'\midrule',
    ]
    for row in comparison:
        top1_s = f'{row.top1 * 100:.2f}' if row.top1 is not None else '--'
        top5_s = f'{row.top5 * 100:.2f}' if row.top5 is not None else '--'
        mean_s = f'{row.mean_class * 100:.2f}' if row.mean_class is not None else '--'
        if row.delta_top1 is not None:
            sign = '+' if row.delta_top1 >= 0 else ''
            delta_s = f'{sign}{row.delta_top1 * 100:.2f}'
        else:
            delta_s = '--'
        # Escape underscores for LaTeX.
        method_tex = row.method.replace('_', r'\_')
        supervision_tex = row.supervision.replace('_', r'\_')
        lines.append(
            f'{method_tex} & {supervision_tex} & {top1_s} & {top5_s} & {mean_s} & {delta_s} \\\\'
        )
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        '',
    ])
    return '\n'.join(lines)


def write_latex(comparison: List[ComparisonRow], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(_build_latex(comparison))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate the main paper comparison table: '
                    'SlowFast baseline vs Gaze-SlowFast.')
    parser.add_argument(
        '--input', default=DEFAULT_INPUT,
        help=f'Path to experiment_summary.csv. Default: {DEFAULT_INPUT}')
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Directory for output files. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument(
        '--prefix', default=DEFAULT_PREFIX,
        help=f'Output filename prefix. Default: {DEFAULT_PREFIX}')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow overwriting existing output files.')
    parser.add_argument(
        '--pretty-print', action='store_true',
        help='Print the markdown table to stdout.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Read input.
    if not os.path.isfile(args.input):
        print(f'[WARN] input file not found: {args.input}', file=sys.stderr)
        print('[WARN] experiment_summary.csv may not have been generated yet.',
              file=sys.stderr)
        print('[HINT] run collect_experiment_results.py first.', file=sys.stderr)
        # Still produce an empty table so downstream scripts don't break.
        rows: List[Dict[str, str]] = []
    else:
        rows = _read_summary_csv(args.input)

    comparison = build_comparison(rows)

    # Prepare output paths.
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(f'[ERROR] cannot create output dir: {exc}', file=sys.stderr)
        return 1

    csv_path = os.path.join(output_dir, f'{args.prefix}.csv')
    md_path = os.path.join(output_dir, f'{args.prefix}.md')
    latex_path = os.path.join(output_dir, f'{args.prefix}_latex.txt')

    existing = [p for p in (csv_path, md_path, latex_path) if os.path.exists(p)]
    if existing and not args.overwrite:
        print('[ERROR] refusing to overwrite existing files (pass --overwrite):',
              file=sys.stderr)
        for p in existing:
            print(f'  - {p}', file=sys.stderr)
        return 2

    try:
        write_csv(comparison, csv_path)
        write_markdown(comparison, md_path)
        write_latex(comparison, latex_path)
    except OSError as exc:
        print(f'[ERROR] write failed: {exc}', file=sys.stderr)
        return 3

    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {md_path}')
    print(f'[OK] wrote {latex_path}')

    if args.pretty_print:
        print()
        print(_build_markdown(comparison))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
