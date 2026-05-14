"""Shared utilities for EGTEA gaze parsing, alignment, and visualization."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


TEXT_EXTENSIONS = {'.txt', '.csv', '.tsv'}
GAZE_EXTENSIONS = TEXT_EXTENSIONS | {'.json'}
DEFAULT_FIXATION_VALUES = {'1', 'fix', 'fixation', 'f'}
HEADER_ALIASES = {
    'frame_id': ('frame', 'frameid', 'frame_idx', 'frameindex', 'framenum',
                 'frame_no', 'frame_number', 'framenumber', 'index', 'idx'),
    'timestamp': ('timestamp', 'time', 't', 'ts', 'time_ms', 'time_sec',
                  'time_s', 'ms', 'sec', 'seconds'),
    'x': ('x', 'gaze_x', 'gazex', 'pointx', 'point_x', 'xcoord', 'x_coord',
          'xpix', 'pixelx', 'normx', 'px', 'porx', 'positionx'),
    'y': ('y', 'gaze_y', 'gazey', 'pointy', 'point_y', 'ycoord', 'y_coord',
          'ypix', 'pixely', 'normy', 'py', 'pory', 'positiony'),
    'gaze_type': ('type', 'gaze_type', 'gazetype', 'event', 'event_type',
                  'fixation', 'label', 'status', 'movement', 'eyemovement'),
    'validity': ('valid', 'validity', 'isvalid', 'is_valid', 'validflag'),
    'tracked': ('tracked', 'istracked', 'is_tracked', 'tracking', 'found'),
    'confidence': ('confidence', 'conf', 'score', 'quality', 'prob',
                   'probability'),
}


@dataclass
class GazeFormat:
    """Detected file format summary."""

    path: str
    kind: str
    delimiter: str = 'whitespace'
    has_header: bool = False
    encoding: str = 'utf-8'
    metadata_lines: List[str] = field(default_factory=list)
    header: List[str] = field(default_factory=list)
    preview_rows: List[List[str]] = field(default_factory=list)
    columns: Dict[str, Optional[int | str]] = field(default_factory=dict)
    coordinate_mode: str = 'unknown'
    warnings: List[str] = field(default_factory=list)
    sampled_rows: int = 0
    skipped_rows: int = 0
    x_range: Tuple[Optional[float], Optional[float]] = (None, None)
    y_range: Tuple[Optional[float], Optional[float]] = (None, None)
    gaze_type_counts: Dict[str, int] = field(default_factory=dict)
    validity_counts: Dict[str, int] = field(default_factory=dict)
    source_resolution: Tuple[Optional[int], Optional[int]] = (None, None)
    out_of_source_bounds_ratio: Optional[float] = None


@dataclass
class ParsedGazeData:
    """Parsed gaze records aligned to one source file."""

    path: str
    records: List[dict]
    gaze_format: GazeFormat
    timestamp_unit: str = 'unknown'


@dataclass
class GazeFileIndex:
    """Fast lookup structure for raw gaze files."""

    all_files: List[str]
    key_to_files: Dict[str, List[str]]
    lower_path_to_file: Dict[str, str]


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', value.lower())


def normalized_header_aliases() -> Dict[str, set[str]]:
    alias_map: Dict[str, set[str]] = {}
    for field, aliases in HEADER_ALIASES.items():
        alias_map[field] = {normalize_token(alias) for alias in aliases}
        alias_map[field].add(normalize_token(field))
    return alias_map


def safe_path_id(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', value).strip('_')


def stable_hash(value: str, length: int = 16) -> str:
    import hashlib
    return hashlib.sha1(value.encode('utf-8')).hexdigest()[:length]


def read_text_lines(path: str,
                    max_lines: Optional[int] = None,
                    encoding_candidates: Sequence[str] = ('utf-8', 'latin-1')
                    ) -> Tuple[List[str], str]:
    last_error = None
    for encoding in encoding_candidates:
        try:
            lines: List[str] = []
            with open(path, 'r', encoding=encoding, errors='strict') as handle:
                for idx, line in enumerate(handle):
                    if max_lines is not None and idx >= max_lines:
                        break
                    line = line.strip()
                    if line:
                        lines.append(line)
            return lines, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Failed to read text file: {path}')


def extract_text_table(lines: Sequence[str]) -> Tuple[List[str], Optional[List[str]], List[List[str]], List[str]]:
    metadata_lines: List[str] = []
    content_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('##'):
            metadata_lines.append(stripped)
            continue
        content_lines.append(stripped)

    if not content_lines:
        return metadata_lines, None, [], []

    delimiter = detect_delimiter(content_lines)
    first_row = split_text_line(content_lines[0], delimiter)
    has_header = looks_like_header(first_row)
    header = first_row if has_header else None
    row_lines = content_lines[1:] if has_header else content_lines
    data_rows = [split_text_line(line, delimiter) for line in row_lines]
    return metadata_lines, header, data_rows, content_lines


def parse_source_resolution(metadata_lines: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    priority_keys = ('calibration area', 'stimulus dimension')
    for key in priority_keys:
        for line in metadata_lines:
            lowered = line.lower()
            if key not in lowered:
                continue
            numbers = [maybe_float(item) for item in re.findall(r'[-+]?\d+(?:\.\d+)?', line)]
            numbers = [item for item in numbers if item is not None]
            if len(numbers) >= 2:
                width = int(round(numbers[0]))
                height = int(round(numbers[1]))
                if width > 0 and height > 0:
                    return width, height
    return None, None


def compute_out_of_source_bounds_ratio(x_values: Sequence[float],
                                       y_values: Sequence[float],
                                       source_resolution: Tuple[Optional[int], Optional[int]],
                                       coordinate_mode: str) -> Optional[float]:
    source_width, source_height = source_resolution
    if coordinate_mode != 'pixel' or not x_values or not y_values:
        return None
    if not source_width or not source_height:
        return None
    xs = np.asarray(x_values, dtype=np.float32)
    ys = np.asarray(y_values, dtype=np.float32)
    in_bounds = (
        (xs >= 0.0) & (xs <= float(source_width)) &
        (ys >= 0.0) & (ys <= float(source_height)))
    return float(np.mean(~in_bounds))


def resolve_source_resolution(gaze_format: GazeFormat,
                              override_width: Optional[float] = None,
                              override_height: Optional[float] = None,
                              fallback_resolution: Optional[Tuple[int, int]] = None
                              ) -> Tuple[Tuple[Optional[int], Optional[int]], str]:
    if override_width and override_height and override_width > 0 and override_height > 0:
        return (int(round(override_width)), int(round(override_height))), 'user_override'
    source_width, source_height = gaze_format.source_resolution
    if source_width and source_height:
        return (int(source_width), int(source_height)), 'metadata'
    if fallback_resolution and fallback_resolution[0] > 0 and fallback_resolution[1] > 0:
        return (int(fallback_resolution[0]), int(fallback_resolution[1])), 'fallback'
    return (None, None), 'none'


def iter_text_rows(path: str,
                   delimiter: str,
                   has_header: bool,
                   max_rows: Optional[int] = None
                   ) -> Iterator[List[str]]:
    lines, _ = read_text_lines(path, max_lines=max_rows + 1 if max_rows else None)
    if has_header and lines:
        lines = lines[1:]
    count = 0
    for line in lines:
        if max_rows is not None and count >= max_rows:
            break
        row = split_text_line(line, delimiter)
        if row:
            yield row
            count += 1


def split_text_line(line: str, delimiter: str) -> List[str]:
    if delimiter == 'comma':
        return next(csv.reader([line], delimiter=','))
    if delimiter == 'tab':
        return next(csv.reader([line], delimiter='\t'))
    if delimiter == 'semicolon':
        return next(csv.reader([line], delimiter=';'))
    return re.split(r'[\s,]+', line.strip())


def detect_delimiter(lines: Sequence[str]) -> str:
    comma = sum(',' in line for line in lines[:20])
    tab = sum('\t' in line for line in lines[:20])
    semicolon = sum(';' in line for line in lines[:20])
    if comma > tab and comma >= semicolon and comma > 0:
        return 'comma'
    if tab > 0 and tab >= comma and tab >= semicolon:
        return 'tab'
    if semicolon > 0 and semicolon >= comma and semicolon >= tab:
        return 'semicolon'
    return 'whitespace'


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace('\ufeff', '')
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def maybe_int(value: object) -> Optional[int]:
    parsed = maybe_float(value)
    if parsed is None:
        return None
    if abs(parsed - round(parsed)) > 1e-4:
        return None
    return int(round(parsed))


def maybe_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {'1', '1.0', 'true', 't', 'yes', 'y', 'valid', 'tracked'}:
        return True
    if text in {'0', '0.0', 'false', 'f', 'no', 'n', 'invalid', 'untracked'}:
        return False
    parsed = maybe_float(text)
    if parsed is None:
        return None
    return parsed > 0


def canonicalize_gaze_type(value: object) -> str:
    if value is None:
        return 'unknown'
    text = str(value).strip().lower()
    if not text:
        return 'unknown'
    aliases = {
        '1.0': '1',
        '0.0': '0',
        '1': '1',
        '0': '0',
        'fix': 'fixation',
        'fixations': 'fixation',
        'sac': 'saccade',
        'saccades': 'saccade',
    }
    text = aliases.get(text, text)
    if 'fixation' in text:
        return 'fixation'
    if 'saccade' in text:
        return 'saccade'
    if text in {'1', '0', 'unknown'}:
        return text
    return 'unknown'


def is_fixation_type(value: object,
                     fixation_values: Optional[Sequence[str]] = None) -> bool:
    fixation_pool = set(DEFAULT_FIXATION_VALUES)
    if fixation_values:
        fixation_pool |= {str(item).strip().lower() for item in fixation_values}
    return canonicalize_gaze_type(value) in fixation_pool


def looks_like_header(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    alias_map = normalized_header_aliases()
    numeric_count = sum(maybe_float(token) is not None for token in tokens)
    alpha_hits = 0
    for token in tokens:
        norm = normalize_token(token)
        if any(norm in aliases for aliases in alias_map.values()):
            alpha_hits += 2
        elif re.search(r'[a-zA-Z]', token):
            alpha_hits += 1
    return alpha_hits >= max(1, len(tokens) // 2) and numeric_count < len(tokens)


def _column_numeric_stats(rows: Sequence[Sequence[str]], idx: int) -> dict:
    values = [maybe_float(row[idx]) for row in rows if idx < len(row)]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return dict(ratio=0.0, min=None, max=None, mean=None, monotonic=False,
                    integer_like=False, std=None, near_one_ratio=0.0,
                    unique_ratio=0.0, unique_count=0)
    monotonic = all(
        numeric[pos] <= numeric[pos + 1] for pos in range(len(numeric) - 1))
    integer_like = all(abs(value - round(value)) < 1e-4 for value in numeric)
    unique_count = len({round(float(value), 6) for value in numeric})
    return dict(
        ratio=len(numeric) / max(1, len(values)),
        min=min(numeric),
        max=max(numeric),
        mean=float(np.mean(numeric)),
        monotonic=monotonic,
        integer_like=integer_like,
        std=float(np.std(numeric)),
        near_one_ratio=float(np.mean(np.isclose(numeric, 1.0, atol=1e-6))),
        unique_ratio=unique_count / max(1, len(numeric)),
        unique_count=unique_count,
    )


def _header_match_score(normalized_name: str, field: str) -> float:
    aliases = normalized_header_aliases()[field]
    if field == 'frame_id':
        if normalized_name == 'frame':
            return 20.0
    if field == 'timestamp':
        if normalized_name == 'time':
            return 20.0
    if field == 'x':
        if normalized_name == 'lporxpx':
            return 30.0
        if 'porx' in normalized_name or 'gazex' in normalized_name:
            return 20.0
        if normalized_name.endswith('xpx'):
            return 18.0
    if field == 'y':
        if normalized_name == 'lporypx':
            return 30.0
        if 'pory' in normalized_name or 'gazey' in normalized_name:
            return 20.0
        if normalized_name.endswith('ypx'):
            return 18.0
    if field == 'gaze_type':
        if normalized_name == 'leventinfo':
            return 30.0
        if 'eventinfo' in normalized_name:
            return 24.0
        if normalized_name in {'gazetype', 'eventtype'}:
            return 18.0
        if normalized_name == 'type':
            return 0.0
    if normalized_name in aliases:
        return 10.0
    if field == 'frame_id' and (
            normalized_name.startswith('frame') or normalized_name.endswith('frame')):
        return 8.0
    if field == 'timestamp' and 'time' in normalized_name:
        return 8.0
    if field == 'x':
        if normalized_name.endswith('x') and any(
                token in normalized_name for token in ('gaze', 'point', 'coord',
                                                       'pixel', 'norm', 'por', 'pos')):
            return 8.0
        if normalized_name in {'xpos', 'xposition'}:
            return 8.0
    if field == 'y':
        if normalized_name.endswith('y') and any(
                token in normalized_name for token in ('gaze', 'point', 'coord',
                                                       'pixel', 'norm', 'por', 'pos')):
            return 8.0
        if normalized_name in {'ypos', 'yposition'}:
            return 8.0
    if field in {'gaze_type', 'validity', 'tracked', 'confidence'}:
        token_map = {
            'gaze_type': ('event', 'fix', 'label', 'status', 'movement'),
            'validity': ('valid',),
            'tracked': ('track', 'found'),
            'confidence': ('conf', 'score', 'quality', 'prob'),
        }
        if any(token in normalized_name for token in token_map[field]):
            return 8.0
    return 0.0


def _textual_values_score(rows: Sequence[Sequence[str]], idx: int, field: str) -> float:
    values = [str(row[idx]).strip() for row in rows if idx < len(row) and str(row[idx]).strip()]
    if not values:
        return 0.0
    normalized = [canonicalize_gaze_type(value) for value in values]
    if field == 'gaze_type':
        event_ratio = float(np.mean([value in {'fixation', 'saccade'} for value in normalized]))
        smp_ratio = float(np.mean([value == 'smp' for value in normalized]))
        return event_ratio * 10.0 - smp_ratio * 10.0
    return 0.0


def _is_suspicious_coordinate_column(stat: dict) -> bool:
    if stat['ratio'] < 0.5:
        return True
    if stat['near_one_ratio'] >= 0.95:
        return True
    if stat['std'] is not None and stat['std'] <= 1e-6:
        return True
    if stat['unique_count'] <= 1:
        return True
    return False


def infer_columns(header: Optional[Sequence[str]],
                  rows: Sequence[Sequence[str]]) -> Dict[str, Optional[int]]:
    columns: Dict[str, Optional[int]] = {
        'frame_id': None,
        'timestamp': None,
        'x': None,
        'y': None,
        'gaze_type': None,
        'validity': None,
        'tracked': None,
        'confidence': None,
    }
    if not rows:
        return columns
    max_cols = max(len(row) for row in rows)
    stats = [_column_numeric_stats(rows, idx) for idx in range(max_cols)]

    if header:
        normalized_header = [normalize_token(item) for item in header]
        for field in columns:
            scored = []
            for idx, name in enumerate(normalized_header):
                score = _header_match_score(name, field)
                if field == 'gaze_type':
                    score += _textual_values_score(rows, idx, field)
                if score > 0:
                    scored.append((score, idx))
            if scored:
                scored.sort(reverse=True)
                columns[field] = scored[0][1]

    for field in ('x', 'y'):
        idx = columns[field]
        if idx is not None:
            stat = _column_numeric_stats(rows, idx)
            if _is_suspicious_coordinate_column(stat):
                columns[field] = None

    monotonic_candidates = [
        idx for idx, stat in enumerate(stats)
        if stat['ratio'] >= 0.8 and stat['monotonic']
    ]
    if columns['frame_id'] is None:
        for idx in monotonic_candidates:
            if stats[idx]['integer_like']:
                columns['frame_id'] = idx
                break
    if columns['timestamp'] is None and not header:
        for idx in monotonic_candidates:
            if idx != columns['frame_id']:
                columns['timestamp'] = idx
                break

    coord_scores = []
    blocked_coordinate_columns = {
        idx for key, idx in columns.items()
        if key in {'gaze_type', 'validity', 'tracked', 'confidence'} and idx is not None
    }
    for idx, stat in enumerate(stats):
        if stat['ratio'] < 0.8:
            continue
        if idx in (columns['frame_id'], columns['timestamp']) or idx in blocked_coordinate_columns:
            continue
        score = 0.0
        min_v, max_v = stat['min'], stat['max']
        if min_v is None or max_v is None:
            continue
        if _is_suspicious_coordinate_column(stat):
            score -= 10.0
        if -0.25 <= min_v <= 1.25 and -0.25 <= max_v <= 1.25:
            score += 4.0
        if -5 <= min_v <= 8192 and 0 <= max_v <= 8192:
            score += 1.5
        if not stat['monotonic']:
            score += 0.5
        if stat['unique_ratio'] > 0.05:
            score += 1.0
        coord_scores.append((score, idx))
    coord_scores.sort(reverse=True)

    if columns['x'] is None:
        for score, idx in coord_scores:
            if score > 0:
                columns['x'] = idx
                break
    if columns['y'] is None:
        for score, idx in coord_scores:
            if score <= 0:
                continue
            if idx != columns['x']:
                columns['y'] = idx
                break
    if columns['y'] is None and columns['x'] is not None:
        fallback = columns['x'] + 1
        if fallback < max_cols and fallback not in blocked_coordinate_columns:
            columns['y'] = fallback

    return columns


def detect_coordinate_mode(x_values: Sequence[float],
                           y_values: Sequence[float]) -> str:
    if not x_values or not y_values:
        return 'unknown'
    xs = np.asarray(x_values, dtype=np.float32)
    ys = np.asarray(y_values, dtype=np.float32)
    norm_ratio = float(np.mean(
        (xs >= -0.05) & (xs <= 1.05) & (ys >= -0.05) & (ys <= 1.05)))
    pixel_ratio = float(np.mean(
        (xs >= -10) & (xs <= 4096) & (ys >= -10) & (ys <= 4096)))
    if norm_ratio >= 0.8:
        return 'normalized'
    if pixel_ratio >= 0.8:
        return 'pixel'
    return 'unknown'


def record_is_explicitly_valid(record: dict) -> bool:
    validity = record.get('validity')
    tracked = record.get('tracked')
    confidence = record.get('confidence')
    if validity is not None and not validity:
        return False
    if tracked is not None and not tracked:
        return False
    if confidence is not None and confidence <= 0.0:
        return False
    return True


def detect_timestamp_unit(timestamps: Sequence[float]) -> str:
    if len(timestamps) < 2:
        return 'unknown'
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 'unknown'
    median_diff = float(np.median(diffs))
    if median_diff > 1.0:
        return 'ms'
    return 'sec'


def parse_text_gaze_file(path: str,
                         max_records: Optional[int] = None) -> ParsedGazeData:
    lines, encoding = read_text_lines(path, max_lines=max_records)
    if not lines:
        fmt = GazeFormat(path=path, kind='text', warnings=['empty file'])
        return ParsedGazeData(path=path, records=[], gaze_format=fmt)
    metadata_lines, header, data_rows, content_lines = extract_text_table(lines)
    delimiter = detect_delimiter(content_lines)
    has_header = header is not None
    columns = infer_columns(header, data_rows)
    source_resolution = parse_source_resolution(metadata_lines)

    records: List[dict] = []
    skipped = 0
    x_values: List[float] = []
    y_values: List[float] = []
    gaze_type_counts: Counter[str] = Counter()
    validity_counts: Counter[str] = Counter()
    timestamps: List[float] = []
    warnings_list: List[str] = []

    for row in data_rows:
        try:
            x = maybe_float(row[columns['x']]) if columns['x'] is not None and columns['x'] < len(row) else None
            y = maybe_float(row[columns['y']]) if columns['y'] is not None and columns['y'] < len(row) else None
            if x is None or y is None:
                skipped += 1
                continue
            frame_id = maybe_int(row[columns['frame_id']]) \
                if columns['frame_id'] is not None and columns['frame_id'] < len(row) else None
            timestamp = maybe_float(row[columns['timestamp']]) \
                if columns['timestamp'] is not None and columns['timestamp'] < len(row) else None
            gaze_type = canonicalize_gaze_type(
                row[columns['gaze_type']]) if columns['gaze_type'] is not None and columns['gaze_type'] < len(row) else 'unknown'
            validity = maybe_bool(
                row[columns['validity']]) if columns['validity'] is not None and columns['validity'] < len(row) else None
            tracked = maybe_bool(
                row[columns['tracked']]) if columns['tracked'] is not None and columns['tracked'] < len(row) else None
            confidence = maybe_float(
                row[columns['confidence']]) if columns['confidence'] is not None and columns['confidence'] < len(row) else None
            record = dict(
                frame_id=frame_id,
                timestamp=timestamp,
                x=x,
                y=y,
                gaze_type=gaze_type,
                validity=validity,
                tracked=tracked,
                confidence=confidence,
            )
            records.append(record)
            x_values.append(x)
            y_values.append(y)
            gaze_type_counts[gaze_type] += 1
            validity_counts['valid' if record_is_explicitly_valid(record) else 'invalid'] += 1
            if timestamp is not None:
                timestamps.append(timestamp)
        except Exception:
            skipped += 1

    x_stats = _column_numeric_stats(data_rows, columns['x']) if columns['x'] is not None else {}
    y_stats = _column_numeric_stats(data_rows, columns['y']) if columns['y'] is not None else {}
    if columns['x'] is None or columns['y'] is None:
        warnings_list.append('x_or_y_column_missing')
    if x_stats and _is_suspicious_coordinate_column(x_stats):
        warnings_list.append('x_column_suspicious')
    if y_stats and _is_suspicious_coordinate_column(y_stats):
        warnings_list.append('y_column_suspicious')
    if x_stats.get('near_one_ratio', 0.0) >= 0.95:
        warnings_list.append('x_column_nearly_constant_one')
    if y_stats.get('near_one_ratio', 0.0) >= 0.95:
        warnings_list.append('y_column_nearly_constant_one')
    if columns['gaze_type'] is None:
        warnings_list.append('gaze_type_column_missing')
    if columns['validity'] is None and columns['tracked'] is None and columns['confidence'] is None:
        warnings_list.append('no_validity_confidence_columns')
    if metadata_lines:
        warnings_list.append('metadata_lines_skipped')
    out_of_source_bounds_ratio = compute_out_of_source_bounds_ratio(
        x_values, y_values, source_resolution, detect_coordinate_mode(x_values, y_values))

    fmt = GazeFormat(
        path=path,
        kind='text',
        delimiter=delimiter,
        has_header=has_header,
        encoding=encoding,
        metadata_lines=metadata_lines[:20],
        header=list(header) if header else [],
        preview_rows=[list(row) for row in data_rows[:5]],
        columns=columns,
        coordinate_mode=detect_coordinate_mode(x_values, y_values),
        warnings=warnings_list,
        sampled_rows=len(records),
        skipped_rows=skipped,
        x_range=(min(x_values), max(x_values)) if x_values else (None, None),
        y_range=(min(y_values), max(y_values)) if y_values else (None, None),
        gaze_type_counts=dict(gaze_type_counts),
        validity_counts=dict(validity_counts),
        source_resolution=source_resolution,
        out_of_source_bounds_ratio=out_of_source_bounds_ratio,
    )
    return ParsedGazeData(
        path=path,
        records=records,
        gaze_format=fmt,
        timestamp_unit=detect_timestamp_unit(timestamps),
    )


def parse_json_gaze_file(path: str,
                         max_records: Optional[int] = None) -> ParsedGazeData:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ('data', 'records', 'gaze', 'samples'):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        payload = []

    sampled = payload[:max_records] if max_records else payload
    normalized_keys = None
    if sampled and isinstance(sampled[0], dict):
        normalized_keys = {normalize_token(key): key for key in sampled[0].keys()}
    frame_key = None
    time_key = None
    x_key = None
    y_key = None
    gaze_type_key = None
    validity_key = None
    tracked_key = None
    confidence_key = None
    if normalized_keys:
        alias_map = normalized_header_aliases()
        for alias in alias_map['frame_id']:
            if alias in normalized_keys:
                frame_key = normalized_keys[alias]
                break
        for alias in alias_map['timestamp']:
            if alias in normalized_keys:
                time_key = normalized_keys[alias]
                break
        for alias in alias_map['x']:
            if alias in normalized_keys:
                x_key = normalized_keys[alias]
                break
        for alias in alias_map['y']:
            if alias in normalized_keys:
                y_key = normalized_keys[alias]
                break
        for alias in alias_map['gaze_type']:
            if alias in normalized_keys:
                gaze_type_key = normalized_keys[alias]
                break
        for alias in alias_map['validity']:
            if alias in normalized_keys:
                validity_key = normalized_keys[alias]
                break
        for alias in alias_map['tracked']:
            if alias in normalized_keys:
                tracked_key = normalized_keys[alias]
                break
        for alias in alias_map['confidence']:
            if alias in normalized_keys:
                confidence_key = normalized_keys[alias]
                break

    records: List[dict] = []
    skipped = 0
    x_values: List[float] = []
    y_values: List[float] = []
    gaze_type_counts: Counter[str] = Counter()
    validity_counts: Counter[str] = Counter()
    timestamps: List[float] = []
    for item in sampled:
        if not isinstance(item, dict):
            skipped += 1
            continue
        x = maybe_float(item.get(x_key)) if x_key else None
        y = maybe_float(item.get(y_key)) if y_key else None
        if x is None or y is None:
            skipped += 1
            continue
        frame_id = maybe_int(item.get(frame_key)) if frame_key else None
        timestamp = maybe_float(item.get(time_key)) if time_key else None
        gaze_type = canonicalize_gaze_type(item.get(gaze_type_key)) \
            if gaze_type_key else 'unknown'
        validity = maybe_bool(item.get(validity_key)) if validity_key else None
        tracked = maybe_bool(item.get(tracked_key)) if tracked_key else None
        confidence = maybe_float(item.get(confidence_key)) if confidence_key else None
        records.append(
            dict(frame_id=frame_id, timestamp=timestamp, x=x, y=y,
                 gaze_type=gaze_type, validity=validity, tracked=tracked,
                 confidence=confidence))
        x_values.append(x)
        y_values.append(y)
        gaze_type_counts[gaze_type] += 1
        validity_counts['valid' if record_is_explicitly_valid(records[-1]) else 'invalid'] += 1
        if timestamp is not None:
            timestamps.append(timestamp)

    fmt = GazeFormat(
        path=path,
        kind='json',
        delimiter='json',
        has_header=True,
        columns=dict(
            frame_id=frame_key,
            timestamp=time_key,
            x=x_key,
            y=y_key,
            gaze_type=gaze_type_key,
            validity=validity_key,
            tracked=tracked_key,
            confidence=confidence_key,
        ),
        coordinate_mode=detect_coordinate_mode(x_values, y_values),
        sampled_rows=len(records),
        skipped_rows=skipped,
        x_range=(min(x_values), max(x_values)) if x_values else (None, None),
        y_range=(min(y_values), max(y_values)) if y_values else (None, None),
        gaze_type_counts=dict(gaze_type_counts),
        validity_counts=dict(validity_counts),
        source_resolution=(None, None),
        out_of_source_bounds_ratio=None,
    )
    return ParsedGazeData(
        path=path,
        records=records,
        gaze_format=fmt,
        timestamp_unit=detect_timestamp_unit(timestamps),
    )


def _parse_gaze_file_impl(path: str,
                          max_records: Optional[int] = None) -> ParsedGazeData:
    ext = Path(path).suffix.lower()
    if ext == '.json':
        return parse_json_gaze_file(path, max_records=max_records)
    return parse_text_gaze_file(path, max_records=max_records)


@lru_cache(maxsize=512)
def parse_gaze_file_sample(path: str,
                           max_rows: int = 2000) -> ParsedGazeData:
    return _parse_gaze_file_impl(path, max_records=max_rows)


@lru_cache(maxsize=512)
def parse_gaze_file_full(path: str) -> ParsedGazeData:
    return _parse_gaze_file_impl(path, max_records=None)


def parse_gaze_file(path: str,
                    max_records: Optional[int] = None) -> ParsedGazeData:
    if max_records is None:
        return parse_gaze_file_full(path)
    return parse_gaze_file_sample(path, max_rows=int(max_records))


def find_gaze_files(gaze_root: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(gaze_root):
        for filename in filenames:
            if Path(filename).suffix.lower() in GAZE_EXTENSIONS:
                files.append(os.path.join(root, filename))
    files.sort()
    return files


def build_gaze_file_index(gaze_root: str) -> GazeFileIndex:
    files = find_gaze_files(gaze_root)
    key_to_files: Dict[str, List[str]] = defaultdict(list)
    lower_path_to_file: Dict[str, str] = {}
    for path in files:
        lower_path_to_file[path.lower()] = path
        rel_parts = Path(path).parts
        key_candidates = set()
        stem = normalize_token(Path(path).stem)
        if stem:
            key_candidates.add(stem)
        for part in rel_parts[-4:]:
            norm = normalize_token(Path(part).stem if '.' in part else part)
            if norm:
                key_candidates.add(norm)
        for key in key_candidates:
            key_to_files[key].append(path)
    return GazeFileIndex(files, dict(key_to_files), lower_path_to_file)


def _video_candidate_keys(video_relpath: str) -> List[str]:
    path = Path(video_relpath)
    stem = path.stem
    parent = path.parent.name
    keys = [stem, parent, f'{parent}{stem}']
    clip_prefix = re.sub(r'-F\d+.*$', '', stem)
    if clip_prefix != stem:
        keys.append(clip_prefix)
    keys.extend(re.findall(r'(?:OP|P)\d+-R\d+-[A-Za-z]+', stem))
    return [normalize_token(item) for item in keys if item]


def match_gaze_file(video_relpath: str,
                    gaze_index: GazeFileIndex) -> Tuple[Optional[str], List[str]]:
    warnings_list: List[str] = []
    candidates = _video_candidate_keys(video_relpath)
    scored: Counter[str] = Counter()
    for key in candidates:
        for path in gaze_index.key_to_files.get(key, []):
            scored[path] += 5
        for path in gaze_index.all_files:
            normalized_path = normalize_token(path)
            if key and key in normalized_path:
                scored[path] += 1
    if not scored:
        warnings_list.append(f'No gaze file matched: {video_relpath}')
        return None, warnings_list
    best_path, _ = scored.most_common(1)[0]
    if len(scored) > 1 and scored.most_common(2)[0][1] == scored.most_common(2)[1][1]:
        warnings_list.append(
            f'Ambiguous gaze match for {video_relpath}; picked {best_path}')
    return best_path, warnings_list


def resolve_video_path(video_root: str, video_relpath: str) -> str:
    root = Path(video_root)
    rel = Path(video_relpath)
    candidates = [
        root / rel,
        root.parent / rel,
    ]
    if root.name == 'cropped_clips' and rel.parts and rel.parts[0] == 'cropped_clips':
        candidates.insert(0, root.parent / rel)
        candidates.insert(1, root / Path(*rel.parts[1:]))
    else:
        candidates.append(root / 'cropped_clips' / rel)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def parse_clip_start_frame(video_path: str) -> Optional[int]:
    frame_info = parse_clip_frame_info(video_path)
    if frame_info is None:
        return None
    return frame_info['start_frame']


def parse_clip_frame_info(video_path: str) -> Optional[dict]:
    candidates = [Path(video_path).stem, Path(video_path).parent.name]
    prioritized_patterns = [
        (re.compile(r'[Ff]0*(\d+)\s*[-_]\s*[Ff]0*(\d+)'), 'F_range'),
        (re.compile(r'[Ff]0*(\d+)[-_]?[Ff]0*(\d+)'), 'F_range'),
        (re.compile(r'Frame0*(\d+)\s*[-_]\s*Frame0*(\d+)', re.IGNORECASE), 'Frame_range'),
    ]
    fallback_patterns = [
        (re.compile(r'0*(\d{3,})\s*[-_]\s*0*(\d{3,})'), 'numeric_range'),
    ]
    for text in candidates:
        for pattern, source in prioritized_patterns:
            match = pattern.search(text)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                if end_frame >= start_frame:
                    return dict(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        parse_source=source,
                        matched_text=match.group(0),
                    )
    for text in candidates:
        for pattern, source in fallback_patterns:
            match = pattern.search(text)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                if end_frame >= start_frame:
                    return dict(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        parse_source=source,
                        matched_text=match.group(0),
                    )
    return None


def parse_clip_frame_range(video_path: str) -> Optional[Tuple[int, int]]:
    frame_info = parse_clip_frame_info(video_path)
    if frame_info is None:
        return None
    return frame_info['start_frame'], frame_info['end_frame']


def get_video_reader(video_path: str):
    try:
        import decord
        return 'decord', decord.VideoReader(video_path)
    except Exception:
        try:
            import cv2
            return 'cv2', cv2.VideoCapture(video_path)
        except Exception as exc:
            raise RuntimeError(f'No video backend available for {video_path}') from exc


def get_video_stats(video_path: str) -> Tuple[int, float]:
    backend, reader = get_video_reader(video_path)
    if backend == 'decord':
        fps = float(reader.get_avg_fps()) if reader.get_avg_fps() else 30.0
        return len(reader), fps
    import cv2
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(reader.get(cv2.CAP_PROP_FPS)) or 30.0
    reader.release()
    return total_frames, fps


def read_video_frame(video_path: str, frame_idx: int):
    backend, reader = get_video_reader(video_path)
    if backend == 'decord':
        frame_idx = int(np.clip(frame_idx, 0, len(reader) - 1))
        return reader[frame_idx].asnumpy()
    import cv2
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(np.clip(frame_idx, 0, max(total_frames - 1, 0)))
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = reader.read()
    reader.release()
    if not ok:
        raise RuntimeError(f'Failed to read frame {frame_idx} from {video_path}')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def choose_record_for_frame(records: Sequence[dict],
                            fixation_only: bool = False,
                            prefer_fixation: bool = False,
                            fixation_values: Optional[Sequence[str]] = None
                            ) -> Tuple[Optional[dict], str]:
    if not records:
        return None, 'no_matching_frame'
    fixation_records = [
        item for item in records
        if is_fixation_type(item.get('gaze_type'), fixation_values)
    ]
    if fixation_only:
        if not fixation_records:
            return None, 'not_fixation'
        return fixation_records[0], 'fixation'
    if prefer_fixation and fixation_records:
        return fixation_records[0], 'fixation_preferred'
    return records[0], canonicalize_gaze_type(records[0].get('gaze_type'))


def pick_record_for_frame(records: Sequence[dict],
                          fixation_only: bool = False,
                          fixation_values: Optional[Sequence[str]] = None
                          ) -> Optional[dict]:
    chosen, _ = choose_record_for_frame(
        records, fixation_only=fixation_only, fixation_values=fixation_values)
    return chosen


def align_gaze_to_clip(parsed_gaze: ParsedGazeData,
                       total_frames: int,
                       fps: float,
                       start_frame: Optional[int] = None,
                       frame_range: Optional[Tuple[int, int]] = None,
                       only_fixation: bool = False,
                       prefer_fixation: bool = False,
                       fixation_values: Optional[Sequence[str]] = None,
                       frame_match_tolerance: Optional[int] = None,
                       nearest_tolerance: float = 0.5) -> dict:
    frame_to_records: Dict[int, List[dict]] = defaultdict(list)
    time_pairs: List[Tuple[float, dict]] = []
    for record in parsed_gaze.records:
        frame_id = record.get('frame_id')
        timestamp = record.get('timestamp')
        if frame_id is not None:
            frame_to_records[int(frame_id)].append(record)
        if timestamp is not None:
            scale = 0.001 if parsed_gaze.timestamp_unit == 'ms' else 1.0
            time_pairs.append((float(timestamp) * scale, record))
    sorted_frame_ids = np.array(sorted(frame_to_records.keys()), dtype=np.int32) \
        if frame_to_records else np.array([], dtype=np.int32)

    gaze_xy = np.zeros((total_frames, 2), dtype=np.float32)
    gaze_valid = np.zeros((total_frames,), dtype=np.uint8)
    gaze_type = np.array(['unknown'] * total_frames, dtype=object)
    global_frame_ids = np.full((total_frames,), -1, dtype=np.int32)
    matched_frame_ids = np.full((total_frames,), -1, dtype=np.int32)
    rows_found = np.zeros((total_frames,), dtype=np.int32)
    invalid_reasons = np.array(['no_matching_frame'] * total_frames, dtype=object)
    warning_codes: List[str] = []
    matched_fixations = 0

    if frame_match_tolerance is None:
        if frame_range is not None and total_frames > 1:
            span = max(frame_range[1] - frame_range[0], 0)
            estimated_step = span / max(total_frames - 1, 1)
            frame_match_tolerance = max(2, int(math.ceil(estimated_step)))
        else:
            frame_match_tolerance = 2

    def estimate_target_frame(clip_idx: int) -> Optional[int]:
        if frame_range is not None:
            start_f, end_f = frame_range
            if total_frames <= 1:
                return int(start_f)
            ratio = clip_idx / max(total_frames - 1, 1)
            return int(round(start_f + ratio * (end_f - start_f)))
        if start_frame is not None:
            return int(start_frame + clip_idx)
        return None

    def pick_nearest_frame_record(target_frame: int) -> Tuple[Optional[dict], Optional[int], int, str]:
        if sorted_frame_ids.size == 0:
            return None, None, 0, 'no_matching_frame'
        pos = int(np.searchsorted(sorted_frame_ids, target_frame))
        candidate_ids: List[int] = []
        if pos < sorted_frame_ids.size:
            candidate_ids.append(int(sorted_frame_ids[pos]))
        if pos > 0:
            candidate_ids.append(int(sorted_frame_ids[pos - 1]))
        if not candidate_ids:
            return None, None, 0, 'no_matching_frame'
        best_frame = min(candidate_ids, key=lambda item: abs(item - target_frame))
        if abs(best_frame - target_frame) > int(frame_match_tolerance):
            return None, None, 0, 'no_matching_frame'
        chosen, reason = choose_record_for_frame(
            frame_to_records.get(best_frame, []),
            fixation_only=only_fixation,
            prefer_fixation=prefer_fixation,
            fixation_values=fixation_values)
        if chosen is None:
            return None, best_frame, len(frame_to_records.get(best_frame, [])), reason
        return chosen, best_frame, len(frame_to_records.get(best_frame, [])), reason

    for clip_idx in range(total_frames):
        local_target = clip_idx
        global_target = estimate_target_frame(clip_idx)
        if global_target is not None:
            global_frame_ids[clip_idx] = int(global_target)
        chosen = None
        chosen_reason = 'no_matching_frame'
        chosen_frame_id = None
        chosen_rows_found = 0
        if frame_to_records:
            if global_target is not None:
                exact_records = frame_to_records.get(global_target, [])
                chosen_rows_found = len(exact_records)
                chosen, chosen_reason = choose_record_for_frame(
                    exact_records,
                    fixation_only=only_fixation,
                    prefer_fixation=prefer_fixation,
                    fixation_values=fixation_values)
                if chosen is not None:
                    chosen_frame_id = int(global_target)
                if chosen is None:
                    chosen, chosen_frame_id, chosen_rows_found, chosen_reason = \
                        pick_nearest_frame_record(global_target)
            if chosen is None and global_target is None:
                local_records = frame_to_records.get(local_target, [])
                chosen_rows_found = len(local_records)
                chosen, chosen_reason = choose_record_for_frame(
                    local_records,
                    fixation_only=only_fixation,
                    prefer_fixation=prefer_fixation,
                    fixation_values=fixation_values)
                if chosen is not None:
                    chosen_frame_id = int(local_target)
                if chosen is None:
                    chosen, chosen_frame_id, chosen_rows_found, chosen_reason = \
                        pick_nearest_frame_record(local_target)
        if chosen is None and time_pairs:
            target_time = clip_idx / max(fps, 1e-6)
            if global_target is not None:
                target_time = global_target / max(fps, 1e-6)
            best_delta = None
            best_record = None
            for timestamp, record in time_pairs:
                delta = abs(timestamp - target_time)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_record = record
            if best_record is not None and (best_delta is None or best_delta <= nearest_tolerance):
                candidate, chosen_reason = choose_record_for_frame(
                    [best_record],
                    fixation_only=only_fixation,
                    prefer_fixation=prefer_fixation,
                    fixation_values=fixation_values)
                chosen = candidate
                chosen_rows_found = 1
                chosen_frame_id = int(best_record.get('frame_id')) \
                    if best_record.get('frame_id') is not None else -1

        if chosen is None:
            rows_found[clip_idx] = int(chosen_rows_found)
            invalid_reasons[clip_idx] = chosen_reason
            continue
        gaze_xy[clip_idx, 0] = float(chosen['x'])
        gaze_xy[clip_idx, 1] = float(chosen['y'])
        gaze_type[clip_idx] = canonicalize_gaze_type(chosen.get('gaze_type'))
        matched_frame_ids[clip_idx] = int(chosen_frame_id) if chosen_frame_id is not None else -1
        rows_found[clip_idx] = int(chosen_rows_found)
        if record_is_explicitly_valid(chosen):
            gaze_valid[clip_idx] = 1
            invalid_reasons[clip_idx] = 'matched'
        else:
            invalid_reasons[clip_idx] = 'explicit_invalid'
        if gaze_valid[clip_idx] and is_fixation_type(chosen.get('gaze_type'), fixation_values):
            matched_fixations += 1

    if start_frame is None and frame_range is None:
        warning_codes.append('missing_clip_start_frame')
    if matched_fixations == 0 and only_fixation:
        warning_codes.append('no_fixation_match')
    return dict(
        gaze_xy=gaze_xy,
        gaze_valid=gaze_valid,
        gaze_type=gaze_type,
        global_frame_ids=global_frame_ids,
        matched_frame_ids=matched_frame_ids,
        rows_found=rows_found,
        invalid_reasons=invalid_reasons,
        frame_indices=np.arange(total_frames, dtype=np.int32),
        warning_codes=warning_codes,
        matched_fixations=matched_fixations,
    )


def normalize_xy_array(gaze_xy: np.ndarray,
                       coordinate_mode: str,
                       image_size: Tuple[int, int],
                       source_resolution: Optional[Tuple[Optional[int], Optional[int]]] = None,
                       clip: bool = False,
                       return_info: bool = False):
    output = np.asarray(gaze_xy, dtype=np.float32).copy()
    width, height = image_size
    source_width = None
    source_height = None
    if source_resolution is not None:
        source_width, source_height = source_resolution
    source_in_bounds = np.ones((output.shape[0],), dtype=bool)
    coordinate_scale_mode = 'identity'
    if coordinate_mode == 'pixel':
        if source_width and source_height and source_width > 0 and source_height > 0:
            coordinate_scale_mode = 'source_resolution'
            source_in_bounds = (
                (output[:, 0] >= 0.0) & (output[:, 0] <= float(source_width)) &
                (output[:, 1] >= 0.0) & (output[:, 1] <= float(source_height)))
            output[:, 0] /= float(source_width)
            output[:, 1] /= float(source_height)
        elif width > 0 and height > 0:
            coordinate_scale_mode = 'frame_size_fallback'
            source_in_bounds = (
                (output[:, 0] >= 0.0) & (output[:, 0] <= float(width)) &
                (output[:, 1] >= 0.0) & (output[:, 1] <= float(height)))
            output[:, 0] /= float(width)
            output[:, 1] /= float(height)
        else:
            raise ValueError('image_size or source_resolution must be positive for pixel coordinates')
    elif coordinate_mode == 'unknown':
        finite = output[np.isfinite(output)]
        if finite.size and float(np.nanmax(np.abs(finite))) > 1.5:
            if source_width and source_height and source_width > 0 and source_height > 0:
                coordinate_scale_mode = 'source_resolution_unknown_mode'
                source_in_bounds = (
                    (output[:, 0] >= 0.0) & (output[:, 0] <= float(source_width)) &
                    (output[:, 1] >= 0.0) & (output[:, 1] <= float(source_height)))
                output[:, 0] /= float(source_width)
                output[:, 1] /= float(source_height)
            elif width > 0 and height > 0:
                coordinate_scale_mode = 'frame_size_unknown_mode'
                source_in_bounds = (
                    (output[:, 0] >= 0.0) & (output[:, 0] <= float(width)) &
                    (output[:, 1] >= 0.0) & (output[:, 1] <= float(height)))
                output[:, 0] /= float(width)
                output[:, 1] /= float(height)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if clip:
        output = np.clip(output, 0.0, 1.0)
    if return_info:
        return output, source_in_bounds.astype(bool), coordinate_scale_mode
    return output


def apply_crop_and_flip_to_xy(gaze_xy: np.ndarray,
                              gaze_valid: np.ndarray,
                              crop_quadruple: Optional[Sequence[float]] = None,
                              flip: bool = False,
                              flip_direction: str = 'horizontal') -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(gaze_xy, dtype=np.float32).copy()
    valid = np.asarray(gaze_valid, dtype=np.uint8).copy()

    if crop_quadruple is not None:
        quad = np.asarray(crop_quadruple, dtype=np.float32).reshape(-1)
        if quad.size == 4:
            crop_x, crop_y, crop_w, crop_h = quad.tolist()
            crop_w = max(float(crop_w), 1e-6)
            crop_h = max(float(crop_h), 1e-6)
            xy[:, 0] = (xy[:, 0] - crop_x) / crop_w
            xy[:, 1] = (xy[:, 1] - crop_y) / crop_h

    if flip and flip_direction == 'horizontal':
        xy[:, 0] = 1.0 - xy[:, 0]
    elif flip and flip_direction == 'vertical':
        xy[:, 1] = 1.0 - xy[:, 1]

    in_range = (
        (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) &
        (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0))
    valid = valid * in_range.astype(np.uint8)
    xy = np.clip(xy, 0.0, 1.0)
    return xy, valid


def gaussian_heatmap_from_xy(x: float,
                             y: float,
                             size: Tuple[int, int],
                             sigma: float) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32))
    px = float(x) * max(width - 1, 1)
    py = float(y) * max(height - 1, 1)
    heatmap = np.exp(-((grid_x - px)**2 + (grid_y - py)**2) /
                     (2.0 * sigma * sigma))
    heatmap_sum = float(heatmap.sum())
    if heatmap_sum <= 0:
        return np.zeros((height, width), dtype=np.float32)
    return (heatmap / heatmap_sum).astype(np.float32)


def gaze_xy_to_heatmaps(gaze_xy: np.ndarray,
                        gaze_valid: np.ndarray,
                        size: Tuple[int, int],
                        sigma: float) -> np.ndarray:
    xy = np.asarray(gaze_xy, dtype=np.float32)
    valid = np.asarray(gaze_valid, dtype=np.uint8)
    heatmaps = np.zeros((xy.shape[0], int(size[1]), int(size[0])),
                        dtype=np.float32)
    for idx, point in enumerate(xy):
        if idx >= valid.shape[0] or not valid[idx]:
            continue
        heatmaps[idx] = gaussian_heatmap_from_xy(point[0], point[1], size, sigma)
    return heatmaps


def summarize_annotation_file(ann_file: str) -> dict:
    with open(ann_file, 'r', encoding='utf-8') as handle:
        entries = [line.strip().split()[0] for line in handle if line.strip()]
    return dict(count=len(entries), entries=entries)


def compare_annotation_lists(reference_ann: str, candidate_ann: str) -> dict:
    ref_summary = summarize_annotation_file(reference_ann)
    cand_summary = summarize_annotation_file(candidate_ann)
    ref_set = set(ref_summary['entries'])
    cand_set = set(cand_summary['entries'])
    overlap = ref_set & cand_set
    return dict(
        reference=reference_ann,
        candidate=candidate_ann,
        reference_count=ref_summary['count'],
        candidate_count=cand_summary['count'],
        overlap_count=len(overlap),
        overlap_ratio_to_reference=len(overlap) / max(1, len(ref_set)),
        overlap_ratio_to_candidate=len(overlap) / max(1, len(cand_set)),
        only_in_reference=sorted(ref_set - cand_set)[:100],
        only_in_candidate=sorted(cand_set - ref_set)[:100],
    )


def dump_json(data: dict, output: str) -> None:
    ensure_dir(Path(output).parent)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
