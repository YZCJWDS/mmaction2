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
                 'frame_no', 'frame_number', 'index', 'idx'),
    'timestamp': ('timestamp', 'time', 't', 'ts', 'time_ms', 'time_sec',
                  'time_s', 'ms', 'sec', 'seconds'),
    'x': ('x', 'gaze_x', 'gazex', 'pointx', 'point_x', 'xcoord', 'x_coord',
          'xpix', 'pixelx', 'normx'),
    'y': ('y', 'gaze_y', 'gazey', 'pointy', 'point_y', 'ycoord', 'y_coord',
          'ypix', 'pixely', 'normy'),
    'gaze_type': ('type', 'gaze_type', 'gazetype', 'event', 'event_type',
                  'fixation', 'label', 'status'),
}


@dataclass
class GazeFormat:
    """Detected file format summary."""

    path: str
    kind: str
    delimiter: str = 'whitespace'
    has_header: bool = False
    columns: Dict[str, Optional[int | str]] = field(default_factory=dict)
    coordinate_mode: str = 'unknown'
    warnings: List[str] = field(default_factory=list)
    sampled_rows: int = 0
    skipped_rows: int = 0
    x_range: Tuple[Optional[float], Optional[float]] = (None, None)
    y_range: Tuple[Optional[float], Optional[float]] = (None, None)
    gaze_type_counts: Dict[str, int] = field(default_factory=dict)


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


def canonicalize_gaze_type(value: object) -> str:
    if value is None:
        return 'unknown'
    text = str(value).strip().lower()
    if not text:
        return 'unknown'
    aliases = {
        '1.0': '1',
        '0.0': '0',
        'fix': 'fixation',
        'fixations': 'fixation',
        'sac': 'saccade',
    }
    return aliases.get(text, text)


def is_fixation_type(value: object,
                     fixation_values: Optional[Sequence[str]] = None) -> bool:
    fixation_pool = set(DEFAULT_FIXATION_VALUES)
    if fixation_values:
        fixation_pool |= {str(item).strip().lower() for item in fixation_values}
    return canonicalize_gaze_type(value) in fixation_pool


def looks_like_header(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    numeric_count = sum(maybe_float(token) is not None for token in tokens)
    alpha_hits = 0
    for token in tokens:
        norm = normalize_token(token)
        if any(norm in aliases for aliases in HEADER_ALIASES.values()):
            alpha_hits += 2
        elif re.search(r'[a-zA-Z]', token):
            alpha_hits += 1
    return alpha_hits >= max(1, len(tokens) // 2) and numeric_count < len(tokens)


def _column_numeric_stats(rows: Sequence[Sequence[str]], idx: int) -> dict:
    values = [maybe_float(row[idx]) for row in rows if idx < len(row)]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return dict(ratio=0.0, min=None, max=None, mean=None, monotonic=False,
                    integer_like=False)
    monotonic = all(
        numeric[pos] <= numeric[pos + 1] for pos in range(len(numeric) - 1))
    integer_like = all(abs(value - round(value)) < 1e-4 for value in numeric)
    return dict(
        ratio=len(numeric) / max(1, len(values)),
        min=min(numeric),
        max=max(numeric),
        mean=float(np.mean(numeric)),
        monotonic=monotonic,
        integer_like=integer_like,
    )


def infer_columns(header: Optional[Sequence[str]],
                  rows: Sequence[Sequence[str]]) -> Dict[str, Optional[int]]:
    columns: Dict[str, Optional[int]] = {
        'frame_id': None,
        'timestamp': None,
        'x': None,
        'y': None,
        'gaze_type': None,
    }
    if not rows:
        return columns
    max_cols = max(len(row) for row in rows)
    stats = [_column_numeric_stats(rows, idx) for idx in range(max_cols)]

    if header:
        normalized_header = [normalize_token(item) for item in header]
        for field, aliases in HEADER_ALIASES.items():
            for idx, name in enumerate(normalized_header):
                if name in aliases:
                    columns[field] = idx
                    break

    monotonic_candidates = [
        idx for idx, stat in enumerate(stats)
        if stat['ratio'] >= 0.8 and stat['monotonic']
    ]
    if columns['frame_id'] is None:
        for idx in monotonic_candidates:
            if stats[idx]['integer_like']:
                columns['frame_id'] = idx
                break
    if columns['timestamp'] is None:
        for idx in monotonic_candidates:
            if idx != columns['frame_id']:
                columns['timestamp'] = idx
                break

    coord_scores = []
    for idx, stat in enumerate(stats):
        if stat['ratio'] < 0.8:
            continue
        if idx in (columns['frame_id'], columns['timestamp']):
            continue
        score = 0.0
        min_v, max_v = stat['min'], stat['max']
        if min_v is None or max_v is None:
            continue
        if -0.25 <= min_v <= 1.25 and -0.25 <= max_v <= 1.25:
            score += 4.0
        if -5 <= min_v <= 8192 and 0 <= max_v <= 8192:
            score += 1.5
        if not stat['monotonic']:
            score += 0.5
        coord_scores.append((score, idx))
    coord_scores.sort(reverse=True)

    if columns['x'] is None and coord_scores:
        columns['x'] = coord_scores[0][1]
    if columns['y'] is None:
        for _, idx in coord_scores[1:]:
            if idx != columns['x']:
                columns['y'] = idx
                break
    if columns['y'] is None and columns['x'] is not None:
        fallback = columns['x'] + 1
        if fallback < max_cols:
            columns['y'] = fallback

    if columns['gaze_type'] is None and header:
        for idx, name in enumerate(normalized_header):
            if any(token in name for token in ('type', 'event', 'label')):
                columns['gaze_type'] = idx
                break

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
    lines, _ = read_text_lines(path, max_lines=(max_records or 2000))
    if not lines:
        fmt = GazeFormat(path=path, kind='text', warnings=['empty file'])
        return ParsedGazeData(path=path, records=[], gaze_format=fmt)
    delimiter = detect_delimiter(lines)
    first_row = split_text_line(lines[0], delimiter)
    has_header = looks_like_header(first_row)
    header = first_row if has_header else None
    row_lines = lines[1:] if has_header else lines
    data_rows = [split_text_line(line, delimiter) for line in row_lines]
    columns = infer_columns(header, data_rows)

    records: List[dict] = []
    skipped = 0
    x_values: List[float] = []
    y_values: List[float] = []
    gaze_type_counts: Counter[str] = Counter()
    timestamps: List[float] = []

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
            record = dict(
                frame_id=frame_id,
                timestamp=timestamp,
                x=x,
                y=y,
                gaze_type=gaze_type,
            )
            records.append(record)
            x_values.append(x)
            y_values.append(y)
            gaze_type_counts[gaze_type] += 1
            if timestamp is not None:
                timestamps.append(timestamp)
        except Exception:
            skipped += 1

    fmt = GazeFormat(
        path=path,
        kind='text',
        delimiter=delimiter,
        has_header=has_header,
        columns=columns,
        coordinate_mode=detect_coordinate_mode(x_values, y_values),
        sampled_rows=len(records),
        skipped_rows=skipped,
        x_range=(min(x_values), max(x_values)) if x_values else (None, None),
        y_range=(min(y_values), max(y_values)) if y_values else (None, None),
        gaze_type_counts=dict(gaze_type_counts),
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
    if normalized_keys:
        for alias in HEADER_ALIASES['frame_id']:
            if alias in normalized_keys:
                frame_key = normalized_keys[alias]
                break
        for alias in HEADER_ALIASES['timestamp']:
            if alias in normalized_keys:
                time_key = normalized_keys[alias]
                break
        for alias in HEADER_ALIASES['x']:
            if alias in normalized_keys:
                x_key = normalized_keys[alias]
                break
        for alias in HEADER_ALIASES['y']:
            if alias in normalized_keys:
                y_key = normalized_keys[alias]
                break
        for alias in HEADER_ALIASES['gaze_type']:
            if alias in normalized_keys:
                gaze_type_key = normalized_keys[alias]
                break

    records: List[dict] = []
    skipped = 0
    x_values: List[float] = []
    y_values: List[float] = []
    gaze_type_counts: Counter[str] = Counter()
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
        records.append(
            dict(frame_id=frame_id, timestamp=timestamp, x=x, y=y,
                 gaze_type=gaze_type))
        x_values.append(x)
        y_values.append(y)
        gaze_type_counts[gaze_type] += 1
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
        ),
        coordinate_mode=detect_coordinate_mode(x_values, y_values),
        sampled_rows=len(records),
        skipped_rows=skipped,
        x_range=(min(x_values), max(x_values)) if x_values else (None, None),
        y_range=(min(y_values), max(y_values)) if y_values else (None, None),
        gaze_type_counts=dict(gaze_type_counts),
    )
    return ParsedGazeData(
        path=path,
        records=records,
        gaze_format=fmt,
        timestamp_unit=detect_timestamp_unit(timestamps),
    )


@lru_cache(maxsize=512)
def parse_gaze_file(path: str,
                    max_records: Optional[int] = None) -> ParsedGazeData:
    ext = Path(path).suffix.lower()
    if ext == '.json':
        return parse_json_gaze_file(path, max_records=max_records)
    return parse_text_gaze_file(path, max_records=max_records)


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
    candidates = [Path(video_path).stem, Path(video_path).parent.name]
    patterns = [
        re.compile(r'F0*(\d{3,})'),
        re.compile(r'frame[_-]?0*(\d{3,})', re.IGNORECASE),
        re.compile(r'0*(\d{3,})-0*(\d{3,})'),
        re.compile(r'start[_-]?0*(\d{3,})', re.IGNORECASE),
    ]
    for text in candidates:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
    return None


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


def pick_record_for_frame(records: Sequence[dict],
                          fixation_only: bool = False,
                          fixation_values: Optional[Sequence[str]] = None
                          ) -> Optional[dict]:
    if not records:
        return None
    if fixation_only:
        filtered = [
            item for item in records
            if is_fixation_type(item.get('gaze_type'), fixation_values)
        ]
        if filtered:
            records = filtered
    return records[0]


def align_gaze_to_clip(parsed_gaze: ParsedGazeData,
                       total_frames: int,
                       fps: float,
                       start_frame: Optional[int] = None,
                       only_fixation: bool = False,
                       fixation_values: Optional[Sequence[str]] = None,
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

    gaze_xy = np.zeros((total_frames, 2), dtype=np.float32)
    gaze_valid = np.zeros((total_frames,), dtype=np.uint8)
    gaze_type = np.array(['unknown'] * total_frames, dtype=object)
    warning_codes: List[str] = []
    matched_fixations = 0

    for clip_idx in range(total_frames):
        local_target = clip_idx
        global_target = (start_frame + clip_idx) if start_frame is not None else local_target
        chosen = None
        if frame_to_records:
            chosen = pick_record_for_frame(
                frame_to_records.get(global_target, []),
                fixation_only=only_fixation,
                fixation_values=fixation_values)
            if chosen is None and start_frame is None:
                chosen = pick_record_for_frame(
                    frame_to_records.get(local_target, []),
                    fixation_only=only_fixation,
                    fixation_values=fixation_values)
        if chosen is None and time_pairs:
            target_time = clip_idx / max(fps, 1e-6)
            if start_frame is not None:
                target_time = global_target / max(fps, 1e-6)
            best_delta = None
            best_record = None
            for timestamp, record in time_pairs:
                delta = abs(timestamp - target_time)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_record = record
            if best_record is not None and (best_delta is None or best_delta <= nearest_tolerance):
                candidate = pick_record_for_frame(
                    [best_record],
                    fixation_only=only_fixation,
                    fixation_values=fixation_values)
                chosen = candidate

        if chosen is None:
            continue
        gaze_xy[clip_idx, 0] = float(chosen['x'])
        gaze_xy[clip_idx, 1] = float(chosen['y'])
        gaze_valid[clip_idx] = 1
        gaze_type[clip_idx] = canonicalize_gaze_type(chosen.get('gaze_type'))
        if is_fixation_type(chosen.get('gaze_type'), fixation_values):
            matched_fixations += 1

    if start_frame is None:
        warning_codes.append('missing_clip_start_frame')
    if matched_fixations == 0 and only_fixation:
        warning_codes.append('no_fixation_match')
    return dict(
        gaze_xy=gaze_xy,
        gaze_valid=gaze_valid,
        gaze_type=gaze_type,
        frame_indices=np.arange(total_frames, dtype=np.int32),
        warning_codes=warning_codes,
        matched_fixations=matched_fixations,
    )


def normalize_xy_array(gaze_xy: np.ndarray,
                       coordinate_mode: str,
                       image_size: Tuple[int, int]) -> np.ndarray:
    output = np.asarray(gaze_xy, dtype=np.float32).copy()
    width, height = image_size
    if coordinate_mode == 'pixel':
        if width <= 0 or height <= 0:
            raise ValueError('image_size must be positive for pixel coordinates')
        output[:, 0] /= float(width)
        output[:, 1] /= float(height)
    elif coordinate_mode == 'unknown':
        finite = output[np.isfinite(output)]
        if finite.size and float(np.nanmax(np.abs(finite))) > 1.5 and width > 0 and height > 0:
            output[:, 0] /= float(width)
            output[:, 1] /= float(height)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
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
