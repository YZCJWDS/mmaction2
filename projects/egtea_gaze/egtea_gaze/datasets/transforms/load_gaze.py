"""Training-time transform for loading EGTEA gaze caches."""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS
from projects.egtea_gaze.egtea_gaze.utils import (apply_crop_and_flip_to_xy,
                                                   gaze_xy_to_heatmaps,
                                                   safe_path_id, stable_hash)


@TRANSFORMS.register_module()
class LoadGazeMap(BaseTransform):
    """Load offline gaze cache and build low-resolution targets after augments.

    The cache stores clip-aligned ``gaze_xy`` and ``gaze_valid`` arrays.
    This transform applies current crop/flip metadata and then produces
    lightweight 14x14 targets for training.
    """

    def __init__(self,
                 gaze_map_root: str,
                 metadata_file: Optional[str] = None,
                 heatmap_size: Tuple[int, int] = (14, 14),
                 sigma: float = 1.5,
                 num_frames: Optional[int] = None,
                 missing_policy: str = 'zeros',
                 temporal_align: str = 'sampled',
                 key_by: str = 'filename',
                 dtype: str = 'float32',
                 gaze_mode: str = 'real',
                 cache_size: int = 128) -> None:
        self.gaze_map_root = gaze_map_root
        self.metadata_file = metadata_file or os.path.join(gaze_map_root,
                                                           'metadata.json')
        self.heatmap_size = tuple(int(v) for v in heatmap_size)
        self.sigma = float(sigma)
        self.num_frames = num_frames
        self.missing_policy = missing_policy
        self.temporal_align = temporal_align
        self.key_by = key_by
        self.dtype = np.float32 if dtype == 'float32' else np.float16
        self.gaze_mode = gaze_mode
        self.cache_size = int(cache_size)
        self._sample_meta: Dict[str, dict] = {}
        self._all_keys = []
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._search_cache: Dict[str, Optional[str]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not os.path.exists(self.metadata_file):
            self._sample_meta = {}
            self._all_keys = []
            return
        with open(self.metadata_file, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        self._sample_meta = metadata.get('samples', {})
        self._all_keys = sorted(self._sample_meta.keys())

    def _candidate_video_keys(self, results: dict) -> Sequence[str]:
        filename = results.get('filename') or results.get('video_path') or ''
        candidates = []
        if 'video_relpath' in results:
            candidates.append(results['video_relpath'])
        if filename:
            normalized = filename.replace('\\', '/')
            candidates.append(normalized)
            if '/cropped_clips/' in normalized:
                candidates.append(normalized.split('/cropped_clips/', 1)[1])
                candidates.append('cropped_clips/' + normalized.split('/cropped_clips/', 1)[1])
            if '/videos/' in normalized:
                tail = normalized.split('/videos/', 1)[1]
                candidates.append(tail)
        unique = []
        for item in candidates:
            if item and item not in unique:
                unique.append(item)
        return unique

    def _resolve_npz_path(self, video_key: str) -> Optional[str]:
        if video_key in self._sample_meta:
            rel_path = self._sample_meta[video_key]['npz_path']
            return os.path.join(self.gaze_map_root, rel_path)
        if video_key in self._search_cache:
            return self._search_cache[video_key]

        stem = Path(video_key).stem
        hashed_name = f'{safe_path_id(stem)}_{stable_hash(video_key)}.npz'
        matches = []
        for root, _, filenames in os.walk(self.gaze_map_root):
            if hashed_name in filenames:
                matches.append(os.path.join(root, hashed_name))
        resolved = matches[0] if matches else None
        self._search_cache[video_key] = resolved
        return resolved

    def _cached_load(self, npz_path: str) -> dict:
        if npz_path in self._cache:
            payload = self._cache.pop(npz_path)
            self._cache[npz_path] = payload
            return payload
        data = np.load(npz_path, allow_pickle=True)
        payload = {key: data[key] for key in data.files}
        self._cache[npz_path] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def _select_source(self, results: dict) -> Tuple[Optional[dict], Optional[str]]:
        if self.gaze_mode in ('center', 'random', 'zeros'):
            return None, None
        candidate_keys = self._candidate_video_keys(results)
        npz_path = None
        if self.gaze_mode == 'shuffle' and self._all_keys:
            random_key = random.choice(self._all_keys)
            info = self._sample_meta.get(random_key)
            if info is not None:
                npz_path = os.path.join(self.gaze_map_root, info['npz_path'])
        else:
            for key in candidate_keys:
                npz_path = self._resolve_npz_path(key)
                if npz_path:
                    break
        if npz_path is None:
            return None, None
        return self._cached_load(npz_path), npz_path

    def _sample_indices(self, results: dict, total_available: int) -> np.ndarray:
        frame_inds = results.get('frame_inds')
        if frame_inds is None:
            count = self.num_frames or total_available
            return np.arange(min(count, total_available), dtype=np.int32)
        if hasattr(frame_inds, 'detach'):
            frame_inds = frame_inds.detach().cpu().numpy()
        frame_inds = np.asarray(frame_inds).reshape(-1).astype(np.int32)
        frame_inds = np.clip(frame_inds, 0, max(total_available - 1, 0))
        return frame_inds

    def _make_base_xy(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.gaze_mode == 'center':
            xy = np.full((length, 2), 0.5, dtype=np.float32)
            valid = np.ones((length,), dtype=np.uint8)
            return xy, valid
        if self.gaze_mode == 'random':
            xy = np.random.uniform(0.0, 1.0, size=(length, 2)).astype(np.float32)
            valid = np.ones((length,), dtype=np.uint8)
            return xy, valid
        if self.gaze_mode == 'zeros':
            return np.zeros((length, 2), dtype=np.float32), np.zeros((length,), dtype=np.uint8)
        raise KeyError(self.gaze_mode)

    def _missing_output(self, results: dict, reason: str) -> dict:
        count = len(results.get('frame_inds', []))
        if count == 0:
            count = self.num_frames or 0
        gaze_xy = np.zeros((count, 2), dtype=np.float32)
        gaze_valid = np.zeros((count,), dtype=np.uint8)
        gaze_maps = gaze_xy_to_heatmaps(
            gaze_xy, gaze_valid, self.heatmap_size, self.sigma).astype(self.dtype)
        results['gaze_xy'] = gaze_xy
        results['gaze_valid'] = gaze_valid
        results['gaze_maps'] = gaze_maps
        results['gaze_source'] = reason
        return results

    def transform(self, results: dict) -> dict:
        payload, source_path = self._select_source(results)
        if payload is None and self.gaze_mode in ('center', 'random', 'zeros'):
            gaze_xy, gaze_valid = self._make_base_xy(
                len(results.get('frame_inds', [])) or (self.num_frames or 0))
        elif payload is None:
            if self.missing_policy == 'skip':
                raise FileNotFoundError(f'Missing gaze cache for: {results.get("filename")}')
            if self.missing_policy == 'warn':
                import warnings
                warnings.warn(f'Gaze cache missing: {results.get("filename")}')
            return self._missing_output(results, 'missing_cache')
        else:
            source_xy = np.asarray(payload['gaze_xy'], dtype=np.float32)
            source_valid = np.asarray(payload['gaze_valid'], dtype=np.uint8)
            frame_inds = self._sample_indices(results, source_xy.shape[0])
            gaze_xy = source_xy[frame_inds]
            gaze_valid = source_valid[frame_inds]

        crop_quadruple = results.get('crop_quadruple')
        flip = bool(results.get('flip', False))
        flip_direction = results.get('flip_direction', 'horizontal')
        gaze_xy, gaze_valid = apply_crop_and_flip_to_xy(
            gaze_xy, gaze_valid, crop_quadruple=crop_quadruple, flip=flip,
            flip_direction=flip_direction)
        gaze_maps = gaze_xy_to_heatmaps(
            gaze_xy, gaze_valid, self.heatmap_size, self.sigma).astype(self.dtype)
        gaze_maps = np.nan_to_num(gaze_maps, nan=0.0, posinf=0.0, neginf=0.0)

        results['gaze_xy'] = gaze_xy.astype(np.float32)
        results['gaze_valid'] = gaze_valid.astype(np.uint8)
        results['gaze_maps'] = gaze_maps.astype(self.dtype)
        results['gaze_source'] = source_path or self.gaze_mode
        return results

    @staticmethod
    def _self_test() -> None:
        transform = LoadGazeMap(gaze_map_root='.', metadata_file='missing.json')
        dummy = dict(frame_inds=np.array([0, 1, 2], dtype=np.int32))
        output = transform._missing_output(dummy, 'test')
        assert output['gaze_maps'].shape[0] == 3
