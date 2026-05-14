"""Overlay helpers for gaze heatmaps and attention maps."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return array


def resize_heatmap(heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode='F')
    image = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def heatmap_to_rgb(heatmap: np.ndarray,
                   color: Tuple[int, int, int] = (255, 64, 64)) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    hm = hm - hm.min()
    max_value = float(hm.max())
    if max_value > 0:
        hm = hm / max_value
    output = np.zeros(hm.shape + (3,), dtype=np.uint8)
    for channel, value in enumerate(color):
        output[..., channel] = np.clip(hm * value, 0, 255).astype(np.uint8)
    return output


def blend_heatmap_on_rgb(image: np.ndarray,
                         heatmap: np.ndarray,
                         alpha: float = 0.45,
                         color: Tuple[int, int, int] = (255, 64, 64)
                         ) -> np.ndarray:
    rgb = ensure_uint8_rgb(image)
    overlay = heatmap_to_rgb(resize_heatmap(heatmap, (rgb.shape[1], rgb.shape[0])),
                             color=color)
    blended = ((1.0 - alpha) * rgb.astype(np.float32) +
               alpha * overlay.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_gaze_point(image: np.ndarray,
                    xy_norm: Sequence[float],
                    radius: int = 10,
                    color: Tuple[int, int, int] = (255, 32, 32),
                    outline: Tuple[int, int, int] = (255, 255, 255)
                    ) -> np.ndarray:
    rgb = ensure_uint8_rgb(image)
    x = float(xy_norm[0]) * max(rgb.shape[1] - 1, 1)
    y = float(xy_norm[1]) * max(rgb.shape[0] - 1, 1)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    cross = radius + 6
    draw.line((x - cross, y, x + cross, y), fill=outline, width=3)
    draw.line((x, y - cross, x, y + cross), fill=outline, width=3)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=color, outline=outline, width=3)
    return np.asarray(pil_image)


def draw_text_box(image: np.ndarray,
                  lines: Sequence[str],
                  anchor: Tuple[int, int] = (8, 8),
                  text_fill: Tuple[int, int, int] = (255, 255, 255),
                  box_fill: Tuple[int, int, int] = (0, 0, 0),
                  box_outline: Tuple[int, int, int] = (255, 255, 255)
                  ) -> np.ndarray:
    rgb = ensure_uint8_rgb(image)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    text = '\n'.join(lines)
    left, top = anchor
    bbox = draw.multiline_textbbox((left, top), text, font=font, spacing=2)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=box_fill,
        outline=box_outline,
        width=1)
    draw.multiline_text((left, top), text, fill=text_fill, font=font, spacing=2)
    return np.asarray(pil_image)


def save_image(array: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_uint8_rgb(array)).save(output_path)


def image_to_data_uri(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(ensure_uint8_rgb(array)).save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def write_simple_gallery(items: Iterable[dict], output_html: str,
                         title: str = 'Gallery') -> None:
    rows: List[str] = [
        '<html><head><meta charset="utf-8"><title>{}</title>'.format(title),
        '<style>body{font-family:Arial,sans-serif;margin:24px;}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;}'
        '.card{border:1px solid #ddd;padding:12px;border-radius:6px;}'
        '.card img{max-width:100%;height:auto;display:block;margin-bottom:8px;}'
        '.meta{font-size:12px;color:#444;white-space:pre-wrap;}</style></head><body>',
        f'<h1>{title}</h1><div class="grid">'
    ]
    for item in items:
        image_src = item.get('image')
        if image_src and os.path.exists(image_src):
            image_src = image_to_data_uri(np.asarray(Image.open(image_src)))
        rows.append('<div class="card">')
        if image_src:
            rows.append(f'<img src="{image_src}" alt="preview">')
        rows.append(f'<div><strong>{item.get("title", "sample")}</strong></div>')
        rows.append(f'<div class="meta">{item.get("meta", "")}</div>')
        rows.append('</div>')
    rows.append('</div></body></html>')
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(rows))
