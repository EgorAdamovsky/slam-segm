"""Маска людей и машин для ORB-SLAM3.

255 — пиксель динамический, точку SuperPoint на нём надо выбросить.
Дорога и прочий фон остаются нулями.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_model = None
_conf = 0.4


def init_mask(project_dir: str, weights: str, conf: float, device: str = "cuda") -> None:
    global _model, _conf
    root = str(Path(project_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from run_video import SegmentationModel

    _conf = float(conf)
    _model = SegmentationModel(Path(weights), device or "cuda", "auto")


def reject_mask(data: bytes, height: int, width: int, rgb: int) -> bytes:
    """Кадр HxWx3. rgb=0 значит BGR, как Camera.RGB: 0 в ORB-SLAM."""
    if _model is None:
        raise RuntimeError("маска не инициализирована")
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    if not rgb:
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    detections = _model(image, _conf, 0.5, 30)
    mask = np.zeros((height, width), dtype=np.uint8)
    for det in detections:
        if det["group"] not in ("person", "car"):
            continue
        raw = det["mask"]
        if raw is None:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 255
            continue
        if raw.shape != (height, width):
            raw = cv2.resize(raw.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask[raw] = 255
    # SuperPoint садится на кромку машины, чуть снаружи маски.
    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel)
    return mask.tobytes()
