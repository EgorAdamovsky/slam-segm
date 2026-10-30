#!/usr/bin/env python3
"""Сегментация видео и оценка диспаратности для стереопары.

Модели родные, PyTorch, без ONNX:
- RF-DETR Seg Preview: как rfdetr.detr.RFDETR.predict. Кадр приводится к 432,
  делится на 255 и нормализуется ImageNet-статистикой. Маски и коробки
  возвращает штатный PostProcess.
- Публичные веса — COCO (человек, машина, автобус, грузовик).
  Если в чекпоинте около 30 классов, включается разметка Cityscapes,
  как SegmentationClass::from_raw в Rust.
- BANet: как test.py. RGB 0..255, паддинг InputPadder до кратного 32.
- Точки переводятся матрицей 4x4 так же, как glam Mat4::project_point3.

Веса по умолчанию (каталог weights в .gitignore):
  weights/rf-detr-seg-preview.pt
  weights/banet-2d/kitti.pth

Интерпретатор с torch и opencv:
  /home/psu320/Egor/SMOKE-DETR/env/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SEG_WEIGHTS = SCRIPT_DIR / "weights" / "rf-detr-seg-preview.pt"
DEPTH_WEIGHTS = SCRIPT_DIR / "weights" / "banet-2d" / "kitti.pth"
SEG_URL = "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt"

# Та же матрица, что PipelineSettings::default в pipeline_settings.rs.
# from_cols_array_2d(...).transpose() даёт эти строки.
DEFAULT_Q = np.array(
    [
        [1.0, 0.0, 0.0, -713.5791168212891],
        [0.0, 1.0, 0.0, -570.9349365234375],
        [0.0, 0.0, 0.0, 1164.6238115833075],
        [0.0, 0.0, 1.9625989469856626, 0.0],
    ],
    dtype=np.float32,
)

CONF_TH = 0.4

# Имена из комментария segmentation.rs, сдвинутые на -1:
# в файле словарь с единицы, в логитах индекс с нуля.
CITYSCAPES_NAMES = [
    "road",
    "sidewalk",
    "parking",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "on rails",
    "motorcycle",
    "bicycle",
    "caravan",
    "trailer",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "pole group",
    "traffic sign",
    "traffic light",
    "vegetation",
    "terrain",
    "sky",
    "ground",
    "dynamic",
    "static",
]

# COCO id совпадает с индексом логита у RF-DETR preview (export_to_onnx.py).
COCO_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# BGR
GROUP_COLOR = {
    "road": (80, 180, 60),
    "person": (60, 60, 220),
    "car": (40, 140, 255),
    "other": (180, 120, 40),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def semantic_group(class_index: int, vocabulary: str) -> str:
    """Группа, которую дальше ест извлечение объектов. Как в Rust."""
    if vocabulary == "coco":
        if class_index == 1:
            return "person"
        if class_index in (3, 6, 8):
            return "car"
        return "other"
    # cityscapes, SegmentationClass::from_raw
    if class_index == 0:
        return "road"
    if class_index in (3, 4):
        return "person"
    # truck (6) в Rust сознательно не входит в Car: там 5 | 7 | 8.
    if class_index in (5, 7, 8):
        return "car"
    return "other"


def class_name(class_index: int, vocabulary: str) -> str:
    if vocabulary == "coco":
        return COCO_NAMES.get(class_index, f"class_{class_index}")
    if 0 <= class_index < len(CITYSCAPES_NAMES):
        return CITYSCAPES_NAMES[class_index]
    return f"class_{class_index}"


def guess_vocabulary(num_classes: int, requested: str) -> str:
    if requested != "auto":
        return requested
    # У preview-головы 90 или 91 логит, у Cityscapes в комментарии 29 имён.
    if num_classes >= 80:
        return "coco"
    return "cityscapes"


def pad_to_multiple(image: np.ndarray, divisor: int = 32) -> tuple[np.ndarray, int, int]:
    """Нули справа и снизу, как DepthModel::process в depth.rs."""
    height, width = image.shape[:2]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    out = np.zeros((height + pad_h, width + pad_w, image.shape[2]), dtype=image.dtype)
    out[:height, :width] = image
    return out, height, width


def disparity_to_points(disparity: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Пиксель (x, y, disparity) -> метры. Деление на w, как project_point3."""
    height, width = disparity.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    ones = np.ones_like(disparity, dtype=np.float32)
    pixels = np.stack([xs, ys, disparity, ones], axis=-1)
    homogeneous = pixels @ q_matrix.T
    w = homogeneous[..., 3:4]
    points = np.zeros((*disparity.shape, 3), dtype=np.float32)
    valid = (disparity > 0) & np.isfinite(w[..., 0]) & (np.abs(w[..., 0]) > 1e-6)
    points[valid] = homogeneous[..., :3][valid] / w[valid]
    return points


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float, limit: int) -> list[dict]:
    if iou_threshold <= 0:
        return detections[:limit] if limit else detections
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    for det in ordered:
        if limit and len(kept) >= limit:
            break
        if all(iou_xyxy(det["xyxy"], prev["xyxy"]) <= iou_threshold for prev in kept):
            kept.append(det)
    return kept


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, беру CPU")
        return "cpu"
    return name


class SegmentationModel:
    """Родной RF-DETR. Тот же вход, что у RFDETR.predict, без ONNX и без supervision."""

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, path: Path, device: str, vocabulary: str):
        # rfdetr/__init__.py тянет supervision. Для инференса она не нужна.
        import types
        if "rfdetr" not in sys.modules:
            package = types.ModuleType("rfdetr")
            package.__path__ = [str(SCRIPT_DIR / "rfdetr")]
            package.__package__ = "rfdetr"
            sys.modules["rfdetr"] = package
        from rfdetr.config import RFDETRSegPreviewConfig
        from rfdetr.main import Model

        config = RFDETRSegPreviewConfig(pretrain_weights=str(path), device=device)
        self.resolution = config.resolution
        self.wrapper = Model(**config.model_dump())
        self.wrapper.model.eval()
        self.device = device
        logit_count = int(self.wrapper.model.class_embed.bias.shape[0])
        self.vocabulary = guess_vocabulary(logit_count, vocabulary)
        print(f"[RF-DETR] {path.name}, {self.vocabulary}, логитов {logit_count}, {device}")

    def __call__(self, image_rgb: np.ndarray, conf: float, iou: float, limit: int) -> list[dict]:
        height, width = image_rgb.shape[:2]
        image = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        image = TF.normalize(image, self.MEANS, self.STDS)
        image = TF.resize(image, [self.resolution, self.resolution], antialias=True)
        batch = image.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            predictions = self.wrapper.model(batch)
        if isinstance(predictions, tuple):
            predictions = {
                "pred_boxes": predictions[0],
                "pred_logits": predictions[1],
                "pred_masks": predictions[2],
            }
        target = torch.tensor([[height, width]], device=self.device)
        result = self.wrapper.postprocess(predictions, target)[0]
        scores = result["scores"].detach().float().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        boxes = result["boxes"].detach().float().cpu().numpy()
        masks = result["masks"].detach().cpu().numpy() if "masks" in result else None

        detections = []
        for index, confidence in enumerate(scores):
            confidence = float(confidence)
            if confidence < conf:
                continue
            class_index = int(labels[index])
            mask = None
            if masks is not None:
                raw = masks[index]
                if raw.ndim == 3:
                    raw = raw[0]
                mask = raw.astype(bool)
            detections.append(
                {
                    "class_index": class_index,
                    "name": class_name(class_index, self.vocabulary),
                    "group": semantic_group(class_index, self.vocabulary),
                    "confidence": confidence,
                    "xyxy": boxes[index].astype(np.float32),
                    "mask": mask,
                }
            )
        return nms(detections, iou, limit)


def _load_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    cleaned = {}
    for key, value in checkpoint.items():
        name = key[7:] if key.startswith("module.") else key
        cleaned[name] = value
    return cleaned


class DepthModel:
    """Родной BANet. Вход как в test.py: RGB 0..255 и InputPadder."""

    def __init__(self, path: Path, device: str):
        banet_root = SCRIPT_DIR / "BANet" / "banet-2d"
        for extra in (banet_root, banet_root / "core"):
            text = str(extra)
            if text not in sys.path:
                sys.path.insert(0, text)
        from core.BANet import BANet
        from core.utils.utils import InputPadder

        self.InputPadder = InputPadder
        self.device = device
        model = BANet(None)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(_load_state_dict(checkpoint), strict=True)
        self.model = model.to(device).eval()
        print(f"[BANet] {path.name}, {device}")

    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        if left_rgb.shape != right_rgb.shape:
            raise ValueError("Левый и правый кадры разного размера")

        def tensor(image: np.ndarray) -> torch.Tensor:
            array = np.ascontiguousarray(image)
            return torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)

        left = tensor(left_rgb).to(self.device)
        right = tensor(right_rgb).to(self.device)
        padder = self.InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with torch.inference_mode():
            disparity = self.model(left, right)
        disparity = padder.unpad(disparity)
        return disparity.squeeze().detach().float().cpu().numpy().astype(np.float32)


def mask_on_frame(mask: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == (height, width):
        return mask
    # Маска модели 108x108, координаты считаются от исходного кадра, как object_at_pixel_coords.
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def draw_segmentation(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    # Сначала дорога, потом остальные, чтобы объекты не потерялись под ней.
    ordered = sorted(detections, key=lambda item: 0 if item["group"] == "road" else 1)
    for det in ordered:
        mask = mask_on_frame(det["mask"], width, height)
        if mask is None:
            continue
        color = GROUP_COLOR.get(det["group"], GROUP_COLOR["other"])
        overlay[mask] = color
    alpha = 0.35
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
    for det in ordered:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = GROUP_COLOR.get(det["group"], GROUP_COLOR["other"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det['name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 4, top + th + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = disparity[disparity > 0]
    if valid.size == 0:
        gray = np.zeros(disparity.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(valid, [2, 98])
        if high <= low:
            high = low + 1.0
        gray = np.clip((disparity - low) / (high - low), 0, 1)
        gray = (gray * 255).astype(np.uint8)
        gray[disparity <= 0] = 0
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def objects_from_points(points: np.ndarray, detections: list[dict]) -> list[dict]:
    """Ближайшая точка человека и машины, как ObjectExtractor."""
    height, width = points.shape[:2]
    found = []
    for det in detections:
        if det["group"] not in ("person", "car"):
            continue
        mask = mask_on_frame(det["mask"], width, height)
        if mask is None:
            x1, y1, x2, y2 = det["xyxy"]
            mask = np.zeros((height, width), dtype=bool)
            mask[int(max(0, y1)) : int(min(height, y2)), int(max(0, x1)) : int(min(width, x2))] = True
        chosen = points[mask]
        chosen = chosen[np.any(chosen != 0, axis=1)]
        if len(chosen) == 0:
            continue
        distances = np.linalg.norm(chosen, axis=1)
        nearest = chosen[int(np.argmin(distances))]
        found.append(
            {
                "group": det["group"],
                "name": det["name"],
                "confidence": det["confidence"],
                "position": [float(v) for v in nearest],
                "center": [float(v) for v in chosen.mean(axis=0)],
            }
        )
    return found


def detections_to_json(detections: list[dict], objects: list[dict] | None) -> dict:
    payload = {
        "detections": [
            {
                "name": det["name"],
                "group": det["group"],
                "class_index": det["class_index"],
                "confidence": round(det["confidence"], 4),
                "bbox_xyxy": [round(float(v), 2) for v in det["xyxy"]],
            }
            for det in detections
        ]
    }
    if objects is not None:
        payload["objects"] = objects
    return payload


def list_images(folder: Path) -> list[Path]:
    files = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda path: path.name)


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Не открывается видео: {path}")
    return capture


def split_stereo(frame: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    if layout == "sbs":
        half = width // 2
        return frame[:, :half], frame[:, half : half * 2]
    half = height // 2
    return frame[:half], frame[half : half * 2]


class FrameReader:
    """Отдаёт левый кадр и, если стерео, правый. Кадры в BGR, как у OpenCV."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.captures: list[cv2.VideoCapture] = []
        self.mode = self._detect_mode()
        self.fps = 30.0
        self.size = (0, 0)

    def _detect_mode(self) -> str:
        if self.args.left_dir or self.args.right_dir:
            if self.args.left_dir is None:
                raise SystemExit("Для папки кадров нужен --left-dir")
            return "folders"
        if self.args.left or self.args.right:
            if not (self.args.left and self.args.right):
                raise SystemExit("Для пары камер нужны оба пути: --left и --right")
            if self.args.left.suffix.lower() in IMAGE_EXTS:
                return "image_pair"
            return "videos"
        if self.args.video is None:
            raise SystemExit("Нужен --video, либо --left и --right, либо --left-dir и --right-dir")
        if self.args.video.suffix.lower() in IMAGE_EXTS:
            return "image"
        return "video"

    def __iter__(self):
        args = self.args
        if self.mode == "folders":
            yield from self._folders()
        elif self.mode == "image_pair":
            left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
            right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
            if left is None or right is None:
                raise FileNotFoundError("Не читается стереопара картинок")
            self.size = (left.shape[1], left.shape[0])
            yield 0, left, right
        elif self.mode == "image":
            frame = cv2.imread(str(args.video), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(args.video)
            self.size = (frame.shape[1], frame.shape[0])
            if args.layout == "mono":
                yield 0, frame, None
            else:
                left, right = split_stereo(frame, args.layout)
                self.size = (left.shape[1], left.shape[0])
                yield 0, left, right
        elif self.mode == "videos":
            yield from self._two_videos()
        else:
            yield from self._one_video()

    def _skip(self, index: int) -> bool:
        if index < self.args.start:
            return True
        if (index - self.args.start) % self.args.stride != 0:
            return True
        return False

    def _folders(self):
        left_files = list_images(self.args.left_dir)
        right_files = list_images(self.args.right_dir) if self.args.right_dir else []
        right_by_name = {path.name: path for path in right_files}
        produced = 0
        for index, left_path in enumerate(left_files):
            if self._skip(index):
                continue
            if self.args.max_frames and produced >= self.args.max_frames:
                break
            left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right = None
            if self.args.right_dir:
                match = right_by_name.get(left_path.name)
                if match is None and index < len(right_files):
                    match = right_files[index]
                if match is None:
                    continue
                right = cv2.imread(str(match), cv2.IMREAD_COLOR)
            if left is None:
                continue
            self.size = (left.shape[1], left.shape[0])
            yield index, left, right
            produced += 1

    def _one_video(self):
        capture = open_capture(self.args.video)
        self.captures.append(capture)
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        produced = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if self._skip(index):
                index += 1
                continue
            if self.args.max_frames and produced >= self.args.max_frames:
                break
            if self.args.layout == "mono":
                self.size = (frame.shape[1], frame.shape[0])
                yield index, frame, None
            else:
                left, right = split_stereo(frame, self.args.layout)
                self.size = (left.shape[1], left.shape[0])
                yield index, left, right
            produced += 1
            index += 1

    def _two_videos(self):
        left_cap = open_capture(self.args.left)
        right_cap = open_capture(self.args.right)
        self.captures.extend([left_cap, right_cap])
        self.fps = left_cap.get(cv2.CAP_PROP_FPS) or 30.0
        produced = 0
        index = 0
        while True:
            ok_l, left = left_cap.read()
            ok_r, right = right_cap.read()
            if not ok_l or not ok_r:
                break
            if self._skip(index):
                index += 1
                continue
            if self.args.max_frames and produced >= self.args.max_frames:
                break
            self.size = (left.shape[1], left.shape[0])
            yield index, left, right
            produced += 1
            index += 1

    def close(self):
        for capture in self.captures:
            capture.release()


def fit_preview(image: np.ndarray, max_width: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сегментация видео. Для стерео дополнительно считается диспаратность."
    )
    parser.add_argument("--video", type=Path, help="Одно видео или одна картинка")
    parser.add_argument("--left", type=Path, help="Левое видео или левая картинка")
    parser.add_argument("--right", type=Path, help="Правое видео или правая картинка")
    parser.add_argument("--left-dir", type=Path, help="Папка кадров левой камеры")
    parser.add_argument("--right-dir", type=Path, help="Папка кадров правой камеры")
    parser.add_argument(
        "--layout",
        choices=("mono", "sbs", "tb"),
        default="mono",
        help="mono: весь кадр. sbs: левая и правая половины. tb: верх и низ",
    )
    parser.add_argument("--seg", type=Path, default=SEG_WEIGHTS, help="Чекпоинт RF-DETR .pt")
    parser.add_argument("--depth", type=Path, default=DEPTH_WEIGHTS, help="Чекпоинт BANet .pth")
    parser.add_argument("--classes", choices=("auto", "cityscapes", "coco"), default="auto")
    parser.add_argument("--conf", type=float, default=CONF_TH)
    parser.add_argument("--nms", type=float, default=0.5, help="Порог IoU. 0 выключает NMS")
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--out", type=Path, help="Куда писать vis.mp4, depth.mp4 и detections.jsonl")
    parser.add_argument("--show", action="store_true", help="Окно предпросмотра, выход по q или Esc")
    parser.add_argument("--no-depth", action="store_true", help="Не считать диспаратность даже для стерео")
    parser.add_argument("--max-frames", type=int, default=0, help="0 значит весь ролик")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--self-check", action="store_true", help="Проверить геометрию без моделей и выйти")
    return parser.parse_args()


def _reader_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        video=None,
        left=None,
        right=None,
        left_dir=None,
        right_dir=None,
        layout="mono",
        start=0,
        stride=1,
        max_frames=0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_clip(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Не пишется проверочное видео: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _check_reader() -> None:
    """Чтение ролика, side-by-side и пары папок без моделей."""
    with tempfile.TemporaryDirectory(prefix="slam-segm-check-") as tmp:
        root = Path(tmp)
        left_frames = []
        right_frames = []
        sbs_frames = []
        for index in range(4):
            left = np.full((8, 10, 3), index, dtype=np.uint8)
            right = np.full((8, 10, 3), 100 + index, dtype=np.uint8)
            left_frames.append(left)
            right_frames.append(right)
            sbs_frames.append(np.hstack([left, right]))
        _write_clip(root / "sbs.avi", sbs_frames)
        _write_clip(root / "left.avi", left_frames)
        _write_clip(root / "right.avi", right_frames)

        sbs = FrameReader(_reader_args(video=root / "sbs.avi", layout="sbs", start=1, stride=2))
        got = [(index, float(left.mean()), float(right.mean()), left.shape[1]) for index, left, right in sbs]
        sbs.close()
        # MJPG слегка меняет заливку, поэтому сравниваем с допуском.
        # Кадры 1 и 3, левая половина — номер кадра, правая — 100+номер, ширина 10.
        expected = [(1, 1.0, 101.0, 10), (3, 3.0, 103.0, 10)]
        if len(got) != len(expected) or any(
            index != exp_i or abs(left_mean - exp_l) > 3 or abs(right_mean - exp_r) > 3 or width != exp_w
            for (index, left_mean, right_mean, width), (exp_i, exp_l, exp_r, exp_w) in zip(got, expected)
        ):
            raise SystemExit(f"Side-by-side прочитался не так: {got}")

        pair = FrameReader(_reader_args(left=root / "left.avi", right=root / "right.avi", max_frames=1))
        pair_got = [(index, float(left.mean()), float(right.mean())) for index, left, right in pair]
        pair.close()
        if len(pair_got) != 1 or pair_got[0][0] != 0 or abs(pair_got[0][1] - 0) > 3 or abs(pair_got[0][2] - 100) > 3:
            raise SystemExit(f"Пара роликов прочиталась не так: {pair_got}")

        left_dir = root / "L"
        right_dir = root / "R"
        left_dir.mkdir()
        right_dir.mkdir()
        cv2.imwrite(str(left_dir / "b.png"), left_frames[1])
        cv2.imwrite(str(left_dir / "a.png"), left_frames[0])
        cv2.imwrite(str(right_dir / "a.png"), right_frames[0])
        cv2.imwrite(str(right_dir / "b.png"), right_frames[1])
        folders = FrameReader(_reader_args(left_dir=left_dir, right_dir=right_dir))
        names = [(int(round(float(left.mean()))), int(round(float(right.mean())))) for _, left, right in folders]
        folders.close()
        if names != [(0, 100), (1, 101)]:
            raise SystemExit(f"Папки кадров сопоставились не по имени: {names}")


def self_check() -> None:
    """Проверка Q, паддинга и чтения кадров без сети и без весов."""
    disparity = np.zeros((4, 6), dtype=np.float32)
    disparity[2, 3] = 10.0
    points = disparity_to_points(disparity, DEFAULT_Q)
    z = float(points[2, 3, 2])
    expected = 1164.6238115833075 / (1.9625989469856626 * 10.0)
    if abs(z - expected) > 1e-3:
        raise SystemExit(f"Ожидалась глубина {expected:.4f}, получилось {z:.4f}")
    padded, height, width = pad_to_multiple(np.zeros((10, 20, 3), dtype=np.uint8))
    if padded.shape[:2] != (32, 32) or (height, width) != (10, 20):
        raise SystemExit(f"Паддинг сломался: {padded.shape}")
    if semantic_group(4, "cityscapes") != "person" or semantic_group(6, "cityscapes") != "other":
        raise SystemExit("Группы классов не совпали с Rust")
    if semantic_group(1, "coco") != "person" or semantic_group(8, "coco") != "car":
        raise SystemExit("Группы COCO не совпали")
    _check_reader()
    print(f"self-check ок, Z при диспаратности 10 px = {z:.2f} м")


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride должен быть >= 1")
    if args.self_check:
        self_check()
        return

    if not args.seg.exists():
        raise SystemExit(
            f"Нет весов сегментации: {args.seg}\n"
            f"Скачайте {SEG_URL} в этот путь или передайте --seg."
        )

    device = pick_device(args.device)
    print("Устройство:", device)
    segmenter = SegmentationModel(args.seg, device, args.classes)

    reader = FrameReader(args)
    depth_model = None
    writer_vis = None
    writer_depth = None
    jsonl = None
    window = "slam-segm"
    processed = 0

    try:
        for index, left_bgr, right_bgr in reader:
            stereo = right_bgr is not None and not args.no_depth
            if stereo and depth_model is None:
                if not args.depth.exists():
                    raise SystemExit(
                        f"Стереокадр есть, а весов глубины нет: {args.depth}\n"
                        "Нужен чекпоинт BANet .pth или флаг --no-depth."
                    )
                depth_model = DepthModel(args.depth, device)

            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            detections = segmenter(left_rgb, args.conf, args.nms, args.max_det)
            disparity = None
            objects = None
            if stereo:
                right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
                disparity = depth_model(left_rgb, right_rgb)
                points = disparity_to_points(disparity, DEFAULT_Q)
                objects = objects_from_points(points, detections)

            vis = draw_segmentation(left_bgr, detections)
            depth_bgr = colorize_disparity(disparity) if disparity is not None else None
            if depth_bgr is not None and depth_bgr.shape[:2] != vis.shape[:2]:
                depth_bgr = cv2.resize(depth_bgr, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            preview = np.hstack([vis, depth_bgr]) if depth_bgr is not None else vis

            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                if writer_vis is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = reader.fps if reader.fps > 1 else 30.0
                    writer_vis = cv2.VideoWriter(str(args.out / "vis.mp4"), fourcc, fps, (vis.shape[1], vis.shape[0]))
                    if depth_bgr is not None:
                        writer_depth = cv2.VideoWriter(
                            str(args.out / "depth.mp4"), fourcc, fps, (depth_bgr.shape[1], depth_bgr.shape[0])
                        )
                    jsonl = open(args.out / "detections.jsonl", "w", encoding="utf-8")
                writer_vis.write(vis)
                if writer_depth is not None and depth_bgr is not None:
                    writer_depth.write(depth_bgr)
                record = detections_to_json(detections, objects)
                record["frame"] = index
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.out and processed == 0:
                cv2.imwrite(str(args.out / "preview.png"), preview)

            processed += 1
            groups = {}
            for det in detections:
                groups[det["group"]] = groups.get(det["group"], 0) + 1
            if processed == 1 or processed % 30 == 0:
                extra = ""
                if objects:
                    extra = " объекты=" + str(len(objects))
                if disparity is not None:
                    valid = disparity[np.isfinite(disparity) & (disparity > 0)]
                    if valid.size:
                        med = float(np.median(valid))
                        depth_m = float(DEFAULT_Q[2, 3] / (DEFAULT_Q[3, 2] * med))
                        extra += f" диспаратность {med:.1f} px, Z {depth_m:.1f} м"
                print(f"кадр {index}: {groups or 'пусто'}{extra}")

            if args.show:
                cv2.imshow(window, fit_preview(preview))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Остановлено с клавиатуры")
                    break
    finally:
        reader.close()
        if writer_vis is not None:
            writer_vis.release()
        if writer_depth is not None:
            writer_depth.release()
        if jsonl is not None:
            jsonl.close()
        if args.show:
            cv2.destroyAllWindows()

    if processed == 0:
        raise SystemExit("Ни одного кадра не прочитано")
    print(f"Готово, кадров: {processed}")
    if args.out:
        print(f"Записано в {args.out}")


if __name__ == "__main__":
    main()
