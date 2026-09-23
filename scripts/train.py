import argparse
from html import parser
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import optuna
from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model


LABEL_MERGE = {
    "OCEAN": "OCEAN",
    "LAND": "LAND",
    "OBSCR": "OBSCR",
    "SHALLOW": "SHALLOW",
    "DEEP": "DEEP",
    "HCRF": "HCRF",
    "SBF": "SBF",
    "OUTFLOW": "OUTFLOW",
}


LABEL_HIERARCHY = {
    "OCEAN": 0,
    "LAND": 1,
    "OBSCR": 2,
    "SHALLOW": 3,
    "HCRF": 4,
    "SBF": 5,
    "OUTFLOW": 6,
    "DEEP": 7,
}


VIS_LABEL_COLOR_MAP = {
    "OCEAN": (0, 100, 200),
    "LAND": (34, 139, 34),
    "OBSCR": (64, 64, 64),
    "SHALLOW": (128, 128, 128),
    "HCRF": (144, 238, 144),
    "DEEP": (255, 0, 0),
    "SBF": (128, 0, 128),
    "OUTFLOW": (255, 255, 0),
}

# LABEL_MERGE = {
#     "OCEAN": "OCEAN",
#     "LAND": "LAND",
#     "OBSCR": "CLOUD",
#     "SHALLOW": "CLOUD",
#     "DEEP": "DEEP",
#     "HCRF": "CLOUD",
#     "SBF": "SBF",
#     "OUTFLOW": "OUTFLOW",
# }

# LABEL_HIERARCHY = {
#     "OCEAN": 0,
#     "LAND": 1,
#     "CLOUD": 2,
#     "SBF": 3,
#     "OUTFLOW": 4,
#     "DEEP": 5,
# }

# VIS_LABEL_COLOR_MAP = {
#     "OCEAN": (0, 100, 200),
#     "LAND": (34, 139, 34),
#     "CLOUD": (64, 64, 64),
#     "DEEP": (255, 0, 0),
#     "SBF": (128, 0, 128),
#     "OUTFLOW": (255, 255, 0),
# }


# IGNORE_INDEX is used in LabelMe masks to mark pixels that should be ignored
# by the loss and metrics (e.g. unlabeled or outside region-of-interest).
IGNORE_INDEX = 255
FILE_RE = re.compile(r"^(\d{8})_(\d{4})_(\d{4})_(sat|rad)$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRecord:
    pair_dt: datetime
    day: str
    sat_path: Optional[Path]
    rad_path: Optional[Path]


@dataclass(frozen=True)
class TemporalSample:
    center_dt: datetime
    day: str
    sat_paths: Tuple[Path, ...]
    rad_paths: Tuple[Path, ...]
    label_json: Path


# ------------------------- U-Net -------------------------
class DoubleConv(nn.Module):
    """Two 3x3 padded convolutions, each followed by ReLU + BatchNorm."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net with padded convs, BatchNorm, and 1x1 skip connections."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64):
        super().__init__()
        c1 = base_channels          # 64
        c2 = base_channels * 2      # 128
        c3 = base_channels * 4      # 256
        c4 = base_channels * 8      # 512
        c5 = base_channels * 16     # 1024

        # Contracting path
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path: 2x2 up-conv halves the number of channels
        self.up4  = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)

        self.up3  = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)

        self.up2  = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1  = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        # Final 1x1 conv -> per-pixel class logits
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting path
        e1 = self.enc1(x)                       # 64  x H     x W
        e2 = self.enc2(self.pool(e1))           # 128 x H/2   x W/2
        e3 = self.enc3(self.pool(e2))           # 256 x H/4   x W/4
        e4 = self.enc4(self.pool(e3))           # 512 x H/8   x W/8
        b  = self.bottleneck(self.pool(e4))     # 1024 x H/16 x W/16

        # Expansive path with plain concatenation skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.out_conv(d1)   # (B, num_classes, H, W)
    

# ------------------------- DINOv2 -------------------------
class Dinov2SegmentationModel(nn.Module):
    """
    Multi-channel Sat/Rad temporal input -> adapter -> DINOv2 backbone -> segmentation logits.
    """

    # automatically puts the images through the CNN so its ready for dinov2 (6 channels -> 3 channels)
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone_name: str = "facebook/dinov2-base",
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size # int value: 768 for dinov2-base (default)
        patch = self.backbone.config.patch_size # 14 for dinov2
        self.patch_size = int(patch)

        # Adapter lets us keep your exact input format (6 * temporal window channels)
        # while still leveraging pretrained DINOv2 weights.
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(), # activation function
            nn.Conv2d(64, 3, kernel_size=1, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        ph = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pw = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if ph == 0 and pw == 0:
            return x, 0, 0
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, ph, pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.shape[-2], x.shape[-1]

        x = self.input_adapter(x)
        x, ph, pw = self._pad_to_patch_multiple(x)

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state

        n_tokens = tokens.shape[1]
        hp = x.shape[-2] // self.patch_size
        wp = x.shape[-1] // self.patch_size
        expected = hp * wp

        if n_tokens == expected + 1:
            tokens = tokens[:, 1:, :]
        elif n_tokens != expected:
            raise RuntimeError(
                f"Unexpected token count. got={n_tokens}, expected={expected} or {expected + 1}"
            )

        feat = tokens.transpose(1, 2).reshape(x.shape[0], -1, hp, wp)
        logits_lowres = self.decoder(feat)

        # upsample to padded image, then crop to original size
        logits = F.interpolate(
            logits_lowres,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear", # also tried bicubic but just takes longer and does not improve visual performance
            align_corners=False,
        )
        if ph > 0 or pw > 0:
            logits = logits[:, :, : h_in, : w_in]

        return logits

# --------------- Loss Function ----------------

class JaccardLoss(nn.Module):
    """
    Multi-class soft Jaccard (IoU) loss.
    Computes 1 - mean_IoU over classes, using softmax probabilities.
    Supports `ignore_index` and optional per-class `weight`.
    """
    def __init__(self, num_classes: int, ignore_index: int = -100,
                 weight: torch.Tensor = None, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer(
            "weight",
            weight if weight is not None else torch.ones(num_classes)
        )
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); target: (N, H, W)
        probs = F.softmax(logits, dim=1)

        # Build a valid-pixel mask to handle ignore_index
        valid = (target != self.ignore_index)                       # (N, H, W)
        target_clamped = target.clone()
        target_clamped[~valid] = 0                                   # safe for one_hot

        target_1h = F.one_hot(target_clamped, num_classes=self.num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).float()            # (N, C, H, W)

        # Zero-out ignored pixels in both prediction and target
        mask = valid.unsqueeze(1).float()
        probs      = probs      * mask
        target_1h  = target_1h  * mask

        dims = (0, 2, 3)  # sum over batch + spatial
        intersection = torch.sum(probs * target_1h, dim=dims)        # (C,)
        cardinality  = torch.sum(probs + target_1h, dim=dims)        # (C,)
        union        = cardinality - intersection

        iou_per_class = (intersection + self.smooth) / (union + self.smooth)

        # Only average over classes that actually appear in this batch's GT.
        # Absent classes have union >> smooth, so their IoU collapses to ~0 and
        # adds a constant n_absent/C offset that both floors the loss and
        # scales the gradient down by n_present/C.
        present = (target_1h.sum(dim=dims) > 0).float()             # (C,)

        w = self.weight.to(iou_per_class.device) * present
        total = w.sum()
        if total <= 0:                     # batch is entirely ignore_index
            return probs.sum() * 0.0       # zero loss, keeps the graph intact
        return 1.0 - (iou_per_class * w).sum() / total
    
class CEJaccardLoss(nn.Module):
    def __init__(self, num_classes, ignore_index, weight=None, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.jac = JaccardLoss(num_classes, ignore_index, weight)
        self.alpha = alpha

    def forward(self, logits, target):
        return self.alpha * self.ce(logits, target) + (1 - self.alpha) * self.jac(logits, target)

    
# ------------------------- Utilities -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pair_dt_from_stem(stem: str) -> Optional[Tuple[datetime, str]]:
    match = FILE_RE.match(stem)
    if not match:
        return None
    date_tok, pair_tok, _, _ = match.groups()
    dt_obj = datetime.strptime(f"{date_tok}_{pair_tok}", "%Y%m%d_%H%M")
    return dt_obj, date_tok


def parse_pair_dt_from_json(json_path: Path) -> Optional[Tuple[datetime, str]]:
    stem_match = parse_pair_dt_from_stem(json_path.stem)
    if stem_match is not None:
        return stem_match
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        image_path = payload.get("imagePath", "")
        image_stem = Path(image_path).stem
        return parse_pair_dt_from_stem(image_stem)
    except Exception:
        return None


def discover_frame_records(image_root: Path) -> Dict[datetime, FrameRecord]:
    """Discover available sat/rad image pairs under `image_root`.

    Returns a mapping from datetime -> `FrameRecord` with optional sat/rad
    filepaths (may be None if missing).
    """
    grouped: Dict[datetime, Dict[str, object]] = defaultdict(
        lambda: {"sat": None, "rad": None, "day": None}
    )
    for file_path in image_root.rglob("*.png"):
        parsed = parse_pair_dt_from_stem(file_path.stem)
        if parsed is None:
            continue
        pair_dt, day = parsed
        modality = file_path.stem.split("_")[-1].lower()
        if modality not in {"sat", "rad"}:
            continue
        slot = grouped[pair_dt]
        if slot[modality] is None:
            slot[modality] = file_path
            slot["day"] = day

    records = {}
    for pair_dt, slot in grouped.items():
        records[pair_dt] = FrameRecord(
            pair_dt=pair_dt,
            day=slot["day"],
            sat_path=slot["sat"],
            rad_path=slot["rad"],
        )
    return records


def discover_labels(label_roots) -> Dict[datetime, Path]:
    """Discover LabelMe JSON label files across one or more roots."""
    if isinstance(label_roots, Path):
        label_roots = [label_roots]

    labels: Dict[datetime, Path] = {}
    for label_root in label_roots:
        for json_path in label_root.rglob("*.json"):
            parsed = parse_pair_dt_from_json(json_path)
            if parsed is None:
                continue
            pair_dt, _ = parsed
            if pair_dt not in labels:
                labels[pair_dt] = json_path
    return labels


def split_samples_random(samples: Sequence[TemporalSample], train_ratio: float, val_ratio: float, seed: int):
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    if n < 3:
        raise ValueError("Need at least 3 samples for train/val/test split")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_samples = items[:n_train]
    val_samples = items[n_train : n_train + n_val]
    test_samples = items[n_train + n_val :]
    if not test_samples:
        test_samples = [items[-1]]
        if items[-1] in val_samples and len(val_samples) > 1:
            val_samples = val_samples[:-1]

    return train_samples, val_samples, test_samples


def build_temporal_samples(frame_records: Dict[datetime, FrameRecord], labels: Dict[datetime, Path], context: int):
    """Construct temporal windows (samples) centered on labeled frames.

    The `context` defines how many frames before/after the center to include
    (window length = 2*context + 1).
    """
    stats = Counter()
    by_day: Dict[str, List[datetime]] = defaultdict(list)
    for pair_dt, rec in frame_records.items():
        if rec.sat_path is not None and rec.rad_path is not None:
            by_day[rec.day].append(pair_dt)
        else:
            stats["dropped_missing_modality"] += 1
            print(f"Warning: dropping {pair_dt} due to missing modality: sat={rec.sat_path is not None}, rad={rec.rad_path is not None}")

    for day in by_day:
        by_day[day] = sorted(by_day[day])

    samples: List[TemporalSample] = []
    for center_dt, label_json in labels.items():
        if center_dt not in frame_records:
            stats["dropped_label_no_frame"] += 1
            continue
        rec = frame_records[center_dt]
        day = rec.day
        day_times = by_day.get(day, [])
        if center_dt not in day_times:
            stats["dropped_center_missing_pair"] += 1
            continue

        idx = day_times.index(center_dt)
        if idx - context < 0 or idx + context >= len(day_times):
            stats["dropped_window_edge"] += 1
            continue

        window = day_times[idx - context : idx + context + 1]
        sat_paths: List[Path] = []
        rad_paths: List[Path] = []
        complete = True
        for dt_obj in window:
            wrec = frame_records[dt_obj]
            if wrec.sat_path is None or wrec.rad_path is None:
                complete = False
                break
            sat_paths.append(wrec.sat_path)
            rad_paths.append(wrec.rad_path)

        if not complete:
            stats["dropped_window_missing_modality"] += 1
            continue

        samples.append(
            TemporalSample(
                center_dt=center_dt,
                day=day,
                sat_paths=tuple(sat_paths),
                rad_paths=tuple(rad_paths),
                label_json=label_json,
            )
        )
        stats["kept"] += 1

    return samples, dict(stats)


# ------------------------- Label and image utilities -------------------------
def draw_shape_mask(width: int, height: int, shape: dict) -> np.ndarray:
    shape_type = str(shape.get("shape_type", "")).lower()
    points = shape.get("points", [])
    tmp = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(tmp)

    if shape_type == "polygon" and len(points) >= 3:
        draw.polygon([tuple(p) for p in points], fill=255)
    elif shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        draw.rectangle([(x1, y1), (x2, y2)], fill=255)
    else:
        return np.zeros((height, width), dtype=np.uint8)

    return np.asarray(tmp, dtype=np.uint8)


def labelme_json_to_mask(json_path: Path, width: int, height: int, class_map: Dict[str, int], class_priority: Dict[str, int])-> np.ndarray:
    # Read LabelMe JSON and rasterize shapes into a single class mask.
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    priority_mask = np.full((height, width), -1, dtype=np.int32)
    class_mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)

    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).upper()
        label = LABEL_MERGE.get(label, label)
        if label not in class_map:
            continue
        pri = class_priority.get(label, -1)
        raster = draw_shape_mask(width, height, shape)
        hit = raster > 0
        update = hit & (priority_mask < pri)
        if np.any(update):
            priority_mask[update] = pri
            class_mask[update] = class_map[label]

    return class_mask


def read_rgb_tensor(path: Path, width: int, height: int, rot_deg: float = 0.0, hflip: bool = False, vflip: bool = False) -> torch.Tensor:
    """Read an RGB PNG into a normalized torch Tensor and optionally apply
    small rotation and flips. Rotations are applied using PIL and keep the same
    image dimensions (expand=False).

    Note: We apply the same augmentation parameters to all channels in a
    temporal window externally in the dataset so that input frames stay aligned.
    """
    with Image.open(path) as im:
        if rot_deg:
            # rotate around center; keep same size (expand=False)
            # small rotations use bilinear resampling to reduce aliasing
            im = im.rotate(rot_deg, resample=Image.BILINEAR, expand=False)
        if hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)

    if arr.shape[0] != height or arr.shape[1] != width:
        raise ValueError(f"Image shape mismatch for {path}: got {arr.shape[:2]}, expected {(height, width)}")
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def build_palette_from_class_map(class_map: Dict[str, int])-> np.ndarray:
    num_classes = len(class_map)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for name, idx in class_map.items():
        if name in VIS_LABEL_COLOR_MAP:
            palette[idx] = np.array(VIS_LABEL_COLOR_MAP[name], dtype=np.uint8)
        else:
            palette[idx] = np.array([255, 0, 0], dtype=np.uint8)
    return palette


def blend_mask_over_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    valid = mask != IGNORE_INDEX
    for cid in range(palette.shape[0]):
        hit = (mask == cid) & valid
        if np.any(hit):
            color = palette[cid].astype(np.float32)
            out[hit] = (1.0 - alpha) * out[hit] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_timestamp_and_legend(
    panel_rgb: np.ndarray,
    timestamp_text: str,
    present_labels: List[str],
    class_map: Dict[str, int],
    palette: np.ndarray,
    legend_panel_idx: int = 2,
    panel_count: int = 4,
    legend_row_idx=0,
    row_count=1,
) -> np.ndarray:
    img = Image.fromarray(panel_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    bbox = (
        draw.textbbox((0, 0), timestamp_text)
        if hasattr(draw, "textbbox")
        else (0, 0, *draw.textsize(timestamp_text))
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    draw.rectangle(
        [(5, 5), (5 + text_w + 2 * pad, 5 + text_h + 2 * pad)],
        fill=(255, 255, 255, 120),
        outline=(0, 0, 0, 180),
        width=1,
    )
    draw.text((5 + pad, 5 + pad), timestamp_text, fill=(0, 0, 0, 220))

    if present_labels:
        w, h = img.size
        tile_w = w // panel_count
        tile_h = h // row_count
        x0 = legend_panel_idx * tile_w
        y0 = legend_row_idx * tile_h

        box_size = 12
        padding = 4
        line_height = box_size + 3
        legend_width = 95
        legend_height = len(present_labels) * line_height + 2 * padding

        legend_x = x0 + tile_w - legend_width - 5
        legend_y = y0 + tile_h - legend_height - 5

        draw.rectangle(
            [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)],
            fill=(255, 255, 255, 120),
            outline=(0, 0, 0, 180),
            width=1,
        )

        for i, label in enumerate(present_labels):
            y_pos = legend_y + padding + i * line_height
            cid = class_map[label]
            color = tuple(int(v) for v in palette[cid])

            draw.rectangle(
                [
                    (legend_x + padding, y_pos),
                    (legend_x + padding + box_size, y_pos + box_size),
                ],
                fill=color + (255,),
                outline=(0, 0, 0, 180),
                width=1,
            )
            draw.text(
                (legend_x + padding + box_size + 5, y_pos + 2),
                label,
                fill=(0, 0, 0, 220),
            )

    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_dinov2_input_view(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    context: int,
    out_dir: Path,
    max_samples: int = 8,
) -> int:
    """
    Visualize the actual tensor DINOv2 receives.
    
    Strategy: per-sample standardization (z-score) using the tensor's own
    mean/std, then map to [0,1] via a fixed sigmoid-like squash. This:
      - preserves relative magnitudes between channels (critical)
      - is stable across samples (comparable visualizations)
      - handles outliers gracefully (no single bright pixel washing out the rest)
      - doesn't depend on arbitrary min/max which can drift
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    captured = {}
    handle = model.input_adapter.register_forward_hook(
        lambda m, i, o: captured.update(adapter_out=o.detach())
    )

    saved = 0
    try:
        for x, y, days, centers in loader:
            x_dev = x.to(device, non_blocking=True)
            _ = model(x_dev)
            feat = captured["adapter_out"].cpu().numpy()  # [B, 3, H, W]
            x_cpu = x.cpu().numpy()

            for b in range(feat.shape[0]):
                if saved >= max_samples:
                    return saved

                f = feat[b]  # [3, H, W]

                # 1. Standardize using GLOBAL stats (across all 3 channels together)
                #    This preserves cross-channel magnitude relationships.
                mu = f.mean()
                sigma = f.std() + 1e-8
                f_z = (f - mu) / sigma

                # 2. Squash to [0,1] via tanh (smooth, bounded, symmetric)
                #    tanh maps ~[-3, 3] sigma into roughly [-1, 1], then we shift to [0,1]
                f_squash = (np.tanh(f_z / 2.0) + 1.0) / 2.0
                rgb = (np.transpose(f_squash, (1, 2, 0)) * 255).astype(np.uint8)

                center = centers[b]
                day = days[b]
                Image.fromarray(rgb).save(
                    out_dir / f"dino_input_{saved:04d}_{day}_{center}.png"
                )
                saved += 1
    finally:
        handle.remove()
    return saved


@torch.no_grad()
def save_prediction_visuals(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_map: Dict[str, int],
    context: int,
    out_dir: Path,
    max_samples: int = 24,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    num_classes = len(class_map)
    palette = build_palette_from_class_map(class_map)
    inv_class = {v: k for k, v in class_map.items()}

    with (out_dir / "palette.json").open("w", encoding="utf-8") as f:
        json.dump(
            {inv_class[i]: palette[i].tolist() for i in range(num_classes)}, f, indent=2
        )

    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Panels (left->right): center SAT RGB | center RAD RGB | center GT overlay | prediction overlay | error map | confidence map | GT probability map | graded error map\n")
        f.write("Overlay blend: 0.6*image + 0.4*label color.\n")

    saved = 0
    for x, y, days, centers in loader:
        x_dev = x.to(device, non_blocking=True)
        logits = model(x_dev) 
        probs_t  = F.softmax(logits, dim=1)    # convert logits to probabilities using softmax
        pred_t   = torch.argmax(logits, dim=1)      # pick the highest prob one
        conf_t, _ = probs_t.max(dim=1)              # the confidence of the highest prob one

        # GT-probability: probability the model assigned to the correct class
        y_dev = y.to(device, non_blocking=True)
        gt_safe = y_dev.clone()
        gt_safe[gt_safe == IGNORE_INDEX] = 0  # avoid out-of-range gather
        gt_prob_t = probs_t.gather(1, gt_safe.unsqueeze(1)).squeeze(1)

        # convert these from tensors to numpy arrays for visualization
        pred = pred_t.cpu().numpy()  # [B, H, W], each value is the predicted class index
        probs = probs_t.cpu().numpy()  # [B, num_classes, H, W], probability distribution over classes for each pixel
        conf = conf_t.cpu().numpy()  # [B, H, W], each value is the confidence of the predicted class
        gt_prob = gt_prob_t.cpu().numpy()

        # pred = torch.argmax(logits, dim=1).cpu().numpy() ###### picks the highest logit class as the prediction for each pixel

        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()

        for b in range(x_cpu.shape[0]):
            if saved >= max_samples:
                return saved
    
            # SATELLITE PANEL
            center_sat_start = context * 6
            center_sat = x_cpu[b, center_sat_start : center_sat_start + 3]
            center_sat = np.transpose(center_sat, (1, 2, 0))
            center_sat = np.clip(center_sat, 0.0, 1.0)
            center_sat_rgb = (center_sat * 255.0).astype(np.uint8)

            # RADAR PANEL
            center_rad_start = context * 6 + 3 # use the next (after satellite) 3 channels for the radar
            center_rad = x_cpu[b, center_rad_start : center_rad_start + 3]
            center_rad = np.transpose(center_rad, (1, 2, 0))
            center_rad = np.clip(center_rad, 0.0, 1.0)
            center_rad_rgb = (center_rad * 255.0).astype(np.uint8)

            # GROUND TRUTH PANEL
            gt = y_cpu[b].astype(np.int64)
            gt_vis = blend_mask_over_image(center_sat_rgb, gt, palette, alpha=0.4)

            # PREDICTION PANEL
            pd = pred[b].astype(np.int64)
            pd_vis = blend_mask_over_image(center_sat_rgb, pd, palette, alpha=0.4)

            valid = gt != IGNORE_INDEX
            err = np.zeros_like(center_sat_rgb, dtype=np.uint8)
            err[(pd == gt) & valid] = np.array([0, 255, 0], dtype=np.uint8)
            err[(pd != gt) & valid] = np.array([255, 0, 0], dtype=np.uint8)

            # CONFIDENCE PANEL
            conf_b = conf[b]                              # [H, W], in [0,1]
            conf_vis = np.stack([(conf_b*255).astype(np.uint8)]*3, axis=-1)

            # GT-PROB PANEL
            gt_prob_b = gt_prob[b].copy()
            gt_prob_b[~valid] = 0.0
            gtp_vis = (cm.viridis(gt_prob_b)[..., :3] * 255).astype(np.uint8)
            gtp_vis[~valid] = 0

            # GRADED ERROR PANEL
            graded = np.zeros_like(center_sat_rgb, dtype=np.uint8)
            correct = (pd == gt) & valid
            wrong   = (pd != gt) & valid
            # if correct then its green, with brightness corresponding to confidence
            graded[correct, 1] = (255 * conf_b[correct]).astype(np.uint8)
            # if wrong then its red, with brightness corresponding to GT-prob
            graded[wrong, 0] = 255
            graded[wrong, 1] = (255 * gt_prob_b[wrong]).astype(np.uint8)

            # map all the panels together into one figure
            top_row = np.concatenate([center_sat_rgb, center_rad_rgb, gt_vis, pd_vis], axis=1)
            # bottom_row = np.concatenate([err, conf_vis, gtp_vis, graded], axis=1)

            # make it two rows of panels
            # panel = np.concatenate([top_row, bottom_row], axis=0)
            panel = top_row
            

            present_ids = set(np.unique(gt[gt != IGNORE_INDEX]).tolist()) | set(np.unique(pd).tolist())
            present_labels = [name for name, idx in class_map.items() if idx in present_ids]
            present_labels = sorted(present_labels, key=lambda n: LABEL_HIERARCHY.get(n, 999))

            center = centers[b]
            toks = center.split("_")
            timestamp_text = f"{toks[0]} @ {toks[1]}z" if len(toks) >= 2 else center

            panel = draw_timestamp_and_legend(
                panel, timestamp_text, present_labels, class_map, palette, legend_panel_idx=3, panel_count=4, legend_row_idx=0, row_count=1,
            )

            day = days[b]
            Image.fromarray(panel).save(out_dir / f"{saved:04d}_{day}_{center}.png")
            saved += 1 # for the file name to change

    return saved


# ------------------------- Dataset -------------------------
class TemporalSatRadDataset(Dataset):
    """Dataset that returns temporal stacks of SAT/RAD frames plus the rasterized
    LabelMe mask. Augmentations include horizontal/vertical flips,
    small-angle rotations, and elastic deformations.
    """
    def __init__(
        self,
        samples: Sequence[TemporalSample],
        width: int,
        height: int,
        class_map: Dict[str, int],
        class_priority: Dict[str, int],
        aug_hflip: bool = False,
        aug_vflip: bool = False,
        aug_rotdeg: float = 0.0,
        aug_elastic: bool = False,
        aug_elastic_alpha: float = 10.0,
        aug_elastic_grid: int = 3,
    ):
        self.samples = list(samples)
        self.width = width
        self.height = height
        self.class_map = class_map
        self.class_priority = class_priority

        self.aug_hflip = aug_hflip
        self.aug_vflip = aug_vflip
        # aug_rotdeg is the max absolute degree to randomly rotate input +/- this value
        self.aug_rotdeg = float(aug_rotdeg)
        self.aug_elastic = bool(aug_elastic)
        self.aug_elastic_alpha = float(aug_elastic_alpha)
        self.aug_elastic_grid = int(aug_elastic_grid)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        # Sample global augmentation parameters that must be applied consistently
        # across all frames in the temporal window so the stack remains aligned.
        do_hflip = self.aug_hflip and (random.random() < 0.5)
        do_vflip = self.aug_vflip and (random.random() < 0.5)
        rot_k = 0

        # small random rotation in degrees [-aug_rotdeg, +aug_rotdeg]
        rot_deg = 0.0
        if self.aug_rotdeg:
            rot_deg = random.uniform(-self.aug_rotdeg, self.aug_rotdeg)

        # Load all channels in the window applying the same augmentation
        chan_tensors: List[torch.Tensor] = []
        for sat_p, rad_p in zip(sample.sat_paths, sample.rad_paths):
            sat = read_rgb_tensor(sat_p, self.width, self.height, rot_deg=rot_deg, hflip=do_hflip, vflip=do_vflip)
            rad = read_rgb_tensor(rad_p, self.width, self.height, rot_deg=rot_deg, hflip=do_hflip, vflip=do_vflip)
            chan_tensors.append(sat)
            chan_tensors.append(rad)

        x = torch.cat(chan_tensors, dim=0)
        y = labelme_json_to_mask(sample.label_json, width=self.width, height=self.height,
                                class_map=self.class_map, class_priority=self.class_priority)
        y_t = torch.from_numpy(y.astype(np.int64))

        # --- elastic deformation (Ronneberger et al. 2015) ---
        if not getattr(self, "skip_augmentation", False):
            if getattr(self, "aug_elastic", False):
                rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
                x, y_t = elastic_deform(
                    x, y_t,
                    alpha=self.aug_elastic_alpha,
                    grid_size=self.aug_elastic_grid,
                    ignore_index=255,
                    rng=rng,
                )

        return x, y_t, sample.day, sample.center_dt.strftime("%Y%m%d_%H%M")

class AugmentedRepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, repeat: int = 2, global_seed: int = 0):
        self.base = base_dataset
        self.repeat = int(repeat)
        self.global_seed = int(global_seed)

    def __len__(self):
        return len(self.base) * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % len(self.base)
        repeat_idx = idx // len(self.base)  # 0, 1, 2, ..., repeat-1

        # Repeat 0 = clean original, no augmentation
        if repeat_idx == 0:
            self.base.skip_augmentation = True
        else:
            self.base.skip_augmentation = False
            # Seed differently per (idx) for reproducibility + diversity
            py_state = random.getstate()
            np_state = np.random.get_state()
            seed = hash((self.global_seed, idx)) & 0xFFFFFFFF
            random.seed(seed)
            np.random.seed(seed)

        item = self.base[base_idx]

        if repeat_idx != 0:
            random.setstate(py_state)
            np.random.set_state(np_state)

        self.base.skip_augmentation = False  # reset
        return item
    

def elastic_deform(
    image: torch.Tensor,
    label: torch.Tensor,
    alpha: float = 10.0,
    grid_size: int = 3,
    ignore_index: int = 255,
    rng: np.random.Generator | None = None,
):
    """Apply random elastic deformation per Ronneberger et al. (2015).

    Random displacement vectors on a coarse `grid_size` x `grid_size` grid,
    sampled from N(0, alpha^2) in pixels, then bicubic-interpolated to
    per-pixel displacements [3].

    Args:
        image: (C, H, W) float tensor in [0, 1] (multi-channel stack OK).
        label: (H, W) long tensor of class indices (with `ignore_index` for
            unlabeled pixels).
        alpha: Std deviation (in pixels) of Gaussian displacement samples.
            Ronneberger et al. (2015) use 10 pixels [3].
        grid_size: Coarse grid resolution for displacement sampling.
            Ronneberger et al. (2015) use a 3x3 grid [3].
        ignore_index: Label value reserved for ignored/unlabeled pixels;
            preserved after warping via nearest-neighbor interpolation.
        rng: Optional numpy Generator for reproducibility. If None, uses
            the global numpy RNG.

    Returns:
        (warped_image, warped_label) with the same shapes/dtypes as inputs.
    """
    if rng is None:
        rng = np.random

    H, W = image.shape[-2:]

    # Sample coarse displacements (grid_size x grid_size), std = alpha pixels [3]
    dx = (rng.standard_normal((grid_size, grid_size)) * alpha).astype(np.float32)
    dy = (rng.standard_normal((grid_size, grid_size)) * alpha).astype(np.float32)

    # Bicubic-interpolate the coarse grid up to per-pixel displacements [3]
    dx_full = cv2.resize(dx, (W, H), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy, (W, H), interpolation=cv2.INTER_CUBIC)

    # Build the absolute sample-coordinate maps for cv2.remap.
    # cv2.remap reads SOURCE pixel at (map_x[y, x], map_y[y, x]) to produce
    # the OUTPUT at (y, x). So we add displacements to the identity grid.
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    map_x = (grid_x + dx_full).astype(np.float32)
    map_y = (grid_y + dy_full).astype(np.float32)

    # ---- Warp the image (per-channel) ----
    # image is (C, H, W) torch tensor; convert to numpy for cv2.
    image_np = image.detach().cpu().numpy()  # (C, H, W), float32
    C = image_np.shape[0]
    warped_image_np = np.empty_like(image_np)
    for c in range(C):
        warped_image_np[c] = cv2.remap(
            image_np[c],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,  # bilinear is fine for image data
            borderMode=cv2.BORDER_REFLECT_101,  # mirror padding, akin to the
                                                # overlap-tile strategy [3]
        )
    warped_image = torch.from_numpy(warped_image_np).to(image.dtype)

    # ---- Warp the label (nearest-neighbor to preserve class indices) ----
    # Use INTER_NEAREST so we never invent new class IDs at boundaries, and
    # so that `ignore_index` regions stay exactly equal to `ignore_index`.
    label_np = label.detach().cpu().numpy().astype(np.int32)
    warped_label_np = cv2.remap(
        label_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(ignore_index),  # anything sampled outside -> ignore
    )
    warped_label = torch.from_numpy(warped_label_np).to(label.dtype)

    return warped_image, warped_label


def split_days(
    all_days: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    days = sorted(set(all_days))
    rng = random.Random(seed)
    rng.shuffle(days)

    if len(days) < 3:
        raise ValueError("Need at least 3 unique days for day-based train/val/test split")

    n = len(days)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_days = set(days[:n_train])
    val_days = set(days[n_train : n_train + n_val])
    test_days = set(days[n_train + n_val :])
    if not test_days:
        test_days = {days[-1]}
        if days[-1] in val_days:
            val_days.remove(days[-1])
            val_days.add(days[-2])
    return train_days, val_days, test_days


def filter_by_days(samples: Sequence[TemporalSample], days: set) -> List[TemporalSample]:
    return [s for s in samples if s.day in days]


# -------------------- Cross Validation Helpers --------------------
def kfold_split_days(
    all_days,
    k: int,
    fold: int,
    val_ratio: float,
    seed: int,
):
    days = sorted(set(all_days))

    if k < 2:
        raise ValueError("cv-k must be at least 2.")
    if fold < 0 or fold >= k:
        raise ValueError(f"cv-fold must be in [0, {k-1}].")
    if len(days) < k:
        raise ValueError(f"Need at least {k} unique days for {k}-fold day CV.")

    rng = random.Random(seed)
    rng.shuffle(days)

    # Split days into k roughly equal folds
    folds = [days[i::k] for i in range(k)]

    test_days = set(folds[fold])
    remaining_days = [d for i, f in enumerate(folds) if i != fold for d in f]

    # Validation comes from the non-test days
    rng.shuffle(remaining_days)

    # If your original split was 70/15/15, then after holding out 20% for test,
    # validation should be about 15 / 85 of the remaining data.
    val_frac_of_remaining = val_ratio / (1.0 - 1.0 / k)

    n_val = max(1, int(round(len(remaining_days) * val_frac_of_remaining)))

    val_days = set(remaining_days[:n_val])
    train_days = set(remaining_days[n_val:])

    if len(train_days) == 0 or len(val_days) == 0 or len(test_days) == 0:
        raise RuntimeError("Empty train/val/test split in CV. Need more days or smaller k.")

    assert train_days.isdisjoint(val_days)
    assert train_days.isdisjoint(test_days)
    assert val_days.isdisjoint(test_days)

    return train_days, val_days, test_days


def kfold_split_samples(
    samples,
    k: int,
    fold: int,
    val_ratio: float,
    seed: int,
):
    xs = list(samples)

    if k < 2:
        raise ValueError("cv-k must be at least 2.")
    if fold < 0 or fold >= k:
        raise ValueError(f"cv-fold must be in [0, {k-1}].")
    if len(xs) < k:
        raise ValueError(f"Need at least {k} samples for {k}-fold CV.")

    rng = random.Random(seed)
    rng.shuffle(xs)

    folds = [xs[i::k] for i in range(k)]

    test_samples = list(folds[fold])
    remaining_samples = [s for i, f in enumerate(folds) if i != fold for s in f]

    rng.shuffle(remaining_samples)

    val_frac_of_remaining = val_ratio / (1.0 - 1.0 / k)
    n_val = max(1, int(round(len(remaining_samples) * val_frac_of_remaining)))

    val_samples = remaining_samples[:n_val]
    train_samples = remaining_samples[n_val:]

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Empty train/val/test split in CV.")

    return train_samples, val_samples, test_samples


def make_train_val_test_split(samples, args):
    if args.cv_k > 0:
        if args.split_type == "day":
            all_days = sorted({s.day for s in samples})

            train_days, val_days, test_days = kfold_split_days(
                all_days,
                k=args.cv_k,
                fold=args.cv_fold,
                val_ratio=args.val_ratio,
                seed=args.cv_seed,
            )

            train_samples = [s for s in samples if s.day in train_days]
            val_samples   = [s for s in samples if s.day in val_days]
            test_samples  = [s for s in samples if s.day in test_days]

        else:
            train_samples, val_samples, test_samples = kfold_split_samples(
                samples,
                k=args.cv_k,
                fold=args.cv_fold,
                val_ratio=args.val_ratio,
                seed=args.cv_seed,
            )

    else:
        if args.split_type == "day":
            all_days = sorted({s.day for s in samples})

            train_days, val_days, test_days = split_days(
                all_days,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )

            train_samples = [s for s in samples if s.day in train_days]
            val_samples   = [s for s in samples if s.day in val_days]
            test_samples  = [s for s in samples if s.day in test_days]

        else:
            rng = random.Random(args.seed)
            xs = list(samples)
            rng.shuffle(xs)

            n = len(xs)
            n_train = int(round(n * args.train_ratio))
            n_val   = int(round(n * args.val_ratio))

            train_samples = xs[:n_train]
            val_samples   = xs[n_train:n_train + n_val]
            test_samples  = xs[n_train + n_val:]

    return train_samples, val_samples, test_samples

# ------------------------- Metrics and losses -------------------------
def compute_class_weights(
    samples: Sequence[TemporalSample],
    width: int,
    height: int,
    class_map: Dict[str, int],
    scheme: str = "inverse",
) -> torch.Tensor:
    '''
    Compute class weights inversely proportional to pixel counts across the dataset.
    - For each sample, rasterize the label polygons to get a class mask.
    - Count the number of pixels for each class across all samples.
    - Compute weights as the inverse of these counts, normalized so that the average weight is 1.0.
    - This helps to mitigate class imbalance by giving more weight to underrepresented classes during training.
    - Note: We use the same rasterization logic as in the dataset to ensure consistency.
    '''
    counts = np.zeros(len(class_map), dtype=np.float64)
    for s in samples:
        m = labelme_json_to_mask(
            s.label_json,
            width=width,
            height=height,
            class_map=class_map,
            class_priority=LABEL_HIERARCHY,
        )
        for cid in range(len(class_map)):
            counts[cid] += np.sum(m == cid)

    counts = np.maximum(counts, 1.0)
    if scheme == "inverse":
        inv = 1.0 / counts
    elif scheme == "sqrt-inverse":
        inv = 1.0 / np.sqrt(counts)     # 4.8:1 instead of 23:1
    else:
        raise ValueError(f"unknown class weighting scheme: {scheme}")
    return torch.tensor(inv / inv.sum() * len(class_map), dtype=torch.float32)


# ------------------------- Training loop -------------------------
def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device, num_classes: int, progress_callback=None, epoch_index: Optional[int] = None, total_epochs: Optional[int] = None,) -> Tuple[float, float, np.ndarray]:
    """
    Run a single epoch of training or evaluation.

    If an optimizer is provided, the model runs in training mode (with backprop);
    otherwise it runs in evaluation mode (forward pass only).

    Returns:
        avg_loss: Average loss per sample over the epoch.
        pixel_acc: Overall pixel-wise accuracy (ignoring IGNORE_INDEX pixels).
        mean_iou: Per-class mean IoU averaged across batches.
    """
    # determine mode: training if optimizer exists, otherwise validation/test
    train_mode = optimizer is not None
    model.train(train_mode)

    # initialize the loss accumulator
    running_loss = 0.0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    # iterate through all batches in the data loader
    for x, y, _, _ in loader:
        # move images to gpu if available
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # reset gradient before each step
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        # forward pass, gives predictions and loss
        logits = model(x)
        loss = criterion(logits, y)

        # backward pass, only for training
        if train_mode:
            loss.backward()
            optimizer.step()

        # loss weighted by batch size
        running_loss += loss.item() * x.size(0)

        # compute metrics
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            valid = y != IGNORE_INDEX # mask out unlabeled pixels, which may be there from augmentation
            p = pred[valid]
            t = y[valid]
            # bincount trick to build confusion matrix
            idx = t * num_classes + p
            binc = torch.bincount(idx, minlength=num_classes**2)
            confusion += binc.reshape(num_classes, num_classes)

    # compute final metrics for the epoch
    avg_loss = running_loss / max(1, len(loader.dataset))
    confusion_np = confusion.cpu().numpy()
    tp = np.diag(confusion_np)
    fp = confusion_np.sum(axis=0) - tp
    fn = confusion_np.sum(axis=1) - tp
    denom = tp + fp + fn

    # A class is scored iff it has ground-truth support in this split (row sum > 0).
    # Keying on gt_support instead of `denom > 0` keeps the mIoU denominator fixed
    # for a given split. Otherwise a single spuriously predicted pixel of a class
    # that never occurs in the labels drags that class into the mean at IoU ~ 0,
    # which costs several mIoU points no matter how small the actual error is --
    # and lets the denominator drift between epochs, so `best_val_miou` would be
    # comparing means taken over different sets of classes.
    gt_support = confusion_np.sum(axis=1)
    iou = np.where(gt_support > 0, tp / np.maximum(denom, 1), np.nan)
    pixel_acc = tp.sum() / max(1, confusion_np.sum())

    return avg_loss, pixel_acc, iou

def save_manifest(samples: Sequence[TemporalSample], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "center": s.center_dt.strftime("%Y%m%d_%H%M"),
                "day": s.day,
                "sat_paths": [str(p) for p in s.sat_paths],
                "rad_paths": [str(p) for p in s.rad_paths],
                "label_json": str(s.label_json),
            }
            f.write(json.dumps(row) + "\n")


# ------------------------- Argument Parsing -------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal fused Sat/Rad multiclass model from LabelMe polygons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--model", type=str, required=True, choices=["unet", "dino"], help="Model architecture to use for segmentation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--image-root", type=Path, default="../images/", help="Root directory to search for input satellite/radar PNG files")
    parser.add_argument("--label-root", type=Path, action="append", default=[Path("../unsampled/dataset/labeled/"), Path("../sampled/dataset/labeled/")], help="Root directory(s) to search for label JSON files. Can specify multiple --label-root arguments to include multiple directories.")
    parser.add_argument("--output-dir", type=str, default="../../../../train_out/{model}/{date:%Y%m%d_%H%M%S}", help="Output directory for model checkpoints and logs")
    parser.add_argument("--height", type=int, default=448, help="Input height in pixels (images will be resized or cropped to this height)")
    parser.add_argument("--width", type=int, default=512, help="Input width in pixels (images will be resized or cropped to this width)")
    parser.add_argument("--context", type=int, default=0, help="Number of past/future frames to include on top of center frame (e.g. context=2 means input has 5 frames: t-2, t-1, t0, t+1, t+2)")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--early-stop-patience", type=int, default=4, help="Stop if val mIoU does not improve for N epochs. 0 disables.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--split-type", type=str, default="rand", choices=["day", "rand"], help="Type of split to use for train/val/test sets")
    parser.add_argument("--label-usage", choices=["all", "train_samp", "only_samp"],default="train_samp",help=("How to use labeled frames:\n""'all' = use sampled+unsampled together, split by day (no day leakage).\n" "  'train_samp'  = train on sampled/ only, val/test from unsampled/.\n" "  'only_samp' = use only sampled/ frames for train/val/test ."))
    parser.add_argument("--train-subset-size", type=int, default=None, help="If not None, limit the number of training samples (after splitting) to this size for faster experiments")
    parser.add_argument("--train-subset-seed", type=int, default=42, help="Random seed for selecting the training subset")
    parser.add_argument("--base-channels", type=int, default=64, help="Number of channels in the first layer of the U-Net")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for the optimizer (only unet)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Learning rate for the backbone network (only dinov2)")
    parser.add_argument("--head-lr", type=float, default=5e-4, help="Learning rate for the segmentation head (only dinov2)")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="Weight decay for the optimizer (only dinov2)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of samples to use for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of samples to use for validation")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum number of samples to use (0 means no limit)")
    parser.add_argument("--save-every", type=int, default=1, help="Save model checkpoint every N epochs")
    parser.add_argument("--aug-hflip", action="store_true", help="Apply horizontal flips")
    parser.add_argument("--aug-vflip", action="store_true", help="Apply vertical flips")
    parser.add_argument("--aug-rotdeg", type=float, default=0.0, help="Random rotation in degrees (e.g. 15 for ±15°)")
    parser.add_argument("--aug-elastic", action="store_true", help="Enable elastic deformation augmentation")
    parser.add_argument("--aug-elastic-alpha", type=float, default=10.0, help="Std-dev of Gaussian displacement in pixels (paper: 10)")
    parser.add_argument("--aug-elastic-grid", type=int, default=3, help="Coarse displacement grid size (paper: 3)")
    parser.add_argument("--aug-expand", type=int, default=1, help="Expand dataset by deterministic augmentations. Each augmentation is a combination of rotations and h/v flips. Setting to 1 means no expansion, 8 means all combinations.")
    parser.add_argument("--viz-samples", type=int, default=24, help="Number of samples to save visualizations for after each epoch")
    parser.add_argument("--viz-dir", type=str, default="visualizations", help="Directory to save visualizations of model predictions")
    parser.add_argument("--dino-name", type=str, default="facebook/dinov2-base", help="Name of the DINOv2 backbone model to use from HuggingFace Transformers")
    parser.add_argument("--decoder-channels", type=int, default=256, help="Number of channels in the segmentation decoder")
    parser.add_argument("--decoder-dropout", type=float, default=0.1, help="Dropout rate for the segmentation decoder")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1, help="Number of epochs to freeze the backbone network")
    parser.add_argument("--cv-k", type=int, default=0, help="If >0, use k-fold cross validation.")
    parser.add_argument("--cv-fold", type=int, default=0, help="Which CV fold to use as the held-out test fold.")
    parser.add_argument("--cv-seed", type=int, default=42, help="Seed used to create CV folds.")
    parser.add_argument("--final-train", action="store_true", help="Train one final model on all available data with no validation/test split.")
    parser.add_argument("--class-weighting", choices=["none", "inverse", "sqrt-inverse"],
                    default="none",
                    help="Per-class loss weighting from training-set pixel frequency. "
                         "Applied to BOTH the CE and Jaccard terms.")

    return parser.parse_args(argv)


def main(argv=None, trial=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    image_root = args.image_root.resolve()
    label_roots = [p.resolve() for p in args.label_root] if args.label_root else [args.image_root.resolve()]

    output_dir = Path(args.output_dir.format(model=args.model, date=datetime.now())).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # class ordering determined by LABEL_HIERARCHY for consistent mapping
    class_names = sorted(LABEL_HIERARCHY.keys(), key=lambda k: LABEL_HIERARCHY[k])
    class_map = {name: idx for idx, name in enumerate(class_names)}

    frame_records = discover_frame_records(image_root)
    labels = discover_labels(label_roots)

    # partition labels by folder (used only for 'sampled_train_only' mode)
    sampled_labels = {
        dt: p for dt, p in labels.items()
        if any(part.lower() == "sampled" for part in p.parts)
    }
    unsampled_labels = {
        dt: p for dt, p in labels.items()
        if any(part.lower() == "unsampled" for part in p.parts)
    }

    # build samples for both modes
    samples_all, drop_stats_all = build_temporal_samples(
        frame_records, labels, context=args.context
    )
    if len(samples_all) == 0:
        raise RuntimeError(
            "No valid samples found. Check roots, filename patterns, and context window."
        )

    if args.final_train:

        train_samples = samples_all
        val_samples = []
        test_samples = []

    else:
        # MODE A: use all labeled samples together
        if args.label_usage == "all":
            samples = samples_all

            if len(samples) == 0:
                raise RuntimeError("No labeled samples found.")

            train_samples, val_samples, test_samples = make_train_val_test_split(
                samples,
                args,
            )

        # MODE B: use only sampled frames for training, and unsampled frames for val/test
        elif args.label_usage == "train_samp":
            sampled_samples,   _ = build_temporal_samples(frame_records, sampled_labels,   context=args.context)
            unsampled_samples, _ = build_temporal_samples(frame_records, unsampled_labels, context=args.context)

            if len(sampled_samples) == 0:
                raise RuntimeError(
                    "No samples from sampled/ folder. Check --label-root or folder names."
                )
            if len(unsampled_samples) == 0:
                raise RuntimeError(
                    "No samples from unsampled/ folder. Cannot build val/test."
                )

            train_samples = sampled_samples

            if args.split_type == "day":
                # Split unsampled days into val/test only.
                # Reuse split_days but treat val/test as the two outputs we keep.
                unsampled_days = sorted({s.day for s in unsampled_samples})

                # Use val_ratio relative to the unsampled pool: val gets val_ratio of it,
                # test gets the rest. Adjust if you want a different convention.
                val_frac_of_unsampled = args.val_ratio / max(1e-6, args.val_ratio + (1.0 - args.train_ratio - args.val_ratio))
                # Simpler & clearer: just split unsampled_days roughly val:test = val_ratio : (1 - train_ratio - val_ratio)

                rng = random.Random(args.seed)
                rng.shuffle(unsampled_days)
                n_days = len(unsampled_days)
                n_val_days = max(1, int(round(n_days * val_frac_of_unsampled))) if n_days > 1 else n_days
                val_days  = set(unsampled_days[:n_val_days])
                test_days = set(unsampled_days[n_val_days:])

                # Make sure no sampled-train day overlaps with val/test days.
                train_days = {s.day for s in train_samples}
                bad_val  = train_days & val_days
                bad_test = train_days & test_days
                if bad_val or bad_test:
                    # move overlapping days out of val/test to keep splits day-disjoint.
                    val_days  -= train_days
                    test_days -= train_days
                    print(f"[warn] removed {len(bad_val)} val day(s) and {len(bad_test)} test day(s) "
                        f"that overlapped with sampled-train days.")

                val_samples  = [s for s in unsampled_samples if s.day in val_days]
                test_samples = [s for s in unsampled_samples if s.day in test_days]

            else:
                rng = random.Random(args.seed)
                xs = list(unsampled_samples)
                rng.shuffle(xs)
                rem_frac = max(1e-6, 1.0 - args.train_ratio)
                val_frac_of_unsampled = args.val_ratio / rem_frac
                n_val = max(1, int(round(len(xs) * val_frac_of_unsampled))) if xs else 0
                val_samples  = xs[:n_val]
                test_samples = xs[n_val:]

        # MODE C: use ONLY sampled frames, split into train/val/test
        elif args.label_usage == "only_samp":
            sampled_samples, _ = build_temporal_samples(
                frame_records,
                sampled_labels,
                context=args.context,
            )

            if len(sampled_samples) == 0:
                raise RuntimeError(
                    "No samples from sampled/ folder. Check --label-root or folder names."
                )

            train_samples, val_samples, test_samples = make_train_val_test_split(
                sampled_samples,
                args,
            )

        else:
            raise ValueError(f"Unknown --label-usage value: {args.label_usage}")

    # see how much the training sample size matters
    if args.train_subset_size is not None:
        rng = random.Random(args.train_subset_seed)
        # sort first for determinism, then shuffle with the subset seed
        pool = sorted(train_samples, key=lambda s: (s.day, s.center_dt))
        rng.shuffle(pool)
        train_samples = pool[:args.train_subset_size]

    # --- Final sanity checks ---
    if len(train_samples) == 0:
        raise RuntimeError("Training set is empty. Check label folders, ratios, and split type.")

    if not args.final_train:
        if len(val_samples) == 0 or len(test_samples) == 0:
            raise RuntimeError(
                "Validation or test set is empty. Lower split ratios, label more days, "
                "or switch --label-usage / --split-type."
            )

    # --- Logging ---
    print(f"[mode] label_usage={args.label_usage}  split_type={args.split_type}")

    if args.cv_k > 0:
        print(
            f"[cv] k={args.cv_k}  fold={args.cv_fold}  cv_seed={args.cv_seed}"
        )

    print(
        f"[split] samples -> train: {len(train_samples)}  "
        f"val: {len(val_samples)}  test: {len(test_samples)}"
    )

    if args.split_type == "day":
        td = {s.day for s in train_samples}
        vd = {s.day for s in val_samples}
        sd = {s.day for s in test_samples}

        print(f"[split] days    -> train: {len(td)}  val: {len(vd)}  test: {len(sd)}")

        overlap = (td & vd) | (td & sd) | (vd & sd)

        if overlap:
            print(f"[warn] day overlap detected across splits: {sorted(overlap)}")

    save_manifest(train_samples, output_dir / "manifest_train.jsonl")
    save_manifest(val_samples,   output_dir / "manifest_val.jsonl")
    save_manifest(test_samples,  output_dir / "manifest_test.jsonl")

    # compile the training dataset based on the configuration
    ds_train = TemporalSatRadDataset(
        train_samples, args.width, args.height, class_map, LABEL_HIERARCHY,
        aug_rotdeg=args.aug_rotdeg,
        aug_elastic=args.aug_elastic,
        aug_elastic_alpha=args.aug_elastic_alpha,
        aug_elastic_grid=args.aug_elastic_grid,
    )

    # expand JUST the training dataset with the augmentations specified
    if args.aug_expand and args.aug_expand > 1:
        ds_train = AugmentedRepeatDataset(ds_train, repeat=args.aug_expand, global_seed=args.seed)

        # --- Sanity check: verify each repeat produces a different augmentation ---
        if args.aug_elastic or args.aug_hflip or args.aug_vflip or args.aug_rotdeg > 0:
            N = len(ds_train.base)
            probe_idx = 0  # any base index works
            x0, y0, *_ = ds_train[probe_idx]
            x1, y1, *_ = ds_train[probe_idx + N]        # same base image, repeat #2
            x2, y2, *_ = ds_train[probe_idx + 2 * N]    # same base image, repeat #3

            same_01 = torch.equal(x0, x1)
            same_02 = torch.equal(x0, x2)
            print(f"[aug-sanity] repeat0 vs repeat1 identical? {same_01}")
            print(f"[aug-sanity] repeat0 vs repeat2 identical? {same_02}")
            if same_01 or same_02:
                print("[aug-sanity] WARNING: repeats produced identical tensors — "
                    "your 16x expansion is NOT giving unique augmentations!")
            else:
                print("[aug-sanity] OK: repeats produce distinct augmentations.")
    ds_val = TemporalSatRadDataset(val_samples, args.width, args.height, class_map, LABEL_HIERARCHY)
    ds_test = TemporalSatRadDataset(test_samples, args.width, args.height, class_map, LABEL_HIERARCHY)

    # create data loaders for each split
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    channels_per_frame = 6 # 3 for satellite image, 3 for radar image
    in_channels = channels_per_frame * (2 * args.context + 1) # chanels in the input will increase if context images are included
    num_classes = len(class_map)

    # use CUDA gpu (desktop - RTX4000 ada and Grace - A100 have them)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.model == "unet":
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=args.base_channels).to(device)

        class_weights = None
        if args.class_weighting != "none":
            class_weights = compute_class_weights(
                train_samples, args.width, args.height, class_map,
                scheme=args.class_weighting,
            ).to(device)
            print(f"[class-weights] {args.class_weighting}:",
                {n: round(float(class_weights[i]), 3) for n, i in class_map.items()})

        criterion = CEJaccardLoss(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            weight=class_weights,          # None -> unweighted, same as today
            alpha=0.25,
        )

        # # use same optimizer from Ronneberger et al. 2015
        # optimizer = torch.optim.SGD(
        # model.parameters(),
        # lr=args.lr,
        # momentum=0.99,
        # )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == "dino":
        # build the overall model configuration and use CNN to go from 6 channels to 3 channels
        model = Dinov2SegmentationModel(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone_name=args.dino_name,
            decoder_channels=args.decoder_channels,
            dropout=args.decoder_dropout,
        ).to(device) # throw it on the gpu, hopefully

        # freeze the backbone a specific number of epochs - ONLY IF SPECIFIED
        if args.freeze_backbone_epochs > 0:
            model.freeze_backbone()

        class_weights = None
        if args.class_weighting != "none":
            class_weights = compute_class_weights(
                train_samples, args.width, args.height, class_map,
                scheme=args.class_weighting,
            ).to(device)
            print(f"[class-weights] {args.class_weighting}:",
                {n: round(float(class_weights[i]), 3) for n, i in class_map.items()})

        criterion = CEJaccardLoss(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            weight=class_weights,          # None -> unweighted, same as today
            alpha=0.25,
        )

        # set up AdamW optimizer with different learning rates for the backbone and the segmentation head
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                {"params": model.input_adapter.parameters(), "lr": args.head_lr},
                {"params": model.decoder.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )


    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    best_val_miou = -math.inf
    log_rows = []

    # Write summary metadata for reproducibility
    summary = {
        "image_root": str(image_root),
        "label_roots": [str(r) for r in label_roots],
        "total_frame_records": len(frame_records),
        "total_labels": len(labels),
        "drop_stats": drop_stats_all,
        "split_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "shape": {"height": args.height, "width": args.width},
        "context": args.context,
        "in_channels": in_channels,
        "classes": class_map,
        "train_subset_size": args.train_subset_size,
        "train_subset_seed": args.train_subset_seed,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Dataset Summary ===")
    print(json.dumps(summary["split_counts"], indent=2))
    print("Dropped stats:", summary["drop_stats"])
    print("Using device:", device)

    epochs_since_improve = 0

    # run through all the epochs requested
    for epoch in range(1, args.epochs + 1):

        # unfreeze the backbone after the specified number of epochs (if any)
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0 and args.model == "dino":
            model.unfreeze_backbone()

        # run training epoch and get metrics
        tr_loss, tr_acc, tr_iou = run_epoch(
            model,
            loader_train,
            criterion,
            optimizer,
            device,
            num_classes,
            total_epochs=args.epochs,
        )

        tr_miou = np.nanmean(tr_iou)

        if args.final_train:
            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_pixel_acc": tr_acc,
                "train_miou": tr_miou,
            }
            log_rows.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} mIoU {tr_miou:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_map": class_map,
                    "context": args.context,
                    "shape": (args.height, args.width),
                    "model": args.model,
                    **({"dino_name": args.dino_name} if args.model == "dino" else {}),
                },
                output_dir / "final_model.pt",
            )

            continue

        # run validation epoch and get metrics (no optimizer, so it runs in eval mode)
        va_loss, va_acc, va_iou = run_epoch(
            model,
            loader_val,
            criterion,
            None,
            device,
            num_classes,
            total_epochs=args.epochs,
        )

        va_miou = np.nanmean(va_iou)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_pixel_acc": tr_acc,
            "train_miou": tr_miou,
            "val_loss": va_loss,
            "val_pixel_acc": va_acc,
            "val_miou": va_miou,
            # number of classes each mean covers -- mIoU is only comparable
            # between runs whose splits scored the same set of classes.
            "train_n_classes": int(np.count_nonzero(~np.isnan(tr_iou))),
            "val_n_classes": int(np.count_nonzero(~np.isnan(va_iou))),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} mIoU {tr_miou:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} mIoU {va_miou:.4f} "
            f"(over {int(np.count_nonzero(~np.isnan(va_iou)))}/{num_classes} classes)"
        )

        # save model checkpoint, only if validation mIoU improved
        if va_miou > best_val_miou:
            best_val_miou = va_miou
            epochs_since_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_map": class_map,
                    "context": args.context,
                    "shape": (args.height, args.width),
                    "model": args.model,
                    **({"dino_name": args.dino_name} if args.model == "dino" else {}),
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_since_improve += 1

        with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
            for r in log_rows:
                f.write(json.dumps(r) + "\n")

        # inside the epoch loop, after computing va_miou
        if trial is not None:
            trial.report(va_miou, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        # early stopping check
        if not args.final_train:
            if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                print(f"Early stopping triggered after {epochs_since_improve} epochs without improvement.")
                break

    if args.final_train:
        print(f"Final model saved to {output_dir / 'final_model.pt'}")
        return

    # load best checkpoint and evaluate on test set
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # run test epoch and get metrics (no optimizer, so it runs in eval mode)
    te_loss, te_acc, te_iou = run_epoch(model, loader_test, criterion, None, device, num_classes)

    te_miou = np.nanmean(te_iou)

    test_report = {
        "test_loss": te_loss,
        "test_pixel_acc": te_acc,
        "test_miou": te_miou,
        "per_class_iou": {
            name: float(te_iou[idx]) if not np.isnan(te_iou[idx]) else None
            for name, idx in class_map.items()
        },
        # test_miou is the mean over these classes only. Runs that scored a
        # different set are not directly comparable.
        "n_classes_evaluated": int(np.count_nonzero(~np.isnan(te_iou))),
        "classes_evaluated": [
            name for name, idx in class_map.items() if not np.isnan(te_iou[idx])
        ],
    }
    with (output_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

        # create json with hpyerparameters
    hyper_info = {
        "model": args.model,
        **({"dino_name": args.dino_name} if args.model == "dino" else {}),
        "context": args.context,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "batch_size": args.batch_size,
        **({"base_channels": args.base_channels, "lr": args.lr} if args.model == "unet" else {}),
        **({"decoder_channels": args.decoder_channels, "decoder_dropout": args.decoder_dropout, "freeze_backbone_epochs": args.freeze_backbone_epochs, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay} if args.model == "dino" else {}),
        "split_type": args.split_type,
        "label_usage": args.label_usage,
        "aug_hflip": args.aug_hflip,
        "aug_vflip": args.aug_vflip,
        "aug_rotdeg": args.aug_rotdeg,
        "aug_elastic": args.aug_elastic,
        "aug_elastic_alpha": args.aug_elastic_alpha,
        "aug_elastic_grid": args.aug_elastic_grid,
        "aug_expand": args.aug_expand,
        "class_weighting": args.class_weighting,
        "hierarchy": LABEL_HIERARCHY,
        "merge_map": LABEL_MERGE,

    }
    with (output_dir / "hyper_info.json").open("w", encoding="utf-8") as f:
        json.dump(hyper_info, f, indent=2)

    print("=== Test Report ===")
    print(json.dumps(test_report, indent=2))

    # save visualization samples of model predictions on the test set, if set
    if args.viz_samples > 0:
        if args.model == "dino":
            a_saved = save_dinov2_input_view(
                model=model,
                loader=loader_test,
                device=device,
                context=args.context,
                out_dir=output_dir / args.viz_dir / "adapter_input",
                max_samples=args.viz_samples,
            )
            print(f"Saved {a_saved} adapter visualizations to: {output_dir / args.viz_dir / 'adapter_input'}")

        n_saved = save_prediction_visuals(
            model=model,
            loader=loader_test,
            device=device,
            class_map=class_map,
            context=args.context,
            out_dir=output_dir / args.viz_dir,
            max_samples=args.viz_samples,
        )
        print(f"Saved {n_saved} visualization image(s) to: {output_dir / args.viz_dir}")

    return best_val_miou

if __name__ == "__main__":
    # time the script so i can compare how long it takes to train on diff hyperaparameters
    timer_start = datetime.now().timestamp()
    main() # run the main function which trains the model and updates the dashboard
    timer_end = datetime.now().timestamp()
    elapsed = timer_end - timer_start
    print(f"Elapsed time: {elapsed:.2f} seconds")
