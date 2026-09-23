from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from email import parser
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import rasterio
from rasterio.transform import from_bounds

from train import (
    Dinov2SegmentationModel,
    UNet,
    build_palette_from_class_map,
    blend_mask_over_image,
    discover_frame_records,
    read_rgb_tensor,
    set_seed,
)


@dataclass(frozen=True)
class DaySample:
    day: str
    center_dt: object
    sat_paths: Tuple[Path, ...]
    rad_paths: Tuple[Path, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict all Sat/Rad frames from a single day.",
        fromfile_prefix_chars="@",
    )
    # same args that train.py uses
    parser.add_argument("--model", type=str, required=True, choices=["unet", "dino"], help="Model architecture to use for segmentation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--image-root", type=Path, default="../images/", help="Root directory to search for input satellite/radar PNG files")
    parser.add_argument("--label-root", type=Path, action="append", default=[Path("../unsampled/dataset/labeled/"), Path("../sampled/dataset/labeled/")], help="Root directory(s) to search for label JSON files. Can specify multiple --label-root arguments to include multiple directories.")
    parser.add_argument("--output-dir", type=Path, default="../../../../train_out/{model}/{date:%Y%m%d_%H%M%S}", help="Directory to save prediction outputs")
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
    parser.add_argument("--dino-name", type=str, default="facebook/dinov2-base", help="Name of the DINOv2 backbone model to use from HuggingFace Transformers")
    parser.add_argument("--decoder-channels", type=int, default=256, help="Number of channels in the segmentation decoder")
    parser.add_argument("--decoder-dropout", type=float, default=0.1, help="Dropout rate for the segmentation decoder")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1, help="Number of epochs to freeze the backbone network")
    parser.add_argument("--run-output-dir", type=str, default="../runs/{version}", help="Directory to save run outputs (e.g. predictions, logs). Supports {version} and {date} placeholders.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--radar", type=str, default="KHGX", help="Radar region to find the images (e.g. KHGX, KHGK, KJAX, KMLB)")
    parser.add_argument("--day", type=str, nargs="+", default=None,
                        help="One or more days YYYYMMDD (space-separated)")
    parser.add_argument("--day-range", type=str, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Inclusive date range: YYYYMMDD YYYYMMDD")
    return parser

def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_day_samples(image_root: Path, day: str, context: int) -> List[DaySample]:
    frame_records = discover_frame_records(image_root)
    day_records = [
        rec for rec in frame_records.values()
        if rec.day == day and rec.sat_path is not None and rec.rad_path is not None
    ]
    day_records = sorted(day_records, key=lambda rec: rec.pair_dt)

    samples: List[DaySample] = []
    for idx, rec in enumerate(day_records):
        if idx - context < 0 or idx + context >= len(day_records):
            continue

        window = day_records[idx - context : idx + context + 1]
        sat_paths = tuple(w.sat_path for w in window if w.sat_path is not None)
        rad_paths = tuple(w.rad_path for w in window if w.rad_path is not None)

        if len(sat_paths) != len(window) or len(rad_paths) != len(window):
            continue

        samples.append(
            DaySample(
                day=day,
                center_dt=rec.pair_dt,
                sat_paths=sat_paths,
                rad_paths=rad_paths,
            )
        )

    return samples


def load_model(
    checkpoint_path: Path,
    run_output_dir: Path,
    device: torch.device,
    context: int,
    height: int,
    width: int,
    model_name: str,
    base_channels: int,
    dino_name: str,
    decoder_channels: int,
    decoder_dropout: float,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_map = ckpt["class_map"]
    checkpoint_context = ckpt.get("context", context)
    if checkpoint_context != context:
        print(f"Warning: checkpoint context={checkpoint_context}, using CLI context={context}")

    in_channels = 6 * (2 * context + 1)
    num_classes = len(class_map)

    if model_name == "unet":
        model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        ).to(device)
    elif model_name == "dino":
        model = Dinov2SegmentationModel(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone_name=dino_name,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_map


def get_config_version_from_argv(argv: Optional[List[str]] = None) -> str:
    argv = sys.argv[1:] if argv is None else argv

    for token in argv:
        if token.startswith("@"):
            config_path = Path(token[1:]).expanduser()
            if not config_path.is_absolute():
                config_path = (Path.cwd() / config_path).resolve()

            # use the parent folder name, e.g. "dino_v1.5"
            return config_path.parent.name or config_path.stem

    return "unknown"


@torch.no_grad()
def predict_day(
    model: torch.nn.Module,
    samples: List[DaySample],
    device: torch.device,
    run_output_dir: Path,
    height: int,
    width: int,
    context: int,
    class_map: Dict[str, int],
) -> int:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    palette = build_palette_from_class_map(class_map)
    saved = 0

    # Build the affine transform once — same bbox/grid for every frame
    # Geographic bounds of the Sat/Rad imagery (from sat_rad.py)
    BBOX_MIN_LON = -96.1164
    BBOX_MAX_LON = -94.0544
    BBOX_MIN_LAT = 28.5698
    BBOX_MAX_LAT = 30.3741
    
    transform = from_bounds(
        west=BBOX_MIN_LON,
        south=BBOX_MIN_LAT,
        east=BBOX_MAX_LON,
        north=BBOX_MAX_LAT,
        width=width,
        height=height,
    )

    for sample in samples:
        chan_tensors = []
        for sat_p, rad_p in zip(sample.sat_paths, sample.rad_paths):
            sat = read_rgb_tensor(sat_p, width, height)
            rad = read_rgb_tensor(rad_p, width, height)
            chan_tensors.append(sat)
            chan_tensors.append(rad)

        x = torch.cat(chan_tensors, dim=0).unsqueeze(0).to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        x_cpu = x.squeeze(0).cpu().numpy()
        center_sat_start = context * 6
        center_sat = x_cpu[center_sat_start : center_sat_start + 3]
        center_sat = np.transpose(center_sat, (1, 2, 0))
        center_sat = np.clip(center_sat, 0.0, 1.0)
        center_sat_rgb = (center_sat * 255.0).astype(np.uint8)

        pred_vis = blend_mask_over_image(center_sat_rgb, pred, palette, alpha=0.3)

        center_str = sample.center_dt.strftime("%Y%m%d_%H%M")
        base_name = f"{sample.day}_{center_str}_pred"

        # 1) Existing PNG overlay
        Image.fromarray(pred_vis).save(run_output_dir / f"{base_name}_overlay.png")

        # 2) New georeferenced class-index mask (single-band uint8 GeoTIFF)
        tif_path = run_output_dir / f"{base_name}_mask.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=pred.shape[0],
            width=pred.shape[1],
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(pred, 1)
            # Store the class map in tags so downstream code can decode class indices
            dst.update_tags(**{f"class_{v}": k for k, v in class_map.items()})

        saved += 1

    return saved

def expand_days(args) -> List[str]:
    days: List[str] = list(args.day or [])
    if args.day_range:
        start = datetime.strptime(args.day_range[0], "%Y%m%d").date()
        end   = datetime.strptime(args.day_range[1], "%Y%m%d").date()
        if end < start:
            raise ValueError(f"--day-range end {end} is before start {start}")
        d = start
        while d <= end:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
    if not days:
        raise ValueError("You must supply --day and/or --day-range")
    return sorted(set(days))

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    version = get_config_version_from_argv()
    output_dir_template = str(args.run_output_dir).format(version=version)
    base_output_dir = Path(output_dir_template).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    days = expand_days(args)
    print(f"Will process {len(days)} day(s): {days}")

    # Load the model ONCE — big win for DINOv2
    model, class_map = load_model(
        checkpoint_path=args.checkpoint.resolve(),
        run_output_dir=base_output_dir,
        device=device,
        context=args.context,
        height=args.height,
        width=args.width,
        model_name=args.model,
        base_channels=args.base_channels,
        dino_name=args.dino_name,
        decoder_channels=args.decoder_channels,
        decoder_dropout=args.decoder_dropout,
    )

    radar = args.radar.upper()

    if args.radar == 'KGHX':
        image_root = args.image_root.resolve()

    else:  # args.radar == 'KHGK'
        image_root_path = f"../other_images/{radar}/"
        image_root = Path(image_root_path)

    total_saved = 0
    for day in days:
        samples = build_day_samples(image_root, day, args.context)
        if not samples:
            print(f"[skip] no valid samples for day {day}")
            continue

        day_out = base_output_dir / day
        n_saved = predict_day(
            model=model,
            samples=samples,
            device=device,
            run_output_dir=day_out,
            height=args.height,
            width=args.width,
            context=args.context,
            class_map=class_map,
        )
        print(f"[{day}] saved {n_saved} mask(s) -> {day_out}")
        total_saved += n_saved

    print(f"Done. Total masks saved: {total_saved}")
    print("Class map:", class_map)


if __name__ == "__main__":
    main()