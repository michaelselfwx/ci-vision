from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import re
from datetime import datetime as dt
import pvlib.solarposition as solarposition
import shutil
import argparse

# ================= USER SETTINGS =================
PARENT_IMAGE_DIR = "../images"  # parent directory containing day subdirectories
DAY_GLOB = "2022*"  # pattern to match day folders (e.g., 20220708)
OUTPUT_CSV = "../output/cloud_cover_summer_conv_6_test1.csv"  # output summary CSV
WHITE_THRESHOLD = 110  # 0–255, higher = stricter white
BIN_EDGES = [0, 5, 15, 30, 50, 70, 85, 95, 100]  # custom bin edges for cloud cover %
SAMPLE_PERCENT = 0.01  # 1% sample per cluster
OUTPUT_CLUSTER_DIR = "../output/clustered_samples"
RANDOM_SEED = 10  # used random seed to sample dataset was 10
SAVE_MASK_PREVIEW = True  # set False to disable saving transparent red overlays
SKIP_DATES = [
    "0603",
    "0604",
    "0605",
    "0606",
    "0607",
    "0608",
    "0609",
    "0611",
    "0612",
    "0624",
    "0625",
    "0719",
    "0720",
    "0912",
    "0913",
    "0921",
    "0922",
    "0923",
    "0924",
    "0925",
    "0926",
    "0927",
    "0928",
    "0929",
    "0930",
]
TAMU_DATES = [
    "0626",
    "0711",
    "0713",
    "0727",
    "0728",
    "0729",
    "0807",
    "0808",
    "0809",
    "0826",
    "0828",
    "0831",
    "0917",
    "0918",
    "0919",
]
# Overlay text settings
OVERLAY_TEXT = True
TEXT_FONT_SIZE = 24
TEXT_COLOR = (255, 255, 255)
TEXT_STROKE = (0, 0, 0)
TEXT_MARGIN = 10
# ================================================


def extract_datetime_from_filename(path):
    """
    Extract the datetime from the filename (which is in YYYYMMDD_HHMM format).
    Args:
        path (str): The file path.
    Returns:
        datetime: The extracted datetime object.
    """
    filename = path.split("/")[-1]
    match = re.search(r"(\d{8})_(\d{4})", filename)
    if not match:
        raise ValueError("Filename does not contain a valid datetime format.")
    date_str, time_str = match.groups()
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])

    # convert to pandas datetime
    time = dt(year, month, day, hour, minute)

    return time


def compute_cloud_percent(image_path):
    """
    Compute the cloud percent in the image at image_path.
    Args:
        image_path (Path): The path to the image file.
    Returns:
        float: The percentage of cloud cover.
        np.ndarray: The white mask array.
        np.ndarray: The original image array.
        float: The solar zenith angle.
        float: The elevation factor based on zenith angle.
    """
    # grab image and convert to numpy rgb array
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # compute solar position by datetime from filename
    image_time = extract_datetime_from_filename(str(image_path))
    image_time_utc = pd.DatetimeIndex([image_time], tz="UTC")
    solar_pos = solarposition.ephemeris(
        image_time_utc, latitude=29.4719, longitude=-95.0787
    )  # the lat and lon of KHGX radar
    zenith = solar_pos.zenith.values[0]
    normalized_zenith = zenith / 90  # 90 degrees zenith is horizon (sunset/sunrise)
    elev_factor = 1 - np.power(
        normalized_zenith, 6
    )  # power scaling cuz it gets dark fast near sunset
    dynamic_threshold = WHITE_THRESHOLD * elev_factor

    # create white mask and compute white percent (cloud cover)
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    white_mask = (
        (r >= dynamic_threshold) & (g >= dynamic_threshold) & (b >= dynamic_threshold)
    )
    total_pixels = img_np.shape[0] * img_np.shape[1]
    white_percent = (np.sum(white_mask) / total_pixels) * 100

    if SAVE_MASK_PREVIEW:
        overlay_dir = Path("../output/cloud_mask_previews")
        overlay_dir.mkdir(parents=True, exist_ok=True)
        base_img = Image.fromarray(img_np).convert("RGBA")
        overlay = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
        # red with visible alpha
        overlay[..., 0] = 255
        overlay[..., 3] = (
            white_mask.astype(np.uint8) * 120
        )  # increase alpha for visibility
        mask_img = Image.fromarray(overlay, mode="RGBA")
        blended = Image.alpha_composite(base_img, mask_img).convert("RGB")
        # ---- add text overlay ----
        if OVERLAY_TEXT:
            text = f"{extract_datetime_from_filename(str(image_path)).strftime('%Y-%m-%d %H:%M UTC')} | cloud {white_percent:.1f}%"
            try:
                font = ImageFont.truetype("arial.ttf", TEXT_FONT_SIZE)
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", TEXT_FONT_SIZE)
                except Exception:
                    font = ImageFont.load_default()
            draw = ImageDraw.Draw(blended)
            # compute bottom-left position
            try:
                bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textsize(text, font=font)
            x = TEXT_MARGIN
            y = blended.height - text_h - TEXT_MARGIN
            draw.text(
                (x, y),
                text,
                fill=TEXT_COLOR,
                font=font,
                stroke_width=2,
                stroke_fill=TEXT_STROKE,
            )
        # ---------------------------
        out_path = overlay_dir / (Path(image_path).stem + "_mask.png")
        blended.save(out_path)
        print(f"Saved mask preview to {out_path}")

    return white_percent, white_mask, img_np, zenith, elev_factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cloud cover and generate mask previews"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute cloud cover and mask previews even if CSV exists",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Process only a specific date (format: YYYYMMDD, e.g., 20220626). If not provided, processes all dates.",
    )

    args = parser.parse_args()

    output_csv_path = Path(OUTPUT_CSV)
    skip_dates_full = {f"2022{d}" for d in SKIP_DATES}
    tamu_dates_full = [f"2022{d}" for d in TAMU_DATES]

    # Check if CSV already exists
    if output_csv_path.exists() and not args.force_recompute:
        print(f"Loading existing CSV from {OUTPUT_CSV}")
        pd_results = pd.read_csv(output_csv_path)
        pd_results["date"] = pd_results["date"].astype(
            str
        )  # Ensure date column is string
        print(f"Loaded {len(pd_results)} images from cached CSV")
    else:
        day_dirs = sorted(Path(PARENT_IMAGE_DIR).glob(DAY_GLOB))
        if not day_dirs:
            raise FileNotFoundError(
                f"No day directories found in {PARENT_IMAGE_DIR} with pattern {DAY_GLOB}"
            )

        all_results = []
        for day_dir in day_dirs:
            # Filter to specific date if provided
            if args.date and day_dir.name != args.date:
                continue

            if not day_dir.is_dir():
                continue
            if day_dir.name in skip_dates_full:
                print(f"Skipping day (in SKIP_DATES): {day_dir.name}")
                continue
            image_files = sorted(day_dir.glob("*_sat.png"))
            if not image_files:
                print(f"No images found in {day_dir.name}")
                continue

            print(f"Processing {day_dir.name} ({len(image_files)} images)...")
            for image_path in image_files:
                white_percent, white_mask, img_np, zenith, elev_factor = (
                    compute_cloud_percent(image_path)
                )
                all_results.append(
                    {
                        "date": day_dir.name,
                        "image": image_path.name,
                        "cloud_percent": white_percent,
                        "zenith": zenith,
                        "elev_factor": elev_factor,
                    }
                )

        pd_results = pd.DataFrame(all_results)
        pd_results["date"] = pd_results["date"].astype(
            str
        )  # Ensure date column is string
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\nProcessed {len(all_results)} images")
        print(f"Saved results to {OUTPUT_CSV}")

    # ========== CLUSTERING AND SAMPLING ==========
    print("\n" + "=" * 50)
    print("CLUSTERING AND SAMPLING")
    print("=" * 50)

    # Custom bin edges clustering
    pd_results["cluster"] = pd.cut(
        pd_results["cloud_percent"],
        bins=BIN_EDGES,
        labels=range(len(BIN_EDGES) - 1),
        include_lowest=True,
    ).astype(int)

    # Print cluster statistics
    print("\nCluster statistics:")
    cluster_stats = pd_results.groupby("cluster")["cloud_percent"].agg(
        ["count", "min", "max", "mean"]
    )
    print(cluster_stats)

    # Create output directory
    output_dir = Path(OUTPUT_CLUSTER_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample and copy images from each cluster
    all_sampled = []
    for cluster_id in sorted(pd_results["cluster"].unique()):
        cluster_df = pd_results[pd_results["cluster"] == cluster_id]
        n_samples = max(1, int(round(len(cluster_df) * SAMPLE_PERCENT, 0)))

        # Split into priority and non-priority
        priority_df = cluster_df[cluster_df["date"].isin(tamu_dates_full)]
        non_priority_df = cluster_df[~cluster_df["date"].isin(tamu_dates_full)]

        # Calculate target: at least 50% from priority days
        n_priority_target = max(1, int(np.ceil(n_samples * 0.5)))
        n_priority_actual = min(n_priority_target, len(priority_df))
        n_non_priority = n_samples - n_priority_actual

        # Sample from each group
        sampled_parts = []
        if n_priority_actual > 0 and len(priority_df) > 0:
            sampled_parts.append(
                priority_df.sample(n=n_priority_actual, random_state=RANDOM_SEED)
            )
        if n_non_priority > 0 and len(non_priority_df) > 0:
            sampled_parts.append(
                non_priority_df.sample(n=n_non_priority, random_state=RANDOM_SEED)
            )

        sampled = (
            pd.concat(sampled_parts, ignore_index=True)
            if sampled_parts
            else pd.DataFrame()
        )

        # Create cluster folder with bin range
        bin_label = f"{BIN_EDGES[cluster_id]}-{BIN_EDGES[cluster_id+1]}"
        cluster_dir = output_dir / f"cluster_{cluster_id}_{bin_label}"
        cluster_dir.mkdir(exist_ok=True)

        print(
            f"\nCluster {cluster_id} ({bin_label}%): {len(cluster_df)} images, sampling {n_samples} ({n_priority_actual} priority, {n_non_priority} non-priority)"
        )

        # Copy sampled images
        for _, row in sampled.iterrows():
            src_path = Path(PARENT_IMAGE_DIR) / str(row["date"]) / row["image"]
            dst_path = cluster_dir / f"{row['date']}_{row['image']}"
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: {src_path} not found")

        all_sampled.append(sampled)

    # Save sampled metadata
    sampled_df = pd.concat(all_sampled, ignore_index=True)
    sampled_df.to_csv(output_dir / "sampled_metadata.csv", index=False)
    print(f"\nTotal sampled: {len(sampled_df)} images")
    print(f"Saved to {OUTPUT_CLUSTER_DIR}")
