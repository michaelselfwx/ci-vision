# ci-vision

Semantic segmentation of convective-initiation (CI) features from co-located **GOES ABI visible satellite** imagery and **NEXRAD radar** reflectivity. Two model families are included — a **U-Net** trained from scratch and a **DINOv2**-backbone segmentation model — trained on hand-labeled summer 2022 imagery over Houston (KHGX).

Each input sample is a satellite/radar image pair (512 × 448 px, 200 km × 200 km box centered on the radar). The model predicts one class per pixel:

| Index | Label | Meaning |
|---|---|---|
| 0 | `OCEAN` | Open water (Gulf of Mexico / Galveston Bay) |
| 1 | `LAND` | Clear land surface |
| 2 | `OBSCR` | Obscured (e.g. high cloud / anvil hiding what's below) |
| 3 | `SHALLOW` | Shallow cumulus |
| 4 | `HCRF` | Horizontal convective roll field |
| 5 | `SBF` | Sea-breeze front |
| 6 | `OUTFLOW` | Outflow boundary |
| 7 | `DEEP` | Deep convection |

Where labels overlap, the higher-index class wins (see `LABEL_HIERARCHY` in `scripts/train.py`).

---

## Repository layout

```
ci-vision/
├── configs/            # Training argument files, one folder per model version (see configs/readme.md)
│   ├── dino_v3.5/dino_v3.5.config
│   ├── unet_v1.2/unet_v1.2.config
│   └── ...
├── sampled/dataset/    # Labeled frames chosen by cloud-cover-stratified sampling (see sampled/README.md)
│   ├── labeled/YYYYMMDD/   # *_sat.png, *_rad.png + LabelMe *.json labels
│   └── visual/YYYYMMDD/    # Label overlays for quick visual inspection
├── unsampled/dataset/  # Additional labeled frames not chosen by the sampler (see unsampled/README.md)
│   └── labeled/YYYYMMDD/
├── scripts/            # Data download, labeling helpers, training, inference, evaluation (see scripts/README.md)
├── LICENSE             # MIT
└── README.md
```

Folders the scripts create **outside** of what's committed (all paths are relative to `scripts/`):

| Path | Created by | Contents |
|---|---|---|
| `../other_images/<RADAR>/YYYYMMDD/{sat,rad}/` | `sat_rad.py` | Newly downloaded image pairs |
| `../../data/` | `sat_rad.py` (goes2go) | Raw GOES NetCDF cache |
| `../runs/<config-version>/YYYYMMDD/` | `run_model.py` | Predictions (overlay PNG + GeoTIFF mask) |
| `../../../../train_out/...` or `../train_out/...` | `train.py`, `evaluate.py`, `crossval.py` | Checkpoints, logs, test reports |

### File naming

Every image and label follows the same pattern, which the code relies on to pair frames:

```
YYYYMMDD_PPPP_TTTT_{sat|rad}.png
         │    │
         │    └── actual scan time of this image (UTC, HHMM)
         └─────── "pair time" — midpoint of the sat & radar scan times (UTC, HHMM)
```

Example: `20220601_2243_2242_sat.png` and `20220601_2243_2243_rad.png` form one pair (pair time 22:43 UTC). A sat and rad image are matched when their pair time is identical, and they're only paired if the scans are within 2.5 minutes of each other.

---

## Installation

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for training and for the DINOv2 model; inference with U-Net runs fine on CPU.

```bash
git clone https://github.com/michaelselfwx/ci-vision.git
cd ci-vision

# 1. PyTorch: pick the right build for your CUDA version from https://pytorch.org
pip install torch torchvision

# 2. cartopy + Py-ART install most reliably from conda-forge
conda install -c conda-forge cartopy arm_pyart

# 3. everything else
pip install -r requirements.txt
```

> **All scripts use paths relative to `scripts/`, so run them from inside that folder:** `cd scripts`

---

## Quick start: run the trained model on your own data

The workflow is: **(1) define your domain → (2) download sat/radar pairs → (3) run inference.**

### 1. Get a trained checkpoint

Model weights are not committed to the repo. Download a checkpoint (`best_model.pt`) from **[TODO: add link to released weights]**, or train your own (see [Training](#training-your-own-model)). A checkpoint is a PyTorch dict containing `model_state`, `class_map`, and `context`.

You also need to know which architecture/config produced it — e.g. a checkpoint trained with `configs/dino_v3.5/dino_v3.5.config` must be run with `--model dino --dino-name facebook/dinov2-base --decoder-channels 256`.

### 2. Get your own data

Imagery is pulled anonymously from the NOAA Open Data buckets on AWS — no account needed.

**a. Compute the domain for your radar.** The model was trained on a 200 km × 200 km box centered on the radar. Edit the latitude/longitude at the bottom of `scripts/bounding_box.py` and run it:

```bash
cd scripts
python bounding_box.py
# Bounding box: (min_lon, max_lon, min_lat, max_lat)
```

Coordinates for a few sites are listed at the bottom of that file (KHGX, KMLB, KJAX). Any NEXRAD site works — look up its lat/lon from the [NOAA NEXRAD site list](https://www.roc.noaa.gov/branches/program-branch/site-id-database/).

**b. Configure `scripts/sat_rad.py`.** Edit the constants at the top of the file and the date range in the `__main__` block:

```python
RADAR_STATION = "KMLB"      # 4-letter NEXRAD ID
GOES_SATELLITE = 16         # 16 = GOES-East before Apr 2025, 19 = GOES-East after; 18 = GOES-West
DOMAIN_BOUNDS = {           # output of bounding_box.py
    "min_lon": -81.6721, "max_lon": -79.6367,
    "min_lat": 27.2106,  "max_lat": 29.0153,
}

# in __main__:
start_date = datetime.date(2023, 7, 1)
end_date   = datetime.date(2023, 7, 3)
```

By default each day is processed from **14:00–23:59 UTC** (daytime, since the satellite input is visible imagery). Change the `start`/`end` times in the loop if needed.

**c. Run it:**

```bash
python sat_rad.py
```

For each day, the script lists all GOES CONUS (ABI-L2-CMIPC) scans and NEXRAD Level II volumes, pairs those within 2.5 minutes, and writes:

```
../other_images/<RADAR_STATION>/YYYYMMDD/sat/YYYYMMDD_PPPP_TTTT_sat.png   # "true-green" visible composite (bands 1/2/3)
../other_images/<RADAR_STATION>/YYYYMMDD/rad/YYYYMMDD_PPPP_TTTT_rad.png   # 0.5° reflectivity, ≥ 8 dBZ, HomeyerRainbow colormap
```

> It uses up to 32 processes / 16 threads (`max_workers` in `__main__`). Lower these on a laptop.

### 3. Run inference

```bash
python run_model.py \
  --model dino \
  --dino-name facebook/dinov2-base \
  --decoder-channels 256 \
  --checkpoint /path/to/best_model.pt \
  --radar KMLB \
  --bbox -81.6721 -79.6367 27.2106 29.0153 \
  --day-range 20230701 20230703 \
  --run-output-dir ../runs/dino_v3.5
```

Key arguments:

| Argument | Description |
|---|---|
| `--model` | `unet` or `dino` — must match the checkpoint |
| `--checkpoint` | Path to `best_model.pt` / `final_model.pt` |
| `--radar` | Station ID of the imagery (default `KHGX`) |
| `--input-root` | Folder of sat/rad PNGs to predict on (default `../other_images/<RADAR>/`, where `sat_rad.py` saves them) |
| `--bbox` | `MIN_LON MAX_LON MIN_LAT MAX_LAT` of the imagery, used to georeference the GeoTIFFs. Defaults to the entry for `--radar` in `RADAR_BOUNDS` (KHGX, KMLB, KJAX); required for any other site |
| `--day` | One or more days: `--day 20230701 20230715` |
| `--day-range` | Inclusive range: `--day-range 20230701 20230731` |
| `--run-output-dir` | Where predictions go (default `../runs/{version}`) |
| `--context` | Temporal frames on each side; must match training (all released configs use `0`) |
| `--base-channels` | U-Net only (default 64) |
| `--dino-name`, `--decoder-channels`, `--decoder-dropout` | DINOv2 only — must match training |
| `--device` | `auto` (default), `cuda`, or `cpu` |

You can also pass a training config with `@`, e.g. `python run_model.py @../configs/unet_v1.2/unet_v1.2.config --checkpoint ... --radar KMLB --day 20230701`. This fills in the architecture args and names the output folder after the config version. Training-only options in the config (epochs, learning rates, augmentation, class weighting) are accepted and ignored.

**Outputs**, per frame, in `<run-output-dir>/YYYYMMDD/`:

- `YYYYMMDD_YYYYMMDD_HHMM_pred_overlay.png` — predicted classes blended over the satellite image
- `YYYYMMDD_YYYYMMDD_HHMM_pred_mask.tif` — single-band `uint8` GeoTIFF (EPSG:4326) of class indices; the class names are stored in the TIFF tags (`class_0 = OCEAN`, …)

> **Running at a new radar site:** pass the same bounds you used for `DOMAIN_BOUNDS` in `sat_rad.py` via `--bbox`, or add the site to `RADAR_BOUNDS` at the top of `run_model.py`, so the GeoTIFF masks are placed correctly.
>
> ⚠️ **Domain shift:** the model was trained only on the Houston area, June–September 2022. Expect degraded skill in other climates, seasons, or coastlines — the `OCEAN`/`LAND`/`SBF` classes are especially location-dependent.

---

## Training your own model

`scripts/train.py` trains and evaluates a model in one run (train → early-stop on val mIoU → test on held-out set). Argument files live in `configs/`; pass one with `@`:

```bash
cd scripts
python train.py @../configs/dino_v3.5/dino_v3.5.config --image-root ../
```

`--image-root` is searched recursively for `*_sat.png` / `*_rad.png`, and labels are read from `--label-root` (defaults to `../sampled/dataset/labeled/` and `../unsampled/dataset/labeled/`). Since the labeled folders already contain their PNGs, pointing `--image-root` at the repo root is enough to train with `--context 0`. For `--context > 0` you need the full, un-thinned image archive for each day.

How the labeled data is split is set by two flags (summarized in `configs/readme.md`):

- `--label-usage`: `train_samp` (train on `sampled/`, val/test on `unsampled/`), `only_samp` (only `sampled/`), or `all`
- `--split-type`: `rand` (random frames) or `day` (whole days held out, avoids same-day leakage)

Each run writes to `--output-dir`: `best_model.pt`, `train_log.jsonl`, `test_report.json` (mIoU, pixel accuracy, per-class IoU), `run_summary.json`, `hyper_info.json`, split manifests, and prediction visualizations.

For repeated-seed runs, k-fold cross-validation, and summarizing results, see [`scripts/README.md`](scripts/README.md).

## Labeling new data

Labels are [LabelMe](https://github.com/wkentaro/labelme) polygon JSON files, one per image, using the eight label names above (upper case). `scripts/generate_ocean_land.py` can pre-populate new JSONs from an `OCEAN`/`LAND` template so only the cloud features need drawing, and `scripts/visualize_labels.py` renders label overlays for checking. See [`scripts/README.md`](scripts/README.md).

## Data sources

- **GOES-16 ABI** Level 2 Cloud & Moisture Imagery, CONUS sector (`ABI-L2-CMIPC`), bands 1–3 — NOAA Open Data on AWS, accessed via [goes2go](https://github.com/blaylockbk/goes2go)
- **NEXRAD Level II** — `unidata-nexrad-level2` bucket on AWS, read with [Py-ART](https://arm-doe.github.io/pyart/)

## License

MIT — see [LICENSE](LICENSE).
