# scripts/

All scripts use paths relative to this folder — run them from here (`cd scripts`). Several scripts have settings hard-coded at the top of the file (marked `USER SETTINGS` or as module constants) rather than command-line flags; check those before running.

## Overview

| Script | Stage | What it does |
|---|---|---|
| `bounding_box.py` | Data | Computes a 200 km × 200 km lat/lon box around a radar site |
| `sat_rad.py` | Data | Downloads & pairs GOES visible + NEXRAD reflectivity, saves 512×448 PNGs |
| `image_org.py` | Data | Copies images from `YYYYMMDD/sat/` and `YYYYMMDD/rad/` into the day folder root (for labeling) |
| `cloud_cover.py` | Sampling | Computes per-image cloud-cover %, bins it, and draws a stratified sample of frames to label |
| `generate_ocean_land.py` | Labeling | Creates starter LabelMe JSONs from an `OCEAN`/`LAND` template |
| `visualize_labels.py` | Labeling | Draws LabelMe labels over images for QA |
| `train.py` | Model | Trains + evaluates U-Net or DINOv2 segmentation models (also holds the model/dataset code) |
| `run_model.py` | Model | Runs a trained checkpoint on every frame of the given day(s) |
| `tune.py` | Model | Optuna hyperparameter search (currently U-Net learning rate) |
| `evaluate.py` | Evaluation | Trains the same config with several seeds |
| `crossval.py` | Evaluation | k-fold cross-validation for one config |
| `summarize_rand.py` | Evaluation | Mean/std of test metrics across seed runs |
| `summarize_cv.py` | Evaluation | Mean/std/95% CI of test metrics across CV folds |
| `summarize_multi_cv.py` | Evaluation | Same as above for multiple seeds × folds |

---

## Data acquisition

### `bounding_box.py`
```bash
python bounding_box.py
```
Edit `center_lat`, `center_lon` (and side length, default 200 km) in `__main__`. Prints `(min_lon, max_lon, min_lat, max_lat)` to paste into `DOMAIN_BOUNDS` in `sat_rad.py`.

### `sat_rad.py`
```bash
python sat_rad.py
```
Settings (top of file / `__main__`): `RADAR_STATION`, `GOES_SATELLITE`, `DOMAIN_BOUNDS`, `start_date`, `end_date`, the 14:00–23:59 UTC daily window, and `max_workers`.

- Satellite: GOES ABI-L2-CMIPC bands 1, 2, 3 → gamma-corrected "true-green" RGB, resampled (nearest neighbor) to the domain grid. Raw NetCDF is cached in `../../data/`.
- Radar: NEXRAD Level II, sweep index 1, reflectivity ≥ 8 dBZ (range 8–65), `HomeyerRainbow` colormap, no axes/borders.
- Output: `../other_images/<RADAR_STATION>/YYYYMMDD/{sat,rad}/YYYYMMDD_PPPP_TTTT_{sat,rad}.png`

### `image_org.py`
Set `BASE_DIR` at the top, then `python image_org.py`. Only needed if you want every image of a day in one folder (e.g. to open in LabelMe). Training/inference search recursively, so they don't need this.

## Sampling & labeling

### `cloud_cover.py`
```bash
python cloud_cover.py [--date YYYYMMDD] [--force-recompute]
```
Thresholds the satellite image (`WHITE_THRESHOLD`) to estimate cloud-cover %, bins frames by `BIN_EDGES`, and randomly samples `SAMPLE_PERCENT` of each bin (`RANDOM_SEED = 10` was used for the released `sampled/` set). `SKIP_DATES` / `TAMU_DATES` control which days are considered. Writes a CSV to `OUTPUT_CSV` and sample copies to `OUTPUT_CLUSTER_DIR`.

### `generate_ocean_land.py`
Set `template`, `image_dir`, and `output_dir` in `__main__`, then run. For each image it copies the template's shapes into a new LabelMe JSON (skips existing JSONs). Since the domain is fixed, the coastline polygons can be reused across all frames of a site.

### `visualize_labels.py`
```bash
python visualize_labels.py --image-dir ../sampled/dataset/labeled/20220610 --output-dir ../label_viz
```

## Modeling

### `train.py`
```bash
python train.py @../configs/dino_v3.5/dino_v3.5.config --image-root ../
python train.py --model unet --image-root ../ --epochs 30 --batch-size 8 --lr 1e-4 --aug-elastic --aug-expand 16
```
Run `python train.py --help` for the full list. Most-used flags:

| Flag | Notes |
|---|---|
| `--model {unet,dino}` | Architecture |
| `--image-root`, `--label-root` | Where to find PNGs / LabelMe JSONs (searched recursively; `--label-root` can repeat) |
| `--context N` | Adds N frames before and after the center frame as extra input channels (6 channels per frame) |
| `--split-type {rand,day}`, `--label-usage {all,train_samp,only_samp}` | How data is split — see `../configs/readme.md` |
| `--aug-elastic`, `--aug-expand K`, `--aug-hflip/vflip/rotdeg` | Augmentation (`--aug-expand 8+` uses all rotation/flip combos) |
| `--dino-name`, `--backbone-lr`, `--head-lr`, `--freeze-backbone-epochs` | DINOv2 settings |
| `--class-weighting {none,inverse,sqrt-inverse}` | Loss weighting by class pixel frequency |
| `--cv-k`, `--cv-fold` | k-fold CV (normally driven by `crossval.py`) |
| `--final-train` | Train on all data with no val/test split; saves `final_model.pt` |

Loss is cross-entropy + Jaccard (IoU). Early stopping monitors validation mIoU.

Outputs in `--output-dir` (supports `{model}` and `{date:...}` placeholders):
`best_model.pt`, `train_log.jsonl`, `test_report.json`, `run_summary.json`, `hyper_info.json`, `manifest_{train,val,test}.jsonl`, `visualizations/`.

### `run_model.py`
See the [main README](../README.md#3-run-inference) for a full walkthrough.
```bash
python run_model.py --model unet --checkpoint ../train_out/unet_v1.2/final_seed42/best_model.pt \
  --radar KHGX --day 20220610
```
Reads frames from `--input-root` (default `../other_images/<RADAR>/`), writes `*_pred_overlay.png` and georeferenced `*_pred_mask.tif` to `--run-output-dir/YYYYMMDD/`. The GeoTIFF bounds come from `--bbox`, or from `RADAR_BOUNDS` at the top of the file for known sites (KHGX, KMLB, KJAX).

### `tune.py`
```bash
python tune.py
```
Search space and fixed args are in `objective()`. Results go to `tune.db` (SQLite, resumable) and `tune_trials_unet_lr.txt`.

## Evaluation

```bash
# 5 seeds -> ../train_out/<version>/final_seed<seed>/
python evaluate.py --config ../configs/unet_v1.2/unet_v1.2.config --image-root ../
python summarize_rand.py --root ../train_out/unet_v1.2

# 5-fold CV -> ../train_out/<version>/cv5_fold<k>/
python crossval.py --config ../configs/dino_v3.5/dino_v3.5.config --cv-k 5 --image-root ../
python summarize_cv.py --root ../train_out/dino_v3.5 --k 5

# multiple seeds x folds, laid out as <root>/seed*/cv5_fold*/
python summarize_multi_cv.py --root ../train_out/dino_v2.7 --k 5
```
`<version>` is taken from the config's parent folder name. `summarize_rand.py` summarizes every `final_seed<N>/` folder it finds under `--root`; pass `--seeds 42 123 ...` to restrict it to specific seeds.
