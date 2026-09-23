# sampled/

The primary labeled dataset: **94 satellite/radar pairs across 51 days** (June–September 2022, KHGX / Houston, 14–24 UTC).

Frames were chosen with `scripts/cloud_cover.py`: every candidate frame's satellite cloud-cover % was computed, frames were binned by cloud cover, and ~1% of each bin was randomly sampled (seed 10). This keeps clear, partly cloudy, and overcast scenes all represented in the labels.

```
sampled/dataset/
├── labeled/YYYYMMDD/
│   ├── YYYYMMDD_PPPP_TTTT_sat.png    # GOES visible "true-green" composite, 512×448
│   ├── YYYYMMDD_PPPP_TTTT_sat.json   # LabelMe polygons for that frame
│   ├── YYYYMMDD_PPPP_TTTT_rad.png    # NEXRAD 0.5° reflectivity, 512×448
│   └── YYYYMMDD_PPPP_TTTT_rad.json   # same polygons, attached to the radar image
└── visual/YYYYMMDD/
    └── *_labeled.png                 # label overlays for a subset of days (for viewing only; not used in training)
```

See the main README for the file-naming convention and class list. In training, this folder is selected by `--label-usage train_samp` (as the training set) or `only_samp` (as train/val/test).
