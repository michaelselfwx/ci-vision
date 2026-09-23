# configs/

Argument files for `scripts/train.py` (also usable with `run_model.py`, `evaluate.py`, `crossval.py`). Each file lists one `--flag` or value per line and is passed with `@`:

```bash
cd scripts
python train.py @../configs/dino_v3.5/dino_v3.5.config --image-root ../
```

Anything given after the `@file` on the command line overrides the file. `evaluate.py` and `crossval.py` name their output folder after the config's **parent folder** (e.g. `configs/dino_rand_train/dino_v1.5.config` → `train_out/dino_rand_train/`).

> Paths inside the configs (`--image-root ../images/`, `--output-dir ../../../../train_out/...`) point to the original author's machine layout. Override them on the command line.

## Version scheme

| Series | Frames used | Split | Flags |
|---|---|---|---|
| **v1** (e.g. `unet_v1.2`, `dino_v1.5`) | Train on `sampled/`; val/test from `unsampled/` | Random val/test assignment | `--split-type rand --label-usage train_samp` |
| **v2** (e.g. `unet_v2.3`, `dino_v2.3`) | Only `sampled/` for train/val/test | 70 / 15 / 15 **by day** | `--split-type day --label-usage only_samp` |
| **v3** (e.g. `unet_v3.3`, `dino_v3.3`) | Only `sampled/`, like v2 | Random | `--split-type rand --label-usage only_samp` |

The named folders are aliases for the representative config of each series:

| Folder | Config | Series |
|---|---|---|
| `unet_rand_train/`, `dino_rand_train/` | `unet_v1.3`, `dino_v1.5` | v1 |
| `unet_day_samp/`, `dino_day_samp/` | `unet_v2.3`, `dino_v2.3` | v2 |
| `unet_rand_samp/`, `dino_rand_samp/` | `unet_v3.3`, `dino_v3.3` | v3 |

## All configs

All use `--context 0`, elastic augmentation, 30 epochs max, early-stop patience 4.

| Config | Model | Split / labels | Notable settings |
|---|---|---|---|
| unet_v1.1 | U-Net | rand / train_samp | batch 4, lr 1e-4, expand 16 |
| unet_v1.2 | U-Net | rand / train_samp | batch 8, lr 1e-4, expand 16 |
| unet_v1.3 (`unet_rand_train`) | U-Net | rand / train_samp | batch 8, lr 1e-4, expand 16 |
| unet_v2.3 (`unet_day_samp`) | U-Net | day / only_samp | batch 8, lr 1e-4, expand 16 |
| unet_v3.3 (`unet_rand_samp`) | U-Net | rand / only_samp | batch 8, lr 1e-4, expand 16 |
| dino_v1.1 | DINOv2-base | rand / train_samp | batch 4, head-lr 7e-4, no expand |
| dino_v1.2 | DINOv2-base | rand / train_samp | batch 12, head-lr 7e-4, no expand |
| dino_v1.3 | DINOv2-base | rand / train_samp | batch 2, expand 16 |
| dino_v1.4 | DINOv2-base | rand / train_samp | batch 2, expand 32, decoder 512 |
| dino_v1.5 (`dino_rand_train`) | DINOv2-base | rand / train_samp | batch 2, expand 16 |
| dino_v1.6 | DINOv2-**large** | rand / only_samp | batch 2, expand 16 |
| dino_v1.7 | DINOv2-base | rand / train_samp | batch 2, class weighting `inverse` |
| dino_v2.3 (`dino_day_samp`) | DINOv2-base | day / only_samp | batch 8, expand 16 |
| dino_v2.4 | DINOv2-base | day / only_samp | batch 8, decoder 512 |
| dino_v2.5 | DINOv2-base | day / only_samp | batch 2 |
| dino_v2.6 | DINOv2-**large** | day / only_samp | batch 8, expand 8 |
| dino_v2.7 | DINOv2-base | day / only_samp | batch 8, expand 16 |
| dino_v3.3 (`dino_rand_samp`) | DINOv2-base | rand / only_samp | batch 8, expand 16 |
| dino_v3.4 | DINOv2-base | rand / only_samp | class weighting `inverse` |
| dino_v3.5 | DINOv2-base | rand / only_samp | class weighting `sqrt-inverse` |

DINOv2 configs use backbone-lr 1e-5, head-lr 5e-4, weight decay 2e-4, decoder dropout 0.1, and freeze the backbone for 1 epoch unless noted.

## Notes

- `dino_v1.5` and `dino_v1.5_2` were slightly different configurations, using DINOv2-large and DINOv2-base respectively (base still gave better results with the updated script).
- After v3.3, all models for both architectures use the updated training script (no longer denoted by `_2`).
