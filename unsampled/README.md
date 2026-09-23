# unsampled/

A secondary labeled set: **15 satellite/radar pairs across 9 days** (summer 2022, KHGX) that were **not** picked by the cloud-cover sampler in `scripts/cloud_cover.py`.

These are used as the validation/test pool for the v1 models (`--label-usage train_samp`), giving an evaluation set drawn independently of the sampled training frames. v2 and v3 models (`--label-usage only_samp`) don't use this folder.

```
unsampled/dataset/labeled/YYYYMMDD/
├── YYYYMMDD_PPPP_TTTT_sat.png / .json
└── YYYYMMDD_PPPP_TTTT_rad.png / .json
```

Format is identical to `sampled/` — see that README and the main README.
