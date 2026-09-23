import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import statistics
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="path to the model output folder, e.g. ../train_out/dino_v2.7")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Find all test reports under seed*/cv{k}_fold*/
    metric_files = sorted(args.root.glob(f"seed*/cv{args.k}_fold*/test_report.json"))
    epoch_files = sorted(args.root.glob(f"seed*/cv{args.k}_fold*/train_log.jsonl"))

    if len(metric_files) == 0:
        raise RuntimeError(f"No test_report.json files found under {args.root}")

    # Parse (seed, fold) from paths
    pat = re.compile(r"seed(\d+)/cv\d+_fold(\d+)")
    results = {}  # (seed, fold) -> row
    for path in metric_files:
        m = pat.search(str(path))
        if not m:
            print(f"Skipping unparseable path: {path}")
            continue
        seed, fold = int(m.group(1)), int(m.group(2))
        with open(path, "r") as f:
            results[(seed, fold)] = json.load(f)

    seeds = sorted({s for s, _ in results})
    folds = sorted({f for _, f in results})

    print(f"Found {len(results)} runs across {len(seeds)} seeds and {len(folds)} folds")
    print(f"Seeds: {seeds}")
    print(f"Folds: {folds}\n")

    # Build (n_seeds, n_folds) arrays
    mious = np.full((len(seeds), len(folds)), np.nan)
    accs = np.full((len(seeds), len(folds)), np.nan)
    for i, s in enumerate(seeds):
        for j, f in enumerate(folds):
            row = results.get((s, f))
            if row is None:
                print(f"Warning: missing seed={s} fold={f}")
                continue
            mious[i, j] = row["test_miou"]
            accs[i, j] = row["test_pixel_acc"]

    # --- Per (seed, fold) table ---
    print("Per-run results (mIoU):")
    header = "seed \\ fold  " + "  ".join(f"fold{f}" for f in folds)
    print(header)
    for i, s in enumerate(seeds):
        vals = "  ".join(f"{v:.4f}" for v in mious[i])
        print(f"seed{s:<8d}   {vals}")
    print()

    # save this table to a csv file
    csv_path = args.root / f"per_run_miou.csv"
    with open(csv_path, "w") as f:
        for i, s in enumerate(seeds):
            vals = ",".join(f"{v:.4f}" for v in mious[i])
            f.write(f"{vals}\n")
    

    # --- Per-seed summary (average over folds within a seed) ---
    print("Per-seed mean mIoU (averaged over folds):")
    per_seed_miou = mious.mean(axis=1)
    for s, v in zip(seeds, per_seed_miou):
        print(f"  seed{s}: {v:.4f}")
    print()

    # --- Per-fold summary (average over seeds within a fold) ---
    print("Per-fold mean mIoU (averaged over seeds):")
    per_fold_miou = mious.mean(axis=0)
    per_fold_std = mious.std(axis=0, ddof=1)
    for f, m, s in zip(folds, per_fold_miou, per_fold_std):
        print(f"  fold{f}: mean={m:.4f}, std={s:.4f}")
    print()

    # --- Overall stats ---
    flat_miou = mious.flatten()
    flat_acc = accs.flatten()
    n = len(flat_miou)

    print("=== Overall (all seed x fold runs) ===")
    print(f"N runs:              {n}")
    print(f"Mean test mIoU:      {flat_miou.mean():.4f}")
    print(f"Std test mIoU:       {flat_miou.std(ddof=1):.4f}")
    print(f"Mean pixel accuracy: {flat_acc.mean():.4f}")
    print(f"Std pixel accuracy:  {flat_acc.std(ddof=1):.4f}")

    # 95% CI using exact t critical value for df = n-1
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci95 = tcrit * flat_miou.std(ddof=1) / np.sqrt(n)
    print(f"95% CI for mIoU:     [{flat_miou.mean() - ci95:.4f}, "
          f"{flat_miou.mean() + ci95:.4f}]  (df={n-1}, tcrit={tcrit:.3f})")

    # More conservative CI: treat per-seed means as independent samples (df = n_seeds - 1)
    if len(seeds) >= 2:
        tcrit_s = stats.t.ppf(0.975, df=len(seeds) - 1)
        ci95_s = tcrit_s * per_seed_miou.std(ddof=1) / np.sqrt(len(seeds))
        print(f"95% CI (per-seed):   [{per_seed_miou.mean() - ci95_s:.4f}, "
              f"{per_seed_miou.mean() + ci95_s:.4f}]  (df={len(seeds)-1})")
    print()

    # --- Per-class IoU summary across all runs ---
    class_names = list(next(iter(results.values()))["per_class_iou"].keys())
    print("Per-class IoU (mean and std across all seed x fold runs):")
    for cls in class_names:
        vals = []
        for row in results.values():
            v = row["per_class_iou"][cls]
            if v is not None:
                vals.append(v)
        vals = np.array(vals, dtype=float)
        if len(vals) == 0:
            print(f"  {cls:10s}: None")
        else:
            print(f"  {cls:10s}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, n={len(vals)}")
    print()

    # --- Epochs to convergence ---
    if epoch_files:
        print("=== Epochs to convergence ===")
        epoch_map = {}
        for path in epoch_files:
            m = pat.search(str(path))
            if not m:
                continue
            seed, fold = int(m.group(1)), int(m.group(2))
            with open(path, "r") as f:
                epoch_map[(seed, fold)] = sum(1 for line in f if line.strip())

        for (s, f), n_ep in sorted(epoch_map.items()):
            print(f"  seed{s} fold{f}: {n_ep} epochs")
        all_epochs = np.array(list(epoch_map.values()))
        pre_conv_epochs = all_epochs - 4  # subtract 4 to get the epoch with the best validation score (the 4th epoch is the first one after the last improvement)
        print(f"Average epochs to convergence (total runs minus 4): {pre_conv_epochs.mean():.2f} "
              f"(std={pre_conv_epochs.std(ddof=1):.2f})")


if __name__ == "__main__":
    main()