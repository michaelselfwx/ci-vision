import argparse
import json
from pathlib import Path
import numpy as np
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    metric_files = sorted(args.root.glob(f"cv{args.k}_fold*/test_report.json"))
    epoch_files = sorted(args.root.glob(f"cv{args.k}_fold*/train_log.jsonl"))

    if len(metric_files) == 0:
        raise RuntimeError(f"No test_report.json files found under {args.root}")

    rows = []

    for path in metric_files:
        with open(path, "r") as f:
            row = json.load(f)

        fold_name = path.parent.name
        rows.append((fold_name, row))

    mious = np.array([row["test_miou"] for _, row in rows], dtype=float)
    accs = np.array([row["test_pixel_acc"] for _, row in rows], dtype=float)

    print("Fold results:")
    for fold_name, row in rows:
        print(
            f"{fold_name}: "
            f"mIoU={row['test_miou']:.4f}, "
            f"pixel_acc={row['test_pixel_acc']:.4f}, "
        )

    print()
    print("Overall:")
    print(f"Mean test mIoU:      {mious.mean():.4f}")
    print(f"Std test mIoU:       {mious.std(ddof=1):.4f}")
    print(f"Mean pixel accuracy: {accs.mean():.4f}")
    print(f"Std pixel accuracy:  {accs.std(ddof=1):.4f}")

    if len(mious) == 5:
        # t critical value for 95% CI with df=4
        tcrit = 2.776
    else:
        # rough fallback
        tcrit = 1.96
        print(f"Warning: Using t critical value of {tcrit} for 95% CI (not exact for df={len(mious)-1})")

    ci95 = tcrit * mious.std(ddof=1) / np.sqrt(len(mious))

    print(f"95% CI for mIoU:     [{mious.mean() - ci95:.4f}, {mious.mean() + ci95:.4f}]")

    # Per-class IoU summary
    class_names = list(rows[0][1]["per_class_iou"].keys())

    print()
    print("Per-class IoU mean across folds:")
    for cls in class_names:
        vals = []

        for _, row in rows:
            val = row["per_class_iou"][cls]
            if val is not None:
                vals.append(val)

        vals = np.array(vals, dtype=float)

        if len(vals) == 0:
            print(f"{cls:10s}: None")
        else:
            print(f"{cls:10s}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}")

    # --- Number of epochs to convergence ---
    rows = []

    for path in epoch_files:
        with open(path, "r") as f:
            row = [json.loads(line) for line in f if line.strip()]

        fold_name = path.parent.name
        rows.append((fold_name, row))

    print("\n=== Number of epochs to convergence ===")

    # average epochs to convergence across seeds
    epochs = np.array([len(row) for _, row in rows], dtype=int)

    for fold_name, row in rows:
        print(f"{fold_name}: {len(row)} epochs")

    print(f"Average epochs to convergence: {statistics.mean(epochs):.2f}")


if __name__ == "__main__":
    main()