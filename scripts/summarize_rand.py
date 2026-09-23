import json
import statistics
from collections import defaultdict
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True, help="Model output folder containing final_seed*/ runs, e.g. ../train_out/unet_v1.2")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Seeds to summarize. Default: every final_seed<N>/ folder found under --root")
args = parser.parse_args()

base = args.root
if args.seeds is not None:
    seeds = args.seeds
else:
    seeds = sorted(
        int(p.name[len("final_seed"):])
        for p in base.glob("final_seed*")
        if p.is_dir() and p.name[len("final_seed"):].isdigit()
    )
if not seeds:
    raise RuntimeError(f"No final_seed*/ folders found under {base}")
print(f"Summarizing seeds: {seeds}\n")

mious = []
per_class = defaultdict(list)  # class_name -> list of IoUs across seeds

for s in seeds:
    report_path = base / f"final_seed{s}" / "test_report.json"
    if not report_path.exists():
        print(f"[warn] missing: {report_path}")
        continue

    with report_path.open() as f:
        report = json.load(f)

    miou = report["test_miou"]  # correct key from train.py
    mious.append(miou)
    print(f"Seed {s}: mIoU = {miou:.4f}")

    for cls_name, iou in report["per_class_iou"].items():
        if iou is not None:  # train.py writes None if the class was absent
            per_class[cls_name].append(iou)

# ---- Overall mIoU stats ----
print("\n=== Overall mIoU ===")
print(f"Mean:  {statistics.mean(mious):.4f}")
print(f"Stdev: {statistics.stdev(mious):.4f}")
print(f"Min:   {min(mious):.4f}")
print(f"Max:   {max(mious):.4f}")

# ---- Overall Pixel Accuracy stats ----
print("\n=== Overall Pixel Accuracy ===")
pixel_accs = []
for s in seeds:
    report_path = base / f"final_seed{s}" / "test_report.json"
    if not report_path.exists():
        continue

    with report_path.open() as f:
        report = json.load(f)

    pixel_acc = report["test_pixel_acc"]  # correct key from train.py
    pixel_accs.append(pixel_acc)
    print(f"Seed {s}: Pixel Accuracy = {pixel_acc:.4f}")

print(f"Mean:  {statistics.mean(pixel_accs):.4f}")
print(f"Stdev: {statistics.stdev(pixel_accs):.4f}")
print(f"Min:   {min(pixel_accs):.4f}")
print(f"Max:   {max(pixel_accs):.4f}")

# ---- Per-class IoU stats ----
print("\n=== Per-class IoU (across seeds) ===")
print(f"{'Class':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>4}")
print("-" * 50)
for cls_name, ious in sorted(per_class.items()):
    n = len(ious)
    mean = statistics.mean(ious)
    std = statistics.stdev(ious) if n > 1 else 0.0
    print(f"{cls_name:<10} {mean:>8.4f} {std:>8.4f} {min(ious):>8.4f} {max(ious):>8.4f} {n:>4}")


# --- Number of epochs to convergence ---
print("\n=== Number of epochs to convergence ===")
# average epochs to convergence across seeds
num_epochs_list = []
for s in seeds:
    train_log_path = base / f"final_seed{s}" / "train_log.jsonl"
    if not train_log_path.exists():
        continue
    with train_log_path.open() as f:
        num_epochs = sum(1 for _ in f)
    print(f"Seed {s}: {num_epochs} epochs")
    num_epochs_list.append(num_epochs)

print(f"Average epochs to convergence: {statistics.mean(num_epochs_list):.2f}")