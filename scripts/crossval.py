import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run k-fold cross validation.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cv-k", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--image-root", type=Path, default="../images")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    model_version = config_path.parent.name

    for fold in range(args.cv_k):
        output_dir = Path("../train_out") / model_version / f"cv{args.cv_k}_fold{fold}"

        print(f"Running fold {fold + 1}/{args.cv_k}")
        print(f"Output: {output_dir}")

        cmd = [
            "python",
            str(args.train_script),
            f"@{args.config}",
            "--cv-k",
            str(args.cv_k),
            "--cv-fold",
            str(fold),
            "--cv-seed",
            str(args.cv_seed),
            "--seed",
            str(args.train_seed),
            "--output-dir",
            str(output_dir),
            "--image-root",
            str(args.image_root),
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()