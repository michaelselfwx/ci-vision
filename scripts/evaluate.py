import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple training seeds.")
    parser.add_argument("--config", type=str, required=True)    
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 2024, 999])
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--image-root", type=Path, default="../images")
    return parser.parse_args()


def main():
    args = parse_args()

    for seed in args.seeds:

        config_path = Path(args.config)
        model_version = config_path.parent.name
        output_dir = Path("../train_out") / model_version / f"final_seed{seed}"

        subprocess.run(
            [
                "python",
                str(args.train_script),
                f"@{args.config}",
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                "--image-root",
                str(args.image_root),
            ],
            check=True,
        )



if __name__ == "__main__":
    main()