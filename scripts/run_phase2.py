#!/usr/bin/env python3
"""Run Phase 2 experiments: Beauty + Electronics + MovieLens."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["beauty", "electronics", "movielens"],
        choices=["beauty", "electronics", "movielens"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--phase", default="all", choices=["prep", "train", "eval", "baselines", "all"])
    p.add_argument("--movies_only", action="store_true", help="MovieLens: skip TMDB API")
    p.add_argument("--skip_download", action="store_true")
    args = p.parse_args()

    py = sys.executable
    for ds in args.datasets:
        cmd = [
            py,
            "scripts/run_dataset.py",
            "--dataset",
            ds,
            "--phase",
            args.phase,
            "--device",
            args.device,
        ]
        if args.movies_only:
            cmd.append("--movies_only")
        if args.skip_download:
            cmd.append("--skip_download")
        print("\n" + "=" * 60)
        print(f"DATASET: {ds}")
        print("=" * 60)
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nPhase 2 complete. Results in results/*.json")
    print("CLCRec comparison: python eval_clcrec.py --dataset beauty")


if __name__ == "__main__":
    main()
