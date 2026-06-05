#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pull", action="store_true")
    args = p.parse_args()

    if args.pull:
        subprocess.run(["git", "pull", "origin", "main"], cwd=ROOT, check=False)

    try:
        from coast.config import get_dataset
        from coast.core.model import COAST
        from coast.train.loop import train_loop

        get_dataset("beauty")
        print("imports ok")
    except Exception as e:
        print("setup failed:", e)
        sys.exit(1)

    print("try: python scripts/prepare_dataset.py --dataset beauty --device cuda --from_hub")


if __name__ == "__main__":
    main()
