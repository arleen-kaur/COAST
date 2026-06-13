#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd, check=True):
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=check)


def main():
    p = argparse.ArgumentParser(description="Run the full COAST pipeline on each dataset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--datasets", nargs="+", default=["beauty", "electronics"])
    p.add_argument("--phase", default="all", choices=["prep", "train", "eval", "baselines", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablations", action="store_true", help="include COAST content-only")
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
            "--seed",
            str(args.seed),
        ]
        if args.ablations:
            cmd.append("--ablations")
        if args.skip_download:
            cmd.append("--skip_download")
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            print(f"warning: {ds} failed, continuing")

    run([py, "scripts/generate_report.py"], check=False)


if __name__ == "__main__":
    main()
