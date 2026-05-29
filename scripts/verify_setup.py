#!/usr/bin/env python3
"""Verify COAST install matches VERSION and required features. Run after git clone/pull."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="Verify COAST installation")
    p.add_argument(
        "--pull",
        action="store_true",
        help="run git pull origin main before checks (Colab)",
    )
    args = p.parse_args()

    if args.pull:
        print("git pull origin main ...")
        subprocess.run(["git", "pull", "origin", "main"], cwd=ROOT, check=False)

    from coast_version import COAST_VERSION, check_installation

    version_file = (ROOT / "VERSION").read_text().strip()
    print(f"COAST VERSION file: {version_file}")
    print(f"git HEAD: {git_head()}")

    main_src = (ROOT / "main.py").read_text()
    checks = {
        "checkpoint auto/best/last": "--checkpoint" in main_src and "auto" in main_src,
        "early stopping (train)": hasattr(
            __import__("train", fromlist=["train"]), "best_checkpoint_path"
        ),
        "movielens dataset": "movielens" in (ROOT / "datasets_config.py").read_text(),
    }
    failed = [k for k, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    try:
        check_installation()
    except SystemExit as e:
        print(e)
        failed.append("check_installation")

    if failed:
        print("\nSetup incomplete. Run: git pull origin main")
        sys.exit(1)

    print(f"\nCOAST v{COAST_VERSION} is ready.")
    print("Eval example (auto-loads best checkpoint if trained with early stopping):")
    print("  python main.py --dataset beauty --mode warm --device cuda --seed 42")


if __name__ == "__main__":
    main()
