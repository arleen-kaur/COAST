#!/usr/bin/env python3
"""Restore a COAST data bundle created by pack_data.py."""

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Restore packed COAST data on Colab")
    p.add_argument("--archive", required=True, help="path to coast_data_bundle.tar.gz")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    archive = Path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)

    print(f"Extracting {archive} -> {ROOT}")
    with tarfile.open(archive, "r:gz") as tar:
        if args.dry_run:
            for m in tar.getmembers():
                print(" ", m.name)
            return
        tar.extractall(ROOT)

    # Quick sanity
    checks = [
        ROOT / "data" / "beauty" / "item_embeddings.npy",
        ROOT / "data" / "item_embeddings.npy",
        ROOT / "data" / "electronics" / "item_embeddings.npy",
    ]
    found = [p for p in checks if p.is_file()]
    print(f"Restored. Found embeddings: {[str(p.relative_to(ROOT)) for p in found] or 'none yet'}")
    print("Skip prepare_dataset.py if splits+embeddings are present. Run train/eval directly.")


if __name__ == "__main__":
    main()
