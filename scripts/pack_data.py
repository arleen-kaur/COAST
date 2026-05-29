#!/usr/bin/env python3
"""Create a portable archive of preprocessed COAST data (splits, meta, embeddings).

Usage:
  python scripts/pack_data.py --dataset beauty
  python scripts/pack_data.py --dataset beauty electronics --include_checkpoints
  python scripts/pack_data.py --all

Upload the .tar.gz to Google Drive; on Colab run scripts/restore_data.py.
"""

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets_config import DATASET_CHOICES, get_dataset


def collect_paths(cfg, include_checkpoints=False):
    """Return (archive_name, path_on_disk) pairs relative to ROOT."""
    entries = []

    def add(path: Path, arcname: str | None = None):
        if path.is_file():
            entries.append((path, arcname or str(path.relative_to(ROOT))))
        elif path.is_dir() and any(path.iterdir()):
            for f in path.rglob("*"):
                if f.is_file():
                    entries.append((f, str(f.relative_to(ROOT))))

    # Splits
    for p in (cfg.train_csv(), cfg.test_csv()):
        add(p)

    # Meta + embeddings
    add(cfg.meta_csv())
    add(cfg.emb_path())
    add(cfg.asin2id_path())

    # SASRec interaction file
    add(cfg.sasrec_txt())

    # MovieLens extras
    if cfg.source == "movielens":
        add(cfg.tmdb_cache_path())
        for p in (cfg.movies_dat(), cfg.links_csv()):
            add(p)

    # Legacy beauty paths (include if present and not already added)
    if cfg.name == "beauty":
        for p in (
            ROOT / "data" / "train.csv",
            ROOT / "data" / "test.csv",
            ROOT / "data" / "item_embeddings.npy",
            ROOT / "data" / "asin2id.json",
            ROOT / "data" / "beauty_meta.csv",
        ):
            add(p)

    if include_checkpoints:
        add(cfg.checkpoint_dir())
        sasrec_ckpt = ROOT / "baselines" / "SASRec.pytorch" / "python" / f"{cfg.name}_default"
        add(sasrec_ckpt)

    # Dedupe by arcname
    seen = set()
    unique = []
    for path, arc in entries:
        if arc not in seen:
            seen.add(arc)
            unique.append((path, arc))
    return unique


def main():
    p = argparse.ArgumentParser(description="Pack preprocessed COAST data for Colab upload")
    p.add_argument("--dataset", action="append", choices=list(DATASET_CHOICES))
    p.add_argument("--all", action="store_true", help="pack beauty + electronics + movielens")
    p.add_argument("--include_checkpoints", action="store_true")
    p.add_argument("--output", default=None, help="output .tar.gz path")
    args = p.parse_args()

    names = list(DATASET_CHOICES) if args.all else (args.dataset or ["beauty"])
    if not names:
        p.error("pass --dataset NAME or --all")

    out = Path(args.output) if args.output else ROOT / "artifacts" / "coast_data_bundle.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for name in names:
        cfg = get_dataset(name)
        entries = collect_paths(cfg, args.include_checkpoints)
        if not entries:
            print(f"warn: no files found for {name}")
        all_entries.extend(entries)

    if not all_entries:
        print("Nothing to pack. Run prepare_dataset.py first.")
        sys.exit(1)

    with tarfile.open(out, "w:gz") as tar:
        for path, arc in sorted(all_entries, key=lambda x: x[1]):
            print(f"  + {arc}")
            tar.add(path, arcname=arc)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out} ({size_mb:.1f} MB, {len(all_entries)} files)")
    print("Upload to Google Drive, then on Colab:")
    print(f"  !python scripts/restore_data.py --archive /content/drive/MyDrive/{out.name}")


if __name__ == "__main__":
    main()
