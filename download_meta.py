"""Download/filter product metadata for train+test items (Colab-friendly)."""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import DownloadMode, VerificationMode, load_dataset
from huggingface_hub import hf_hub_download

from datasets_config import DATASET_CHOICES, HF_REPO, get_dataset

META_COLS = [
    "parent_asin",
    "title",
    "features",
    "description",
    "main_category",
    "average_rating",
    "rating_number",
]


def needed_asins(cfg):
    train = pd.read_csv(cfg.train_csv(), usecols=["parent_asin"])
    test = pd.read_csv(cfg.test_csv(), usecols=["parent_asin"])
    return set(train["parent_asin"]).union(test["parent_asin"])


def meta_from_hub(cfg, out_path):
    print(f"downloading {cfg.meta_hub} ...")
    meta = load_dataset(
        cfg.meta_hub,
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df = meta["train"].to_pandas()
    needed = needed_asins(cfg)
    df = df[df["parent_asin"].isin(needed)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("saved", out_path, "shape", df.shape)


def meta_from_jsonl(cfg, out_path):
    needed = needed_asins(cfg)
    print(f"filtering {cfg.meta_jsonl} for {len(needed):,} ASINs ...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=cfg.meta_jsonl,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("parent_asin") not in needed:
                continue
            row = {k: obj.get(k) for k in META_COLS}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("saved", out_path, "shape", df.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_CHOICES))
    p.add_argument(
        "--movies_only",
        action="store_true",
        help="MovieLens: use movies.dat only (no TMDB API)",
    )
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    if not cfg.train_csv().is_file():
        raise FileNotFoundError(f"run preprocess.py --dataset {cfg.name} first")

    out = cfg.meta_csv()
    if cfg.source == "movielens":
        from fetch_tmdb import build_meta_without_tmdb, build_movielens_meta
        import os

        if args.movies_only or not os.environ.get("TMDB_API_KEY"):
            if not args.movies_only:
                print("TMDB_API_KEY not set — using movies.dat only")
            build_meta_without_tmdb(cfg, out)
        else:
            build_movielens_meta(cfg, out)
        return

    if cfg.meta_hub:
        meta_from_hub(cfg, out)
    else:
        meta_from_jsonl(cfg, out)


if __name__ == "__main__":
    main()
