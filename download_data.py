"""Download reviews (+ optional full metadata) for an Amazon category."""

import argparse
import csv
import json
import os

import pandas as pd
from datasets import DownloadMode, VerificationMode, load_dataset
from huggingface_hub import hf_hub_download

from datasets_config import HF_REPO, get_dataset

REVIEW_COLS = ["user_id", "parent_asin", "timestamp", "rating"]


def stream_reviews_to_csv(cfg, out_path, max_rows=None, force=False):
    if out_path.is_file() and out_path.stat().st_size > 1_000_000 and not force:
        print(f"using cached {out_path}")
        return

    print(f"downloading reviews {cfg.reviews_jsonl} ...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=cfg.reviews_jsonl,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, encoding="utf-8") as fin, open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=REVIEW_COLS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for line in fin:
            obj = json.loads(line)
            writer.writerow({k: obj[k] for k in REVIEW_COLS})
            rows += 1
            if rows % 500_000 == 0:
                print(f"  wrote {rows:,} rows ...")
            if max_rows and rows >= max_rows:
                break
    print(f"saved {out_path} ({rows:,} rows)")


def download_meta_hub(cfg, out_path, force=False):
    if out_path.is_file() and out_path.stat().st_size > 0 and not force:
        print(f"using cached {out_path}")
        return
    if not cfg.meta_hub:
        raise ValueError(f"{cfg.name} has no smartcat hub; use download_meta.py")

    print(f"downloading {cfg.meta_hub} ...")
    meta = load_dataset(
        cfg.meta_hub,
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df = meta["train"].to_pandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved {out_path} shape {df.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=["beauty", "electronics"])
    p.add_argument(
        "--max_review_rows",
        type=int,
        default=None,
        help="cap rows when building reviews CSV (default: full file for beauty sample cap in preprocess)",
    )
    p.add_argument("--meta", action="store_true", help="also download metadata CSV")
    args = p.parse_args()

    cfg = get_dataset(args.dataset)
    force = os.environ.get("COAST_FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes")

    max_rows = args.max_review_rows
    if max_rows is None and cfg.name == "electronics":
        max_rows = cfg.sample_nrows

    stream_reviews_to_csv(cfg, cfg.reviews_csv(), max_rows=max_rows, force=force)

    if args.meta and cfg.meta_hub:
        download_meta_hub(cfg, cfg.meta_csv(), force=force)


if __name__ == "__main__":
    main()
