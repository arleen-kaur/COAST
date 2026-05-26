import csv
import os

import pandas as pd
from datasets import DownloadMode, VerificationMode, load_dataset
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)

META_CSV = "data/beauty_meta.csv"
REVIEWS_CSV = "data/beauty_reviews.csv"
redo = os.environ.get("COAST_FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes")

if os.path.isfile(META_CSV) and os.path.getsize(META_CSV) > 0 and not redo:
    print(f"using cached {META_CSV}")
    df_meta = pd.read_csv(META_CSV, low_memory=False)
    print(df_meta.shape)
else:
    print("downloading metadata...")
    meta = load_dataset(
        "smartcat/Amazon_Beauty_and_Personal_Care_2023",
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df_meta = meta["train"].to_pandas()
    df_meta.to_csv(META_CSV, index=False)
    print(df_meta.shape)

if os.path.isfile(REVIEWS_CSV) and os.path.getsize(REVIEWS_CSV) > 1_000_000 and not redo:
    print(f"using cached {REVIEWS_CSV}")
else:
    print("downloading reviews...")
    path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        repo_type="dataset",
        filename="raw/review_categories/Beauty_and_Personal_Care.jsonl",
    )
    reviews = load_dataset(
        "json",
        data_files=path,
        split="train",
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df_reviews = reviews.to_pandas()
    df_reviews.to_csv(
        REVIEWS_CSV,
        index=False,
        quoting=csv.QUOTE_ALL,
        doublequote=True,
    )
    print(df_reviews.shape)
    print(df_reviews.columns.tolist())
