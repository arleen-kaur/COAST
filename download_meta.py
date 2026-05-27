"""Download product metadata only (for Colab — avoids uploading a huge CSV)."""

from pathlib import Path

import pandas as pd
from datasets import DownloadMode, VerificationMode, load_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "beauty_meta.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("downloading smartcat/Amazon_Beauty_and_Personal_Care_2023 ...")
    meta = load_dataset(
        "smartcat/Amazon_Beauty_and_Personal_Care_2023",
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df = meta["train"].to_pandas()
    df.to_csv(OUT, index=False)
    print("saved", OUT, "shape", df.shape)


if __name__ == "__main__":
    main()
