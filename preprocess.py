"""5-core filter + leave-last-out split."""

import argparse

import pandas as pd

from datasets_config import get_dataset

COLS = ["user_id", "parent_asin", "timestamp", "rating"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=["beauty", "electronics"])
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    reviews = cfg.reviews_csv()
    if not reviews.is_file():
        raise FileNotFoundError(f"need {reviews}; run download_data.py --dataset {cfg.name}")

    print(f"loading sample ({cfg.sample_nrows:,} rows max) ...")
    df = pd.read_csv(
        reviews,
        nrows=cfg.sample_nrows,
        usecols=COLS,
        low_memory=False,
    )
    print(len(df), "rows")

    print("filtering 5-core x3 ...")
    for _ in range(3):
        uc = df["user_id"].value_counts()
        ic = df["parent_asin"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= 5].index)]
        df = df[df["parent_asin"].isin(ic[ic >= 5].index)]

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(
        "users", df["user_id"].nunique(),
        "items", df["parent_asin"].nunique(),
        "interactions", len(df),
    )

    df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
    test = df[df["rank"] == 1].drop(columns="rank")
    train = df[df["rank"] > 1].drop(columns="rank")

    cfg.data_dir().mkdir(parents=True, exist_ok=True)
    train.to_csv(cfg.train_csv(), index=False)
    test.to_csv(cfg.test_csv(), index=False)
    print("train", len(train), "test", len(test))


if __name__ == "__main__":
    main()
