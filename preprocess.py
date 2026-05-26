from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REVIEWS_CSV = ROOT / "data" / "beauty_reviews.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV = ROOT / "data" / "test.csv"

SAMPLE_NROWS = 2_000_000
COLS = ["user_id", "parent_asin", "timestamp", "rating"]


def main():
    if not REVIEWS_CSV.is_file():
        raise FileNotFoundError("need data/beauty_reviews.csv from download_data.py")

    print("loading sample...")
    df = pd.read_csv(
        REVIEWS_CSV,
        nrows=SAMPLE_NROWS,
        usecols=COLS,
        low_memory=False,
    )
    print(len(df), "rows")

    print("filtering 5-core x3...")
    for _ in range(3):
        uc = df["user_id"].value_counts()
        ic = df["parent_asin"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= 5].index)]
        df = df[df["parent_asin"].isin(ic[ic >= 5].index)]

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print("users", df["user_id"].nunique(), "items", df["parent_asin"].nunique(), "interactions", len(df))

    df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
    test = df[df["rank"] == 1].drop(columns="rank")
    train = df[df["rank"] > 1].drop(columns="rank")

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print("train", len(train), "test", len(test))


if __name__ == "__main__":
    main()
