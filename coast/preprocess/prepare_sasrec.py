import argparse

import pandas as pd

from coast.config import DATASET_CHOICES, get_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_CHOICES))
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    train = pd.read_csv(cfg.train_csv())
    test = pd.read_csv(cfg.test_csv())
    df = pd.concat([train, test], ignore_index=True)

    user2id = {u: i + 1 for i, u in enumerate(df["user_id"].unique())}
    item2id = {it: i + 1 for i, it in enumerate(df["parent_asin"].unique())}

    df["user_int"] = df["user_id"].map(user2id)
    df["item_int"] = df["parent_asin"].map(item2id)
    df = df.sort_values(["user_int", "timestamp"]).reset_index(drop=True)

    out = cfg.sasrec_txt()
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["user_int", "item_int"]].to_csv(out, sep=" ", index=False, header=False)

    print(out)
    print(df["user_int"].nunique(), "users,", df["item_int"].nunique(), "items,", len(df), "lines")

if __name__ == "__main__":
    main()
