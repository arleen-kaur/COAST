import argparse

import numpy as np
import pandas as pd

from datasets_config import get_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=["beauty", "electronics"])
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    train = pd.read_csv(cfg.train_csv())
    test = pd.read_csv(cfg.test_csv())

    top10 = train["parent_asin"].value_counts().index[:10].tolist()

    hits = 0
    ndcg_sum = 0.0
    for _, row in test.iterrows():
        item = row["parent_asin"]
        if item in top10:
            hits += 1
            r = top10.index(item) + 1
            ndcg_sum += 1.0 / np.log2(r + 1)

    n = len(test)
    print(f"top-10 popularity baseline ({cfg.name})")
    print("hit rate@10", hits / n)
    print("ndcg@10", ndcg_sum / n)
    print("users", n)


if __name__ == "__main__":
    main()
