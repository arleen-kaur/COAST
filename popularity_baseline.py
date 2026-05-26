from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

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
    print("top-10 popularity baseline")
    print("hit rate@10", hits / n)
    print("ndcg@10", ndcg_sum / n)
    print("users", n)


if __name__ == "__main__":
    main()
