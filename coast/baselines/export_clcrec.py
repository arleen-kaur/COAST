import json
from pathlib import Path

import numpy as np
import torch

from coast.config import get_dataset
from coast.config.datasets import REPO_ROOT
from coast.core.data import data_partition, load_item_embeddings, set_dataset, train_items

CLCREC_SRC = REPO_ROOT / "baselines" / "CLCRec" / "src"


def _global_item(num_user: int, coast_item_id: int) -> int:
    return num_user + (coast_item_id - 1)


def _global_user(coast_user_id: int) -> int:
    return coast_user_id - 1


def export_clcrec_data(coast_dataset: str, out_root: Path | None = None) -> Path:
    cfg = get_dataset(coast_dataset)
    set_dataset(coast_dataset)

    train, valid, test, usernum, itemnum = data_partition(cfg=cfg)
    seen = train_items(train)
    num_user = usernum
    num_item = itemnum
    num_warm = len(seen)

    data_name = cfg.clcrec_data_name()
    out_dir = (out_root or CLCREC_SRC / "Data") / data_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    user_item_train = {}
    user_item_all = {}
    for u in range(1, usernum + 1):
        gu = _global_user(u)
        user_item_train[gu] = set()
        user_item_all[gu] = set()
        for it in train[u]:
            gi = _global_item(num_user, it)
            pairs.append([gu, gi])
            user_item_train[gu].add(gi)
        for it in train[u] + valid[u] + test[u]:
            user_item_all[gu].add(_global_item(num_user, it))

    train_data = np.asarray(pairs, dtype=np.int64)

    warm = np.array(
        [_global_item(num_user, i) for i in range(1, itemnum + 1) if i in seen],
        dtype=np.int64,
    )
    cold = np.array(
        [_global_item(num_user, i) for i in range(1, itemnum + 1) if i not in seen],
        dtype=np.int64,
    )

    val_full, val_warm, val_cold = [], [], []
    test_full, test_warm, test_cold = [], [], []
    for u in range(1, usernum + 1):
        if valid[u]:
            row = np.array(
                [_global_user(u)] + [_global_item(num_user, it) for it in valid[u]],
                dtype=np.int64,
            )
            val_full.append(row)
            (val_warm if valid[u][0] in seen else val_cold).append(row)
        if test[u]:
            row = np.array(
                [_global_user(u)] + [_global_item(num_user, it) for it in test[u]],
                dtype=np.int64,
            )
            test_full.append(row)
            (test_warm if test[u][0] in seen else test_cold).append(row)

    emb = load_item_embeddings(cfg)[1:]
    feat_v = emb.astype(np.float32)

    np.save(out_dir / "train.npy", train_data)
    np.save(out_dir / "val_full.npy", np.array(val_full, dtype=object))
    np.save(out_dir / "val_warm.npy", np.array(val_warm, dtype=object))
    np.save(out_dir / "val_cold.npy", np.array(val_cold, dtype=object))
    np.save(out_dir / "test_full.npy", np.array(test_full, dtype=object))
    np.save(out_dir / "test_warm.npy", np.array(test_warm, dtype=object))
    np.save(out_dir / "test_cold.npy", np.array(test_cold, dtype=object))
    np.save(out_dir / "warm_set.npy", warm)
    np.save(out_dir / "cold_set.npy", cold)
    np.save(out_dir / "user_item_train_dict.npy", user_item_train, allow_pickle=True)
    np.save(out_dir / "user_item_all_dict.npy", user_item_all, allow_pickle=True)
    np.save(out_dir / "feat_v.npy", feat_v)

    meta = {
        "num_user": num_user,
        "num_item": num_item,
        "num_warm_item": num_warm,
        "coast_dataset": coast_dataset,
        "feat_dim": int(feat_v.shape[1]),
        "has_v": True,
        "has_a": False,
        "has_t": False,
    }
    with open(out_dir / "coast_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"exported CLCRec data -> {out_dir}")
    print(f"  users={num_user} items={num_item} warm={num_warm} cold={len(cold)}")
    return out_dir


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["beauty", "electronics", "movielens"])
    args = p.parse_args()
    export_clcrec_data(args.dataset)


if __name__ == "__main__":
    main()
