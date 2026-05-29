#!/usr/bin/env python3
"""Content-similarity cold-start baseline (no training)."""

import argparse
import copy
import random
import sys

import numpy as np

from data import data_partition, load_item_embeddings, set_dataset, train_items
from datasets_config import DATASET_CHOICES, get_dataset


def content_scores(seq, candidates, content_emb):
    """Score candidates by cosine sim to mean content vector of non-zero history."""
    idx = seq[seq != 0]
    if len(idx) == 0:
        anchor = content_emb[candidates[0]]
    else:
        anchor = content_emb[idx].mean(axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-8)
    cands = content_emb[candidates]
    cands = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-8)
    return cands @ anchor


def evaluate_content(dataset, content_emb, args, cold_only=False, warm_only=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
    seen_train = train_items(train)

    ndcg, ht = 0.0, 0.0
    n_users = 0.0
    users = (
        random.sample(range(1, usernum + 1), 10000)
        if usernum > 10000
        else range(1, usernum + 1)
    )

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        target = test[u][0]
        is_cold = target not in seen_train
        if cold_only and not is_cold:
            continue
        if warm_only and is_cold:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        candidates = np.array([target], dtype=np.int32)
        extra = []
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            extra.append(t)
        candidates = np.concatenate([candidates, np.array(extra, dtype=np.int32)])

        preds = -content_scores(seq, candidates, content_emb)
        rank = preds.argsort().argsort()[0].item()
        n_users += 1
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1
        if int(n_users) % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    if n_users == 0:
        return 0.0, 0.0
    return ndcg / n_users, ht / n_users


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_CHOICES))
    p.add_argument("--mode", default="cold_start", choices=["evaluate", "warm", "cold_start"])
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)

    if not cfg.emb_path().is_file():
        raise FileNotFoundError(f"run encode_items.py --dataset {cfg.name} first")

    content_emb = load_item_embeddings(cfg)
    dataset = data_partition(cfg=cfg)
    cold = args.mode == "cold_start"
    warm = args.mode == "warm"

    print(f"content-similarity baseline ({cfg.name}, {args.mode}, seed={args.seed}) ...")
    ndcg, hr = evaluate_content(
        dataset, content_emb, args, cold_only=cold, warm_only=warm, seed=args.seed
    )
    print(f"{args.mode} ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")


if __name__ == "__main__":
    main()
