import copy
import random
import sys

import numpy as np

from data import train_items


def random_neq(lo, hi, seen):
    t = np.random.randint(lo, hi)
    while t in seen:
        t = np.random.randint(lo, hi)
    return t


def sample_batch(user_train, usernum, itemnum, batch_size, maxlen):
    uids, seqs, pos, neg = [], [], [], []
    for _ in range(batch_size):
        uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        s = np.zeros([maxlen], dtype=np.int32)
        p = np.zeros([maxlen], dtype=np.int32)
        n = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1
        seen = set(user_train[uid])

        for item in reversed(user_train[uid][:-1]):
            s[idx] = item
            p[idx] = nxt
            n[idx] = random_neq(1, itemnum + 1, seen)
            nxt = item
            idx -= 1
            if idx == -1:
                break

        uids.append(uid)
        seqs.append(s)
        pos.append(p)
        neg.append(n)

    return np.array(uids), np.array(seqs), np.array(pos), np.array(neg)


def evaluate(
    model,
    dataset,
    args,
    cold_only=False,
    warm_only=False,
    seed=42,
    eval_split="test",
):
    """eval_split: 'test' (default) or 'valid' for early stopping."""
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

    predict_kwargs = {}
    if getattr(model, "hybrid", False):
        predict_kwargs["seen_train"] = seen_train
        # Cold target vs 100 warm random negs would always lose if only the target
        # strips ID embeddings; rank all candidates content-only instead.
        if cold_only:
            predict_kwargs["candidates_content_only"] = True

    for u in users:
        if eval_split == "valid":
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
            target = valid[u][0]
        else:
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
        if eval_split == "test" and len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        candidates = [target]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            candidates.append(t)

        preds = -model.predict(
            np.array([u]),
            np.array([seq]),
            np.array(candidates),
            **predict_kwargs,
        )[0]
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
