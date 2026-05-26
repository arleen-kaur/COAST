import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "data" / "item_embeddings.npy"
ASIN2ID_PATH = ROOT / "data" / "asin2id.json"


def load_asin2id():
    with open(ASIN2ID_PATH) as f:
        return json.load(f)


def load_item_embeddings():
    return np.load(EMB_PATH)


def data_partition(fname="beauty"):
    path = ROOT / "baselines/SASRec.pytorch/python/data" / f"{fname}.txt"
    usernum = 0
    itemnum = 0
    user_items = defaultdict(list)

    with open(path) as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u, i = int(u), int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_items[u].append(i)

    user_train, user_valid, user_test = {}, {}, {}
    for user, items in user_items.items():
        if len(items) < 4:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return user_train, user_valid, user_test, usernum, itemnum


def train_items(user_train):
    seen = set()
    for items in user_train.values():
        seen.update(items)
    return seen
