"""Data loading: read the SASRec-style interaction file and build per-user splits."""
from collections import defaultdict

import numpy as np

from coast.config import get_dataset

# module-level dataset config so the eval/baseline scripts can set it once
_cfg = None

def set_dataset(name="beauty"):
    global _cfg
    _cfg = get_dataset(name)
    return _cfg

def _cfg_or_default():
    global _cfg
    if _cfg is None:
        _cfg = get_dataset("beauty")
    return _cfg

def _missing(cfg, path):
    return FileNotFoundError(
        f"missing {path}; run: python scripts/prepare_dataset.py --dataset {cfg.name}"
    )

def load_item_embeddings(cfg=None):
    cfg = cfg or _cfg_or_default()
    path = cfg.emb_path()
    if not path.is_file():
        raise _missing(cfg, path)
    return np.load(path)

def data_partition(cfg=None):
    cfg = cfg or _cfg_or_default()
    path = cfg.sasrec_txt()
    if not path.is_file():
        raise _missing(cfg, path)
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

    # leave-one-out: last item -> test, second-to-last -> validation, rest -> train
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
    # set of items that appear in training; used to split warm vs cold-start items
    seen = set()
    for items in user_train.values():
        seen.update(items)
    return seen
