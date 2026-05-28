import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets_config import get_dataset

ROOT = Path(__file__).resolve().parent
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


def load_asin2id(cfg=None):
    cfg = cfg or _cfg_or_default()
    with open(cfg.asin2id_path()) as f:
        return json.load(f)


def load_item_embeddings(cfg=None):
    cfg = cfg or _cfg_or_default()
    return np.load(cfg.emb_path())


def data_partition(fname=None, cfg=None):
    cfg = cfg or _cfg_or_default()
    fname = fname or cfg.name
    path = cfg.sasrec_txt() if fname == cfg.name else (
        ROOT / "baselines/SASRec.pytorch/python/data" / f"{fname}.txt"
    )
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
