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


def _missing_data_message(cfg, path: Path, step: str) -> str:
    return (
        f"Missing {path}\n\n"
        f"Run data prep for dataset={cfg.name!r} (once per Colab session):\n"
        f"  python scripts/prepare_dataset.py --dataset {cfg.name}\n\n"
        f"Or step by step:\n"
        f"  {step}"
    )


def load_asin2id(cfg=None):
    cfg = cfg or _cfg_or_default()
    path = cfg.asin2id_path()
    if not path.is_file():
        raise FileNotFoundError(
            _missing_data_message(
                cfg,
                path,
                f"python encode_items.py --dataset {cfg.name} --device cuda --batch_size 512",
            )
        )
    with open(path) as f:
        return json.load(f)


def load_item_embeddings(cfg=None):
    cfg = cfg or _cfg_or_default()
    path = cfg.emb_path()
    if not path.is_file():
        raise FileNotFoundError(
            _missing_data_message(
                cfg,
                path,
                (
                    f"python download_meta.py --dataset {cfg.name}\n"
                    f"  python prepare_sasrec.py --dataset {cfg.name}\n"
                    f"  python encode_items.py --dataset {cfg.name} --device cuda "
                    f"--batch_size 512 --from_hub"
                ),
            )
        )
    return np.load(path)


def data_partition(fname=None, cfg=None):
    cfg = cfg or _cfg_or_default()
    fname = fname or cfg.name
    path = cfg.sasrec_txt() if fname == cfg.name else (
        ROOT / "baselines/SASRec.pytorch/python/data" / f"{fname}.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            _missing_data_message(
                cfg,
                path,
                f"python prepare_sasrec.py --dataset {cfg.name}",
            )
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
