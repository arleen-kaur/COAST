
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from coast.config import DATASET_CHOICES, get_dataset
from coast.core.data import data_partition, set_dataset
from coast.core.evaluate import evaluate
from coast.config.datasets import REPO_ROOT

SASREC_DIR = REPO_ROOT / "baselines" / "SASRec.pytorch" / "python"

def find_checkpoint(
    dataset_name,
    path=None,
    hidden_units=50,
    maxlen=50,
    num_blocks=2,
    num_heads=1,
    lr=0.001,
):
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    preferred = (
        f"SASRec.epoch=*."
        f"lr={lr}."
        f"layer={num_blocks}."
        f"head={num_heads}."
        f"hidden={hidden_units}."
        f"maxlen={maxlen}.pth"
    )

    search_dirs = []
    for name in (f"{dataset_name}_default", "default", f"{dataset_name}_training"):
        d = SASREC_DIR / name
        if d.is_dir():
            search_dirs.append(d)

    matches = []
    for d in search_dirs:
        matches.extend(sorted(d.glob(preferred)))
    if not matches:
        for d in search_dirs:
            matches.extend(sorted(d.glob("SASRec.epoch=*.pth")))
    if not matches:
        matches = sorted(SASREC_DIR.rglob("SASRec.epoch=*.pth"))

    if not matches:
        raise FileNotFoundError(
            f"No SASRec checkpoint for dataset={dataset_name!r}.\n\n"
            "Train once (from baselines/SASRec.pytorch/python):\n"
            f"  python main.py --dataset={dataset_name} --train_dir=default --maxlen=50 "
            "--device cuda --num_epochs=20 --batch_size=512 "
            "--hidden_units=50 --num_blocks=2 --num_heads=1 --lr=0.001\n\n"
            f"Ensure data/{dataset_name}.txt exists "
            f"(python -m coast.preprocess.prepare_sasrec --dataset {dataset_name})."
        )

    return matches[-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_CHOICES))
    p.add_argument("--mode", default="evaluate", choices=["evaluate", "warm", "cold_start"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=50)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--norm_first", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)

    ckpt = find_checkpoint(
        args.dataset,
        args.checkpoint,
        hidden_units=args.hidden_units,
        maxlen=args.maxlen,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        lr=args.lr,
    )
    print("checkpoint:", ckpt)

    dataset = data_partition(cfg=cfg)
    _, _, _, usernum, itemnum = dataset

    sasrec_args = argparse.Namespace(
        device=args.device,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        norm_first=args.norm_first,
    )

    sys.path.insert(0, str(SASREC_DIR))
    from model import SASRec

    model = SASRec(usernum, itemnum, sasrec_args).to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()

    cold = args.mode == "cold_start"
    warm = args.mode == "warm"
    print(f"evaluating SASRec ({cfg.name}, {args.mode}, seed={args.seed})...")
    ndcg, hr = evaluate(
        model, dataset, sasrec_args, cold_only=cold, warm_only=warm, seed=args.seed
    )
    print(f"{args.mode} ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")

if __name__ == "__main__":
    main()
