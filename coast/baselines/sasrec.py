"""SASRec baseline: load a checkpoint trained with the vendored SASRec.pytorch repo and
evaluate it with our shared warm/cold-start ranking metrics."""
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

def find_checkpoint(dataset_name, path=None):
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    ckpts = sorted((SASREC_DIR / f"{dataset_name}_default").glob("SASRec.epoch=*.pth"))
    if not ckpts:
        raise FileNotFoundError(
            f"no SASRec checkpoint for {dataset_name!r}; train it first from "
            "baselines/SASRec.pytorch/python (see run_dataset.py)"
        )
    return ckpts[-1]

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
    p.add_argument("--norm_first", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)

    ckpt = find_checkpoint(args.dataset, args.checkpoint)
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
