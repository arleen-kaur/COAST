#!/usr/bin/env python3
"""Evaluate SASRec with the same warm/cold protocol as COAST."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SASREC_DIR = ROOT / "baselines" / "SASRec.pytorch" / "python"

from data import data_partition
from evaluate import evaluate

sys.path.insert(0, str(SASREC_DIR))
from model import SASRec  # noqa: E402


def find_checkpoint(path=None, hidden_units=50, maxlen=50, num_blocks=2, num_heads=1, lr=0.001):
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    folder = SASREC_DIR / "beauty_default"
    if not folder.is_dir():
        folder = SASREC_DIR / "default"

    pattern = (
        f"SASRec.epoch=*."
        f"lr={lr}."
        f"layer={num_blocks}."
        f"head={num_heads}."
        f"hidden={hidden_units}."
        f"maxlen={maxlen}.pth"
    )
    matches = sorted(folder.glob(pattern))
    if not matches:
        matches = sorted(folder.glob("SASRec.epoch=*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"no SASRec checkpoint in {folder}. "
            "Train from baselines/SASRec.pytorch/python first."
        )
    return matches[-1]


def main():
    p = argparse.ArgumentParser()
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

    ckpt = find_checkpoint(
        args.checkpoint,
        hidden_units=args.hidden_units,
        maxlen=args.maxlen,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        lr=args.lr,
    )
    print("checkpoint:", ckpt)

    dataset = data_partition()
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

    model = SASRec(usernum, itemnum, sasrec_args).to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()

    cold = args.mode == "cold_start"
    warm = args.mode == "warm"
    print(f"evaluating SASRec ({args.mode}, seed={args.seed})...")
    ndcg, hr = evaluate(
        model, dataset, sasrec_args, cold_only=cold, warm_only=warm, seed=args.seed
    )
    print(f"{args.mode} ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")


if __name__ == "__main__":
    main()
