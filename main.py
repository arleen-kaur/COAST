import argparse
from pathlib import Path

import torch

from data import data_partition, load_item_embeddings
from evaluate import evaluate
from model import COAST
from train import train_loop

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train", choices=["train", "evaluate", "cold_start", "warm"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--norm_first", action="store_true")
    return p.parse_args()


def load_model(args):
    _, _, _, _, itemnum = data_partition()
    content_emb = load_item_embeddings()
    model = COAST(itemnum, content_emb, args).to(args.device)
    ckpt = CKPT_DIR / f"coast_epoch{args.num_epochs}.pt"
    if not ckpt.is_file():
        ckpts = sorted(CKPT_DIR.glob("coast_epoch*.pt"))
        if not ckpts:
            raise FileNotFoundError("no checkpoint in checkpoints/ — run train first")
        ckpt = ckpts[-1]
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    return model


def main():
    args = parse_args()
    if args.mode == "train":
        train_loop(args)
        return

    model = load_model(args)
    model.eval()
    dataset = data_partition()
    cold = args.mode == "cold_start"
    warm = args.mode == "warm"
    label = args.mode
    print(f"evaluating ({label})...")
    ndcg, hr = evaluate(model, dataset, args, cold_only=cold, warm_only=warm)
    print(f"{label} ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")


if __name__ == "__main__":
    main()
