import argparse

import torch

from checkpointing import resolve_checkpoint
from coast_version import check_installation, print_banner
from data import data_partition, load_item_embeddings, set_dataset
from datasets_config import DATASET_CHOICES, get_dataset
from evaluate import evaluate
from model import COAST
from train import train_loop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_CHOICES))
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--content_only",
        action="store_true",
        help="COAST v1: frozen content vectors only (no ID embeddings)",
    )
    p.add_argument(
        "--checkpoint",
        default="auto",
        choices=["auto", "best", "last"],
        help="eval checkpoint: auto=best if saved else --num_epochs else latest",
    )
    p.add_argument("--no_early_stop", action="store_true", help="train all num_epochs")
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--min_epochs", type=int, default=2)
    return p.parse_args()


def load_model(args, cfg):
    content_emb = load_item_embeddings(cfg)
    args.hybrid = not args.content_only
    _, _, _, _, itemnum = data_partition(cfg=cfg)
    model = COAST(itemnum, content_emb, args).to(args.device)

    ckpt = resolve_checkpoint(
        cfg,
        hybrid=args.hybrid,
        strategy=args.checkpoint,
        num_epochs=args.num_epochs,
    )
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state)
    return model


def main():
    print_banner()
    check_installation()

    args = parse_args()
    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)
    args.hybrid = not args.content_only
    args.early_stop = not args.no_early_stop

    if args.mode == "train":
        train_loop(args)
        return

    model = load_model(args, cfg)
    model.eval()
    dataset = data_partition(cfg=cfg)
    cold = args.mode == "cold_start"
    warm = args.mode == "warm"
    label = args.mode
    variant = "hybrid" if args.hybrid else "content-only"
    print(f"evaluating ({cfg.name}, {label}, {variant}, seed={args.seed})...")
    ndcg, hr = evaluate(
        model, dataset, args, cold_only=cold, warm_only=warm, seed=args.seed
    )
    print(f"{label} ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")


if __name__ == "__main__":
    main()
