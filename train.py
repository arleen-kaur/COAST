import numpy as np
import torch

from data import data_partition, load_item_embeddings, set_dataset
from datasets_config import get_dataset
from evaluate import evaluate, sample_batch
from model import COAST


def checkpoint_path(epoch, hybrid, cfg):
    prefix = "coast_hybrid" if hybrid else "coast"
    d = cfg.checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prefix}_epoch{epoch}.pt"


def train_loop(args):
    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)

    if not cfg.emb_path().is_file():
        raise FileNotFoundError(f"run encode_items.py --dataset {cfg.name} first")

    dataset = data_partition(cfg=cfg)
    user_train, _, _, usernum, itemnum = dataset
    content_emb = load_item_embeddings(cfg)
    model = COAST(itemnum, content_emb, args).to(args.device)

    for p in model.parameters():
        try:
            torch.nn.init.xavier_normal_(p.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    if model.hybrid:
        model.id_emb.weight.data[0, :] = 0

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    variant = "hybrid" if args.hybrid else "content-only"
    print(f"training COAST ({variant}) on {cfg.name} for {args.num_epochs} epochs")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for step in range(num_batch):
            u, seq, pos, neg = sample_batch(
                user_train, usernum, itemnum, args.batch_size, args.maxlen
            )
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            idx = np.where(pos != 0)
            loss = bce(pos_logits[idx], pos_labels[idx]) + bce(neg_logits[idx], neg_labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        ckpt = checkpoint_path(epoch, args.hybrid, cfg)
        torch.save(model.state_dict(), ckpt)
        print("saved", ckpt)

        model.eval()
        print("evaluating...")
        ndcg, hr = evaluate(model, dataset, args, seed=args.seed)
        print(f"epoch {epoch} test ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")
