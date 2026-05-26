from pathlib import Path

import numpy as np
import torch

from data import data_partition, load_item_embeddings
from evaluate import evaluate, sample_batch
from model import COAST

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def train_loop(args):
    if not (ROOT / "data" / "item_embeddings.npy").is_file():
        raise FileNotFoundError("run encode_items.py first")

    dataset = data_partition()
    user_train, _, _, usernum, itemnum = dataset
    content_emb = load_item_embeddings()
    model = COAST(itemnum, content_emb, args).to(args.device)

    for p in model.parameters():
        try:
            torch.nn.init.xavier_normal_(p.data)
        except Exception:
            pass

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    CKPT_DIR.mkdir(exist_ok=True)
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

        ckpt = CKPT_DIR / f"coast_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        print("saved", ckpt)

        model.eval()
        print("evaluating...")
        ndcg, hr = evaluate(model, dataset, args)
        print(f"epoch {epoch} test ndcg@10 {ndcg:.4f} hr@10 {hr:.4f}")
