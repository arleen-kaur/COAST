import numpy as np
import torch

from coast.config import get_dataset
from coast.core.data import data_partition, load_item_embeddings, set_dataset
from coast.core.evaluate import evaluate, sample_batch
from coast.core.model import COAST


def checkpoint_path(epoch, hybrid, cfg):
    prefix = "coast_hybrid" if hybrid else "coast"
    d = cfg.checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prefix}_epoch{epoch}.pt"


def best_checkpoint_path(hybrid, cfg):
    d = cfg.checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / cfg.best_checkpoint_name(hybrid)


def train_loop(args):
    cfg = get_dataset(args.dataset)
    set_dataset(args.dataset)

    if not cfg.emb_path().is_file():
        raise FileNotFoundError(
            f"run: python -m coast.preprocess.encode_items --dataset {cfg.name} --device cuda"
        )

    dataset = data_partition(cfg=cfg)
    user_train, _, _, usernum, itemnum = dataset
    content_emb = load_item_embeddings(cfg)
    model = COAST(itemnum, content_emb, args).to(args.device)

    for p in model.parameters():
        try:
            torch.nn.init.xavier_normal_(p.data)
        except ValueError:
            pass

    model.pos_emb.weight.data[0, :] = 0
    if model.hybrid:
        model.id_emb.weight.data[0, :] = 0

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    early_stop = getattr(args, "early_stop", True)
    patience = getattr(args, "early_stop_patience", 4)
    best_val_ndcg = -1.0
    patience_left = patience
    best_epoch = 0

    print(f"training on {cfg.name}, epochs={args.num_epochs}, device={args.device}")

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

        torch.save(model.state_dict(), checkpoint_path(epoch, args.hybrid, cfg))

        model.eval()
        val_ndcg, val_hr = evaluate(model, dataset, args, seed=args.seed, eval_split="valid")
        print(f"epoch {epoch} valid ndcg@10 {val_ndcg:.4f} hr@10 {val_hr:.4f}")

        # keep the checkpoint with the best validation NDCG; stop once it stalls
        if early_stop:
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                patience_left = patience
                torch.save(model.state_dict(), best_checkpoint_path(args.hybrid, cfg))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"early stop at epoch {epoch}, best was {best_epoch}")
                    break
