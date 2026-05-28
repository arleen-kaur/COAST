import argparse
import ast
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from datasets_config import get_dataset
from download_meta import meta_from_hub, meta_from_jsonl

MODEL_NAME = "all-MiniLM-L6-v2"


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_list_field(val):
    if pd.isna(val) or val == "":
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return s


def build_text(row, asin):
    title = "" if pd.isna(row.get("title", np.nan)) else str(row["title"])
    feats = parse_list_field(row.get("features", ""))
    desc = parse_list_field(row.get("description", ""))
    text = ". ".join(x for x in [title, feats, desc] if x).strip()
    return text if text else asin


def load_meta_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError as e:
        size_mb = path.stat().st_size / (1024 * 1024)
        raise pd.errors.ParserError(
            f"{path} looks corrupt or incomplete ({size_mb:.0f} MB on disk). "
            f"Delete it and run: python download_meta.py --dataset ... "
            f"Original error: {e}"
        ) from e


def load_item_ids(cfg):
    train = pd.read_csv(cfg.train_csv())
    test = pd.read_csv(cfg.test_csv())
    df = pd.concat([train, test], ignore_index=True)
    asins = df["parent_asin"].unique()
    asin2id = {asin: i + 1 for i, asin in enumerate(asins)}
    return asin2id


def resolve_meta_path(cfg, from_hub=False):
    if from_hub or not cfg.meta_csv().is_file():
        if cfg.meta_hub:
            meta_from_hub(cfg, cfg.meta_csv())
        else:
            meta_from_jsonl(cfg, cfg.meta_csv())
    return cfg.meta_csv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beauty", choices=["beauty", "electronics"])
    p.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--from_hub",
        action="store_true",
        help="download/filter metadata from Hugging Face (use on Colab)",
    )
    args = p.parse_args()
    cfg = get_dataset(args.dataset)
    device = pick_device(args.device)

    asin2id = load_item_ids(cfg)
    asins = [None] + sorted(asin2id, key=asin2id.get)
    n_items = len(asin2id)

    meta_path = resolve_meta_path(cfg, from_hub=args.from_hub)
    print("meta:", meta_path)
    meta = load_meta_csv(meta_path)
    meta = meta.drop_duplicates("parent_asin", keep="first").set_index("parent_asin")

    texts = []
    for asin in asins[1:]:
        if asin in meta.index:
            texts.append(build_text(meta.loc[asin], asin))
        else:
            texts.append(asin)

    print(f"encoding {n_items} items ({cfg.name}) on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    vectors = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=args.batch_size,
        device=device,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    emb = np.zeros((n_items + 1, vectors.shape[1]), dtype=np.float32)
    for asin, idx in asin2id.items():
        emb[idx] = vectors[idx - 1]

    emb_path = cfg.emb_path()
    asin_path = cfg.asin2id_path()
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    with open(asin_path, "w") as f:
        json.dump(asin2id, f)

    print(emb_path, emb.shape)
    print(asin_path, len(asin2id), "items")


if __name__ == "__main__":
    main()
