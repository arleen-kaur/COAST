import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
META_CSV = ROOT / "data" / "beauty_meta.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV = ROOT / "data" / "test.csv"
EMB_PATH = ROOT / "data" / "item_embeddings.npy"
ASIN2ID_PATH = ROOT / "data" / "asin2id.json"
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


def load_item_ids():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    df = pd.concat([train, test], ignore_index=True)
    asins = df["parent_asin"].unique()
    asin2id = {asin: i + 1 for i, asin in enumerate(asins)}
    return asin2id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    device = pick_device(args.device)

    asin2id = load_item_ids()
    asins = [None] + sorted(asin2id, key=asin2id.get)
    n_items = len(asin2id)

    meta = pd.read_csv(META_CSV, low_memory=False)
    meta = meta.drop_duplicates("parent_asin", keep="first").set_index("parent_asin")

    texts = []
    for asin in asins[1:]:
        if asin in meta.index:
            texts.append(build_text(meta.loc[asin], asin))
        else:
            texts.append(asin)

    print("encoding", n_items, "items on", device, "...")
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

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    with open(ASIN2ID_PATH, "w") as f:
        json.dump(asin2id, f)

    print(EMB_PATH, emb.shape)
    print(ASIN2ID_PATH, len(asin2id), "items")


if __name__ == "__main__":
    main()
