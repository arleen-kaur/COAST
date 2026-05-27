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
META_FALLBACK = ROOT / "beauty_data.csv"
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


def download_meta_from_hub():
    from datasets import DownloadMode, VerificationMode, load_dataset

    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("downloading metadata from Hugging Face...")
    meta = load_dataset(
        "smartcat/Amazon_Beauty_and_Personal_Care_2023",
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    df = meta["train"].to_pandas()
    df.to_csv(META_CSV, index=False)
    print("saved", META_CSV, df.shape)
    return META_CSV


def resolve_meta_path(from_hub=False):
    if from_hub:
        return download_meta_from_hub()
    if META_CSV.is_file():
        return META_CSV
    if META_FALLBACK.is_file():
        return META_FALLBACK
    raise FileNotFoundError(
        f"need {META_CSV} or {META_FALLBACK}. "
        "Run: python download_meta.py   (best on Colab instead of uploading)"
    )


def load_meta_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError as e:
        size_mb = path.stat().st_size / (1024 * 1024)
        raise pd.errors.ParserError(
            f"{path} looks corrupt or incomplete ({size_mb:.0f} MB on disk). "
            f"Colab uploads often truncate large CSVs. Delete it and run: "
            f"python download_meta.py   Original error: {e}"
        ) from e


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
    p.add_argument(
        "--from_hub",
        action="store_true",
        help="download beauty_meta.csv from Hugging Face (use on Colab)",
    )
    args = p.parse_args()
    device = pick_device(args.device)

    asin2id = load_item_ids()
    asins = [None] + sorted(asin2id, key=asin2id.get)
    n_items = len(asin2id)

    meta_path = resolve_meta_path(from_hub=args.from_hub)
    print("meta:", meta_path)
    meta = load_meta_csv(meta_path)
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
