
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coast.config import DATASET_CHOICES, get_dataset

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    p = argparse.ArgumentParser(description="Download metadata, build SASRec file, encode items")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CHOICES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--from_hub", action="store_true", help="fetch meta from HF in encode_items")
    p.add_argument("--skip_download", action="store_true", help="skip download_data (if splits exist)")
    args = p.parse_args()
    cfg = get_dataset(args.dataset)
    py = sys.executable

    if not args.skip_download and not cfg.train_csv().is_file():
        run([py, "-m", "coast.preprocess.download_data", "--dataset", cfg.name])
        run([py, "-m", "coast.preprocess.filter_splits", "--dataset", cfg.name])
    elif not cfg.train_csv().is_file():
        print(f"Need {cfg.train_csv()} — run download_data + filter_splits")
        sys.exit(1)
    run([py, "-m", "coast.preprocess.download_meta", "--dataset", cfg.name])

    run([py, "-m", "coast.preprocess.prepare_sasrec", "--dataset", cfg.name])

    enc = [
        py,
        "-m",
        "coast.preprocess.encode_items",
        "--dataset",
        cfg.name,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.from_hub:
        enc.append("--from_hub")
    run(enc)

    emb = cfg.emb_path()
    if emb.is_file():
        print(f"\nReady: {emb}")
    else:
        print(f"\nExpected embeddings at {emb} but file missing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
