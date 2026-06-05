
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coast.config import DATASET_CHOICES, get_dataset

SASREC_SAMPLE_DATA = ROOT / "baselines" / "SASRec.pytorch" / "python" / "data"
UNUSED_SASREC_TXT = (
    "Beauty.txt",
    "Steam.txt",
    "Video.txt",
    "ml-1m.txt",
    "wikipedia.txt",
)

def fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"

def remove_if(path: Path, dry_run: bool) -> float:
    if not path.is_file():
        return 0.0
    size = path.stat().st_size
    if dry_run:
        print(f"  would delete {path} ({fmt_mb(path)})")
    else:
        path.unlink()
        print(f"  deleted {path} ({size / (1024 * 1024):.1f} MB)")
    return size

def main():
    p = argparse.ArgumentParser(description="Free disk space after data prep")
    p.add_argument("--dataset", action="append", choices=list(DATASET_CHOICES))
    p.add_argument("--all", action="store_true", help="all configured datasets")
    p.add_argument("--sasrec_samples", action="store_true", help="remove bundled SASRec sample .txt")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    names = list(DATASET_CHOICES) if args.all else (args.dataset or ["beauty"])
    freed = 0.0

    for name in names:
        cfg = get_dataset(name)
        print(f"\n{name}:")
        if cfg.train_csv().is_file() and cfg.test_csv().is_file():
            reviews = cfg.reviews_csv()
            freed += remove_if(reviews, args.dry_run)
        else:
            print("  skip reviews CSV (splits missing — run prepare_dataset first)")

        legacy_meta = ROOT / "data" / f"{name}_meta.csv"
        if name == "beauty":
            legacy_meta = ROOT / "data" / "beauty_meta.csv"
        if cfg.meta_csv().is_file() and legacy_meta.is_file() and legacy_meta != cfg.meta_csv():
            freed += remove_if(legacy_meta, args.dry_run)

    if args.sasrec_samples or args.all:
        print("\nSASRec bundled samples:")
        for fname in UNUSED_SASREC_TXT:
            freed += remove_if(SASREC_SAMPLE_DATA / fname, args.dry_run)

    print(f"\n{'Would free' if args.dry_run else 'Freed'} ~{freed / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    main()
