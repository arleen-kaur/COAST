"""Download and extract MovieLens-1M."""

import argparse
import zipfile
from urllib.request import urlretrieve

from datasets_config import ML1M_URL, get_dataset


def download_movielens(cfg, force=False):
    raw_dir = cfg.movielens_raw_dir()
    ratings = cfg.ratings_dat()
    if ratings.is_file() and not force:
        print(f"using cached {ratings}")
        return

    zip_path = cfg.data_dir() / "ml-1m.zip"
    cfg.data_dir().mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file() or force:
        print(f"downloading {ML1M_URL} ...")
        urlretrieve(ML1M_URL, zip_path)
        print("saved", zip_path)

    print("extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.data_dir() / "raw")
    print("extracted to", raw_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="movielens", choices=["movielens"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = get_dataset(args.dataset)
    download_movielens(cfg, force=args.force)


if __name__ == "__main__":
    main()
