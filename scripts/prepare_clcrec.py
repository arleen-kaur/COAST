#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLCREC_SRC = ROOT / "baselines" / "CLCRec" / "src"
HOOK = ROOT / "baselines" / "CLCRec" / "coast_data_load.py"
PATCH_MARKER = "COAST_DATA_LOAD_HOOK"


def clone_clcrec():
    if (CLCREC_SRC / "main.py").is_file():
        print(f"CLCRec already at {CLCREC_SRC}")
        return
    CLCREC_SRC.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/iLearn-Lab/MM21-CLCRec.git", str(CLCREC_SRC)],
        check=True,
    )


def patch_dataset_py():
    dataset_py = CLCREC_SRC / "Dataset.py"
    if not dataset_py.is_file():
        raise FileNotFoundError(dataset_py)
    text = dataset_py.read_text()
    if PATCH_MARKER in text:
        return
    shutil.copy(HOOK, CLCREC_SRC / "coast_data_load.py")
    hook = f"""
    # {PATCH_MARKER}
    coast_meta = './Data/' + dataset + '/coast_meta.json'
    if os.path.isfile(coast_meta):
        from coast_data_load import data_load_coast
        return data_load_coast(dataset, has_v=has_v, has_a=has_a, has_t=has_t)
"""
    needle = "def data_load(dataset, has_v=True, has_a=True, has_t=True):"
    if needle not in text:
        raise RuntimeError("could not patch Dataset.py")
    text = text.replace(needle, needle + hook)
    dataset_py.write_text(text)
    print("patched Dataset.py for COAST exports")


def patch_modern_cuda():
    main_py = CLCREC_SRC / "main.py"
    if main_py.is_file():
        text = main_py.read_text()
        text = text.replace(
            'device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")',
            'device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")',
        )
        main_py.write_text(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", choices=["beauty", "electronics", "movielens"])
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    datasets = ["beauty", "electronics", "movielens"] if args.all else (args.dataset or ["movielens"])

    clone_clcrec()
    patch_dataset_py()
    patch_modern_cuda()

    from coast.baselines.export_clcrec import export_clcrec_data

    for ds in datasets:
        cfg_ds = __import__("coast.config", fromlist=["get_dataset"]).get_dataset(ds)
        if not cfg_ds.emb_path().is_file():
            print(f"skip {ds}: run prepare_dataset first")
            continue
        export_clcrec_data(ds)

    print("\nCLCRec ready. Train with:")
    print("  python scripts/run_clcrec.py --dataset movielens --device cuda")


if __name__ == "__main__":
    main()
