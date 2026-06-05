#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLCREC_SRC = ROOT / "baselines" / "CLCRec" / "src"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["beauty", "electronics", "movielens", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_epoch", type=int, default=200)
    p.add_argument("--timeout", type=int, default=14400)
    args = p.parse_args()

    from coast.config import get_dataset
    from scripts.prepare_clcrec import clone_clcrec, patch_dataset_py, patch_modern_cuda

    clone_clcrec()
    patch_dataset_py()
    patch_modern_cuda()

    datasets = (
        ["beauty", "electronics", "movielens"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets:
        cfg = get_dataset(ds)
        data_path = cfg.clcrec_data_name()
        data_dir = CLCREC_SRC / "Data" / data_path
        if not (data_dir / "train.npy").is_file():
            print(f"missing {data_dir} — run: python scripts/prepare_clcrec.py --dataset {ds}")
            continue

        cmd = [
            sys.executable,
            "main.py",
            "--model_name=CLCRec",
            f"--data_path={data_path}",
            "--l_r=0.001",
            "--num_workers=0",
            f"--num_epoch={args.num_epoch}",
            "--has_v=True",
            "--has_a=False",
            "--has_t=False",
            "--reg_weight=0.001",
            "--num_neg=128",
            "--lr_lambda=0.9",
            "--num_sample=0.5",
            f"--save_file=coast_{ds}",
        ]
        if ds == "movielens":
            cmd.extend(["--reg_weight=0.1", "--lr_lambda=0.5", "--temp_value=2.0"])

        print(">>>", " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=CLCREC_SRC, check=True, timeout=args.timeout)
        except subprocess.CalledProcessError as e:
            print(f"CLCRec train failed for {ds}: {e}")
        except subprocess.TimeoutExpired:
            print(f"CLCRec train timed out for {ds}")

        from scripts.clcrec_results import write_clcrec_json

        write_clcrec_json(ds)


if __name__ == "__main__":
    main()
