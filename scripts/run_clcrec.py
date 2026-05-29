#!/usr/bin/env python3
"""
Attempt to run official CLCRec (MM21) or record cited paper metrics.

CLCRec requires preprocessed data under baselines/CLCRec/src/Data/{amazon,movielens}/.
That data is NOT in the GitHub repo — obtain from the authors/paper supplement if reproducing.

If data is missing or training fails, writes cited Table-2 numbers to results/clcrec_{dataset}.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clcrec_results import cited_metrics, clcrec_paper_key, get_clcrec_metrics

CLCREC_SRC = ROOT / "baselines" / "CLCRec" / "src"


def clone_clcrec():
    target = CLCREC_SRC
    if (target / "main.py").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning CLCRec ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/iLearn-Lab/MM21-CLCRec.git", str(target)],
        check=True,
    )
    return target


def run_clcrec_train(data_path: str, device_ok: bool = True, timeout_s: int = 7200):
    """Run CLCRec main.py in its repo (old torch-geometric — may fail on modern Colab)."""
    data_dir = CLCREC_SRC / "Data" / data_path
    if not (data_dir / "train.npy").is_file():
        return False, f"Missing {data_dir} — CLCRec preprocessed data not installed."

    cmd = [
        sys.executable,
        "main.py",
        "--model_name=CLCRec",
        f"--data_path={data_path}",
        "--l_r=0.001",
        "--num_workers=0",
        "--num_epoch=1000",
    ]
    if data_path == "amazon":
        cmd.extend(
            [
                "--reg_weight=0.001",
                "--num_neg=512",
                "--has_v=True",
                "--lr_lambda=0.9",
                "--num_sample=0.5",
            ]
        )
    else:
        cmd.extend(
            [
                "--reg_weight=0.1",
                "--num_neg=128",
                "--has_a=True",
                "--has_t=True",
                "--has_v=True",
                "--lr_lambda=0.5",
                "--temp_value=2.0",
                "--num_sample=0.5",
            ]
        )

    print(">>>", " ".join(cmd), f"(cwd={CLCREC_SRC}, timeout={timeout_s}s)")
    try:
        subprocess.run(cmd, cwd=CLCREC_SRC, check=True, timeout=timeout_s)
        return True, "finished"
    except subprocess.TimeoutExpired:
        return False, "timeout — check partial result_*.txt in Data/"
    except subprocess.CalledProcessError as e:
        return False, f"CLCRec failed (old deps?): {e}"
    except Exception as e:
        return False, str(e)


def save_clcrec_json(coast_dataset: str, extra: dict | None = None):
    metrics = get_clcrec_metrics(coast_dataset)
    if extra:
        metrics.update(extra)
    out = ROOT / "results" / f"clcrec_{coast_dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": coast_dataset, "CLCRec": metrics}, f, indent=2)
    print(f"Wrote {out} (source={metrics.get('source')})")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["beauty", "electronics", "movielens", "all"],
        default="all",
    )
    p.add_argument("--train", action="store_true", help="attempt full CLCRec training (slow)")
    p.add_argument("--timeout", type=int, default=7200)
    args = p.parse_args()

    clone_clcrec()
    datasets = (
        ["beauty", "electronics", "movielens"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets:
        paper_key = clcrec_paper_key(ds)
        print(f"\n=== CLCRec for COAST dataset {ds} (paper data: {paper_key}) ===")

        train_note = None
        if args.train:
            ok, msg = run_clcrec_train(paper_key, timeout_s=args.timeout)
            train_note = msg
            if not ok:
                print(f"Train skipped/failed: {msg}")

        metrics = save_clcrec_json(
            ds,
            extra={"train_attempt": train_note} if train_note else None,
        )
        print(
            f"  cold NDCG@10={metrics['cold_ndcg']:.4f} "
            f"HR@10={metrics['cold_hr']:.4f} [{metrics['source']}]"
        )


if __name__ == "__main__":
    main()
