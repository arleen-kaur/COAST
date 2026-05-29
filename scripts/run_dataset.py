#!/usr/bin/env python3
"""
Run full COAST experiment suite for one dataset: prep → train → eval → baselines.

Examples:
  python scripts/run_dataset.py --dataset beauty --phase all --device cuda
  python scripts/run_dataset.py --dataset movielens --phase prep --device cuda
  python scripts/run_dataset.py --dataset electronics --phase eval --device cuda
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets_config import DATASET_CHOICES, get_dataset

SASREC_DIR = ROOT / "baselines" / "SASRec.pytorch" / "python"
METRIC_RE = re.compile(r"ndcg@10\s+([\d.]+)\s+hr@10\s+([\d.]+)", re.I)


def run(cmd, cwd=None):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def capture_metrics(cmd, cwd=None):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    out = subprocess.run(
        cmd, cwd=cwd or ROOT, check=True, capture_output=True, text=True
    )
    print(out.stdout, end="")
    if out.stderr:
        print(out.stderr, end="", file=sys.stderr)
    m = METRIC_RE.search(out.stdout)
    if not m:
        raise RuntimeError(f"could not parse metrics from:\n{out.stdout[-500:]}")
    return float(m.group(1)), float(m.group(2))


def phase_prep(args, cfg):
    cmd = [
        sys.executable,
        "scripts/prepare_dataset.py",
        "--dataset",
        cfg.name,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
    ]
    if cfg.source == "amazon":
        cmd.append("--from_hub")
    if args.movies_only and cfg.source == "movielens":
        cmd.append("--movies_only")
    if args.skip_download:
        cmd.append("--skip_download")
    run(cmd)


def phase_train(args, cfg):
    defaults = cfg.coast_train_defaults()
    run(
        [
            sys.executable,
            "main.py",
            "--dataset",
            cfg.name,
            "--mode",
            "train",
            "--device",
            args.device,
            "--num_epochs",
            str(args.num_epochs or defaults["num_epochs"]),
            "--maxlen",
            str(args.maxlen or defaults["maxlen"]),
            "--batch_size",
            str(args.batch_size),
            "--dropout_rate",
            str(args.dropout_rate if args.dropout_rate is not None else defaults["dropout_rate"]),
            "--seed",
            str(args.seed),
        ]
    )


def phase_eval_coast(args, cfg, results):
    base = [
        sys.executable,
        "main.py",
        "--dataset",
        cfg.name,
        "--device",
        args.device,
        "--checkpoint",
        "best",
        "--seed",
        str(args.seed),
    ]
    for mode, key_ndcg, key_hr in (
        ("warm", "warm_ndcg", "warm_hr"),
        ("cold_start", "cold_ndcg", "cold_hr"),
    ):
        ndcg, hr = capture_metrics(base + ["--mode", mode])
        results["COAST"][key_ndcg] = ndcg
        results["COAST"][key_hr] = hr


def phase_baselines(args, cfg, results):
    py = sys.executable
    for mode, key_ndcg, key_hr in (
        ("warm", "warm_ndcg", "warm_hr"),
        ("cold_start", "cold_ndcg", "cold_hr"),
    ):
        ndcg, hr = capture_metrics(
            [
                py,
                "content_baseline.py",
                "--dataset",
                cfg.name,
                "--mode",
                mode,
                "--seed",
                str(args.seed),
            ]
        )
        results["content_baseline"][key_ndcg] = ndcg
        results["content_baseline"][key_hr] = hr

    if not args.skip_sasrec:
        sasrec_ckpt_dir = SASREC_DIR / f"{cfg.name}_default"
        if not any(sasrec_ckpt_dir.glob("SASRec.epoch=*.pth")):
            run(
                [
                    py,
                    "main.py",
                    f"--dataset={cfg.name}",
                    "--train_dir=default",
                    "--maxlen=50",
                    f"--device={args.device}",
                    "--num_epochs=20",
                    "--batch_size=512",
                    "--hidden_units=50",
                    "--num_blocks=2",
                    "--num_heads=1",
                    "--lr=0.001",
                ],
                cwd=SASREC_DIR,
            )
        for mode in ("warm", "cold_start"):
            key_ndcg = f"{'warm' if mode == 'warm' else 'cold'}_ndcg"
            key_hr = f"{'warm' if mode == 'warm' else 'cold'}_hr"
            ndcg, hr = capture_metrics(
                [
                    py,
                    "eval_sasrec.py",
                    "--dataset",
                    cfg.name,
                    "--mode",
                    mode,
                    "--device",
                    args.device,
                    "--maxlen",
                    "50",
                    "--seed",
                    str(args.seed),
                ]
            )
            results["SASRec"][key_ndcg] = ndcg
            results["SASRec"][key_hr] = hr


def save_results(cfg, results):
    out = cfg.results_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": cfg.name, "methods": results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results → {out}")
    try:
        from scripts.clcrec_results import get_clcrec_metrics

        with open(out) as f:
            data = json.load(f)
        data.setdefault("methods", {})["CLCRec"] = get_clcrec_metrics(cfg.name)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print("merged CLCRec into", out)
    except Exception as e:
        print("CLCRec merge skipped:", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CHOICES))
    p.add_argument(
        "--phase",
        default="all",
        choices=["prep", "train", "eval", "baselines", "all"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--maxlen", type=int, default=None)
    p.add_argument("--dropout_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from_hub", action="store_true")
    p.add_argument("--movies_only", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_sasrec", action="store_true")
    args = p.parse_args()
    cfg = get_dataset(args.dataset)

    results = {
        "COAST": {},
        "SASRec": {},
        "content_baseline": {},
    }

    if args.phase in ("prep", "all"):
        phase_prep(args, cfg)
    if args.phase in ("train", "all"):
        phase_train(args, cfg)
    if args.phase in ("eval", "all"):
        phase_eval_coast(args, cfg, results)
    if args.phase in ("baselines", "all"):
        phase_baselines(args, cfg, results)

    if args.phase in ("eval", "baselines", "all"):
        save_results(cfg, results)


if __name__ == "__main__":
    main()
