#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coast.config import DATASET_CHOICES, get_dataset
from coast.config.datasets import REPO_ROOT
from scripts.results_aggregate import aggregate_seed_runs

SASREC_DIR = REPO_ROOT / "baselines" / "SASRec.pytorch" / "python"
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


def _train_cmd(args, cfg, seed, content_only=False):
    defaults = cfg.coast_train_defaults()
    cmd = [
        sys.executable,
        "-m",
        "coast.cli.main",
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
        "--early_stop_patience",
        str(args.early_stop_patience or defaults["early_stop_patience"]),
        "--min_epochs",
        str(args.min_epochs if args.min_epochs is not None else defaults["min_epochs"]),
        "--early_stop_min_delta",
        str(
            args.early_stop_min_delta
            if args.early_stop_min_delta is not None
            else defaults["early_stop_min_delta"]
        ),
        "--seed",
        str(seed),
    ]
    if content_only:
        cmd.append("--content_only")
    return cmd


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


def phase_eval_coast(args, cfg, seed, results, content_only=False):
    key = "COAST_content_only" if content_only else "COAST"
    results.setdefault(key, {})
    base = [
        sys.executable,
        "-m",
        "coast.cli.main",
        "--dataset",
        cfg.name,
        "--device",
        args.device,
        "--checkpoint",
        "best",
        "--seed",
        str(seed),
    ]
    if content_only:
        base.append("--content_only")
    for mode, key_ndcg, key_hr in (
        ("warm", "warm_ndcg", "warm_hr"),
        ("cold_start", "cold_ndcg", "cold_hr"),
    ):
        ndcg, hr = capture_metrics(base + ["--mode", mode])
        results[key][key_ndcg] = ndcg
        results[key][key_hr] = hr


def phase_baselines(args, cfg, seed, results):
    py = sys.executable
    for mode, key_ndcg, key_hr in (
        ("warm", "warm_ndcg", "warm_hr"),
        ("cold_start", "cold_ndcg", "cold_hr"),
    ):
        ndcg, hr = capture_metrics(
            [
                py,
                "-m",
                "coast.baselines.content",
                "--dataset",
                cfg.name,
                "--mode",
                mode,
                "--seed",
                str(seed),
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
                    "-m",
                    "coast.baselines.sasrec",
                    "--dataset",
                    cfg.name,
                    "--mode",
                    mode,
                    "--device",
                    args.device,
                    "--maxlen",
                    "50",
                    "--seed",
                    str(seed),
                ]
            )
            results["SASRec"][key_ndcg] = ndcg
            results["SASRec"][key_hr] = hr


def save_results(cfg, seeds, per_seed_results, train_defaults):
    aggregated = aggregate_seed_runs(per_seed_results)
    out = cfg.results_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": cfg.name,
        "seeds": seeds,
        "train_defaults": train_defaults,
        "methods": aggregated,
        "per_seed": per_seed_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results -> {out}")

    try:
        from scripts.clcrec_results import get_clcrec_metrics

        with open(out) as f:
            data = json.load(f)
        data.setdefault("methods", {})["CLCRec"] = get_clcrec_metrics(cfg.name)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
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
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--min_epochs", type=int, default=None)
    p.add_argument("--early_stop_min_delta", type=float, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--seed", type=int, default=None, help="single seed override")
    p.add_argument("--ablations", action="store_true", help="train/eval COAST content-only")
    p.add_argument("--movies_only", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_sasrec", action="store_true")
    p.add_argument("--prepare_clcrec", action="store_true")
    args = p.parse_args()

    seeds = [args.seed] if args.seed is not None else args.seeds
    cfg = get_dataset(args.dataset)
    per_seed_results = []

    if args.phase in ("prep", "all"):
        phase_prep(args, cfg)
        if args.prepare_clcrec:
            run(
                [
                    sys.executable,
                    "scripts/prepare_clcrec.py",
                    "--dataset",
                    cfg.name,
                ]
            )

    run_eval = args.phase in ("eval", "baselines", "all")
    run_train = args.phase in ("train", "all")

    if run_eval and not run_train:
        for seed in seeds:
            seed_results = {
                "COAST": {},
                "content_baseline": {},
                "SASRec": {},
            }
            if args.phase in ("eval", "all"):
                phase_eval_coast(args, cfg, seed, seed_results, content_only=False)
                if args.ablations:
                    phase_eval_coast(args, cfg, seed, seed_results, content_only=True)
            if args.phase in ("baselines", "all"):
                phase_baselines(args, cfg, seed, seed_results)
            per_seed_results.append(seed_results)
        save_results(cfg, seeds, per_seed_results, cfg.coast_train_defaults())
    else:
        for seed in seeds:
            if run_train:
                run(_train_cmd(args, cfg, seed, content_only=False))
                if args.ablations:
                    run(_train_cmd(args, cfg, seed, content_only=True))
            if run_eval:
                seed_results = {
                    "COAST": {},
                    "content_baseline": {},
                    "SASRec": {},
                }
                if args.phase in ("eval", "all"):
                    phase_eval_coast(args, cfg, seed, seed_results, content_only=False)
                    if args.ablations:
                        phase_eval_coast(args, cfg, seed, seed_results, content_only=True)
                if args.phase in ("baselines", "all"):
                    phase_baselines(args, cfg, seed, seed_results)
                per_seed_results.append(seed_results)
        if run_eval:
            save_results(cfg, seeds, per_seed_results, cfg.coast_train_defaults())


if __name__ == "__main__":
    main()
