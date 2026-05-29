#!/usr/bin/env python3
"""
One command for overnight publication runs.

Runs COAST experiments (Beauty + Electronics + MovieLens), CLCRec (cited + optional train),
packs artifacts, and writes results/PUBLICATION_REPORT.md for your report.

Colab:
  %cd /content/COAST
  !python scripts/run_publication_overnight.py --device cuda --movies_only

With existing Beauty splits:
  !python scripts/run_publication_overnight.py --device cuda --movies_only --skip_download

Restore Drive bundle first:
  !python scripts/run_publication_overnight.py --device cuda --restore /content/drive/MyDrive/coast_data_bundle.tar.gz --phase eval
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd, cwd=None, check=True):
    print("\n" + "=" * 70)
    print(">>>", " ".join(str(c) for c in cmd))
    print("=" * 70, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=cwd or ROOT, check=check)
    print(f"({time.time() - t0:.0f}s)\n")


def main():
    p = argparse.ArgumentParser(description="Overnight COAST + baselines + CLCRec + report")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["beauty", "electronics", "movielens"],
    )
    p.add_argument(
        "--phase",
        default="all",
        choices=["prep", "train", "eval", "baselines", "all"],
        help="COAST pipeline phase (use 'eval' if data+checkpoints restored)",
    )
    p.add_argument("--movies_only", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_coast", action="store_true")
    p.add_argument("--skip_clcrec", action="store_true")
    p.add_argument("--skip_pack", action="store_true")
    p.add_argument("--clcrec_train", action="store_true", help="try official CLCRec train (often fails without their Data/)")
    p.add_argument("--restore", default=None, help="path to coast_data_bundle.tar.gz")
    p.add_argument("--drive_out", default=None, help="copy bundle here when done, e.g. /content/drive/MyDrive/coast_data_bundle.tar.gz")
    args = p.parse_args()

    py = sys.executable
    log_path = ROOT / "results" / "overnight_log.json"
    log = {"started": time.time(), "steps": []}

    def step(name, ok=True, detail=""):
        log["steps"].append({"name": name, "ok": ok, "detail": detail})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    run([py, "scripts/verify_setup.py", "--pull"])
    step("verify_setup")

    if args.restore:
        run([py, "scripts/restore_data.py", "--archive", args.restore])
        step("restore_data", detail=args.restore)

    if not args.skip_coast:
        for ds in args.datasets:
            cmd = [
                py,
                "scripts/run_dataset.py",
                "--dataset",
                ds,
                "--phase",
                args.phase,
                "--device",
                args.device,
            ]
            if args.movies_only:
                cmd.append("--movies_only")
            if args.skip_download:
                cmd.append("--skip_download")
            try:
                run(cmd)
                step(f"coast_{ds}", detail=args.phase)
            except subprocess.CalledProcessError as e:
                step(f"coast_{ds}", ok=False, detail=str(e))
                print(f"WARNING: {ds} failed — continuing")

    if not args.skip_clcrec:
        cmd = [py, "scripts/run_clcrec.py", "--dataset", "all"]
        if args.clcrec_train:
            cmd.append("--train")
        try:
            run(cmd, check=False)
            step("clcrec")
        except Exception as e:
            step("clcrec", ok=False, detail=str(e))

        # Merge CLCRec into coast results JSON
        from scripts.clcrec_results import get_clcrec_metrics

        for ds in args.datasets:
            coast_path = ROOT / "results" / f"{ds}.json"
            if not coast_path.is_file():
                continue
            with open(coast_path) as f:
                data = json.load(f)
            data.setdefault("methods", {})["CLCRec"] = get_clcrec_metrics(ds)
            with open(coast_path, "w") as f:
                json.dump(data, f, indent=2)
        step("merge_clcrec")

    run([py, "scripts/generate_publication_report.py"])
    step("report")

    if not args.skip_pack and args.phase in ("all", "train", "eval", "baselines"):
        pack_cmd = [py, "scripts/pack_data.py", "--include_checkpoints"]
        for ds in args.datasets:
            pack_cmd.extend(["--dataset", ds])
        run(pack_cmd)
        step("pack_data")
        if args.drive_out:
            import shutil

            src = ROOT / "artifacts" / "coast_data_bundle.tar.gz"
            if src.is_file():
                shutil.copy(src, args.drive_out)
                step("copy_drive", detail=args.drive_out)

    log["finished"] = time.time()
    log["elapsed_hours"] = (log["finished"] - log["started"]) / 3600
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "#" * 70)
    print("OVERNIGHT RUN COMPLETE")
    print(f"  Report:  {ROOT / 'results' / 'PUBLICATION_REPORT.md'}")
    print(f"  JSON:    {ROOT / 'results'}/*.json")
    print(f"  Log:     {log_path}")
    print("#" * 70)


if __name__ == "__main__":
    main()
