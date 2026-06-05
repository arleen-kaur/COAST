#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd, cwd=None, check=True):
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=check)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--datasets", nargs="+", default=["beauty", "electronics", "movielens"])
    p.add_argument("--phase", default="all", choices=["prep", "train", "eval", "baselines", "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--movies_only", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--clcrec_train", action="store_true")
    p.add_argument("--prepare_clcrec", action="store_true")
    p.add_argument("--restore", default=None)
    p.add_argument("--drive_out", default=None)
    args = p.parse_args()

    py = sys.executable

    run([py, "scripts/verify_setup.py", "--pull"], check=False)

    if args.restore:
        run([py, "scripts/restore_data.py", "--archive", args.restore])

    if args.prepare_clcrec or args.clcrec_train:
        run([py, "scripts/prepare_clcrec.py", "--all"])

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
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        if args.ablations:
            cmd.append("--ablations")
        if args.movies_only:
            cmd.append("--movies_only")
        if args.skip_download:
            cmd.append("--skip_download")
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            print(f"warning: {ds} failed, continuing")

    if args.clcrec_train:
        for ds in args.datasets:
            run([py, "scripts/run_clcrec.py", "--dataset", ds], check=False)

    from scripts.clcrec_results import get_clcrec_metrics

    for ds in args.datasets:
        path = ROOT / "results" / f"{ds}.json"
        if not path.is_file():
            continue
        with open(path) as f:
            data = json.load(f)
        data.setdefault("methods", {})["CLCRec"] = get_clcrec_metrics(ds)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    run([py, "scripts/generate_report.py"], check=False)

    if args.phase in ("all", "train", "eval", "baselines"):
        pack_cmd = [py, "scripts/pack_data.py", "--include_checkpoints"]
        for ds in args.datasets:
            pack_cmd.extend(["--dataset", ds])
        run(pack_cmd, check=False)
        if args.drive_out:
            import shutil

            src = ROOT / "artifacts" / "coast_data_bundle.tar.gz"
            if src.is_file():
                shutil.copy(src, args.drive_out)


if __name__ == "__main__":
    main()
