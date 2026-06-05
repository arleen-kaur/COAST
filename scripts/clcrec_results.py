import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITED = ROOT / "baselines" / "CLCRec" / "cited_results.json"
CLCREC_SRC = ROOT / "baselines" / "CLCRec" / "src"


def clcrec_data_path(coast_dataset: str) -> str:
    from coast.config import get_dataset

    return get_dataset(coast_dataset).clcrec_data_name()


def cited_metrics(coast_dataset: str) -> dict:
    with open(CITED) as f:
        cited = json.load(f)
    key = "amazon" if coast_dataset in ("beauty", "electronics", "amazon") else "movielens"
    row = cited[key]
    return {
        "source": "cited",
        "paper": cited["source"],
        "protocol_note": cited["note"],
        "warm_ndcg": row["warm"]["ndcg"],
        "warm_hr": row["warm"]["recall"],
        "cold_ndcg": row["cold"]["ndcg"],
        "cold_hr": row["cold"]["recall"],
        "all_ndcg": row["all"]["ndcg"],
        "all_hr": row["all"]["recall"],
    }


def parse_clcrec_result_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text()
    out = {"source": "reproduced", "result_file": str(path)}

    block = {
        "cold": r"Test Cold Precition:([\d.]+) Recall:([\d.]+) NDCG:([\d.]+)",
        "warm": r"Test Warm Precition:([\d.]+) Recall:([\d.]+) NDCG:([\d.]+)",
    }
    for setting, pat in block.items():
        m = re.search(pat, text)
        if m:
            out[f"{setting}_ndcg"] = float(m.group(3))
            out[f"{setting}_hr"] = float(m.group(2))

    if "cold_ndcg" in out:
        return out
    return None


def find_reproduced_results(coast_dataset: str) -> dict | None:
    data_path = clcrec_data_path(coast_dataset)
    data_dir = CLCREC_SRC / "Data" / data_path
    if not data_dir.is_dir():
        return None
    patterns = [
        f"result_coast_{coast_dataset}.txt",
        "result_*.txt",
    ]
    for pattern in patterns:
        for p in sorted(data_dir.glob(pattern), reverse=True):
            parsed = parse_clcrec_result_file(p)
            if parsed:
                parsed["coast_dataset"] = coast_dataset
                parsed["data_path"] = data_path
                return parsed
    return None


def get_clcrec_metrics(coast_dataset: str) -> dict:
    reproduced = find_reproduced_results(coast_dataset)
    if reproduced:
        return reproduced
    return cited_metrics(coast_dataset)


def write_clcrec_json(coast_dataset: str, extra: dict | None = None) -> dict:
    metrics = get_clcrec_metrics(coast_dataset)
    if extra:
        metrics.update(extra)
    out = ROOT / "results" / f"clcrec_{coast_dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": coast_dataset, "CLCRec": metrics}, f, indent=2)
    print(f"Wrote {out} (source={metrics.get('source')})")
    return metrics


def clone_clcrec() -> Path:
    if (CLCREC_SRC / "main.py").is_file():
        return CLCREC_SRC
    CLCREC_SRC.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning CLCRec ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/iLearn-Lab/MM21-CLCRec.git", str(CLCREC_SRC)],
        check=True,
    )
    return CLCREC_SRC


def run_clcrec_for_datasets(
    datasets: list[str],
    train: bool = False,
    timeout_s: int = 7200,
) -> None:
    if train:
        py = sys.executable
        for ds in datasets:
            subprocess.run(
                [py, str(ROOT / "scripts" / "run_clcrec.py"), "--dataset", ds],
                cwd=ROOT,
                check=False,
            )
        return

    for ds in datasets:
        write_clcrec_json(ds)
