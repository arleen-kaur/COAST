"""Load or build CLCRec metrics for results tables."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITED = ROOT / "baselines" / "CLCRec" / "cited_results.json"
CLCREC_SRC = ROOT / "baselines" / "CLCRec" / "src"


def clcrec_paper_key(coast_dataset: str) -> str:
    return "amazon" if coast_dataset in ("beauty", "electronics", "amazon") else "movielens"


def cited_metrics(coast_dataset: str) -> dict:
    with open(CITED) as f:
        cited = json.load(f)
    key = clcrec_paper_key(coast_dataset)
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
    """Parse CLCRec ./Data/{amazon|movielens}/result_*.txt output."""
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


def find_reproduced_results(data_path: str) -> dict | None:
    data_dir = CLCREC_SRC / "Data" / data_path
    if not data_dir.is_dir():
        return None
    for p in sorted(data_dir.glob("result*.txt"), reverse=True):
        parsed = parse_clcrec_result_file(p)
        if parsed:
            return parsed
    return None


def get_clcrec_metrics(coast_dataset: str) -> dict:
    """Reproduced CLCRec if Data/ exists and training finished; else cited paper numbers."""
    data_path = clcrec_paper_key(coast_dataset)
    reproduced = find_reproduced_results(data_path)
    if reproduced:
        return reproduced
    return cited_metrics(coast_dataset)
