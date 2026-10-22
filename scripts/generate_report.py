#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def fmt_metric(m, key):
    if key not in m:
        return "—"
    return f"{m[key]:.4f}"


def main():
    lines = [f"COAST results — {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    for ds in ["beauty", "electronics"]:
        path = RESULTS / f"{ds}.json"
        if not path.is_file():
            continue
        with open(path) as f:
            data = json.load(f)
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("method\twarm_ndcg\twarm_hr\tcold_ndcg\tcold_hr")
        for name, m in data.get("methods", {}).items():
            lines.append(
                f"{name}\t{fmt_metric(m, 'warm_ndcg')}\t{fmt_metric(m, 'warm_hr')}\t"
                f"{fmt_metric(m, 'cold_ndcg')}\t{fmt_metric(m, 'cold_hr')}"
            )
        lines.append("")

    out = RESULTS / "RESULTS.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
