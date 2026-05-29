#!/usr/bin/env python3
"""Build results/PUBLICATION_REPORT.md from all experiment JSON files."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_json(path):
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(m, k_ndcg, k_hr):
    if not m:
        return "—", "—"
    return f"{m.get(k_ndcg, 0):.4f}", f"{m.get(k_hr, 0):.4f}"


def main():
    datasets = ["beauty", "electronics", "movielens"]
    lines = [
        "# COAST Publication Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Main results (COAST protocol: leave-last-out, 100 negatives, HR@10 / NDCG@10)",
        "",
    ]

    for ds in datasets:
        coast = load_json(RESULTS / f"{ds}.json")
        clcrec = load_json(RESULTS / f"clcrec_{ds}.json")
        if not coast and not clcrec:
            continue

        lines.append(f"### {ds.capitalize()}")
        lines.append("")
        lines.append(
            "| Method | Warm NDCG@10 | Warm HR@10 | Cold NDCG@10 | Cold HR@10 | Notes |"
        )
        lines.append("|--------|--------------|------------|--------------|------------|-------|")

        methods = (coast or {}).get("methods", {})
        for name, m in methods.items():
            wn, wh = fmt(m, "warm_ndcg", "warm_hr")
            cn, ch = fmt(m, "cold_ndcg", "cold_hr")
            lines.append(f"| {name} | {wn} | {wh} | {cn} | {ch} | COAST eval |")

        if clcrec:
            m = clcrec.get("CLCRec", {})
            src = m.get("source", "?")
            wn, wh = fmt(m, "warm_ndcg", "warm_hr")
            cn, ch = fmt(m, "cold_ndcg", "cold_hr")
            note = "paper Table 2" if src == "cited" else "CLCRec repo run"
            lines.append(f"| CLCRec | {wn} | {wh} | {cn} | {ch} | {note} |")

        lines.append("")

    lines.extend(
        [
            "## Training (early stopping)",
            "",
        ]
    )
    for ds in datasets:
        log = load_json(ROOT / "checkpoints" / ds / "train_log.json")
        if not log:
            log = load_json(ROOT / "checkpoints" / "train_log.json") if ds == "beauty" else None
        if log:
            lines.append(
                f"- **{ds}**: best epoch {log.get('best_epoch')} "
                f"(valid NDCG@10={log.get('best_valid_ndcg')})"
            )

    lines.extend(
        [
            "",
            "## Paper writing notes",
            "",
            "- **Warm:** COAST hybrid should match or beat SASRec.",
            "- **Cold:** SASRec typically 0/0; COAST uses content-only candidate scoring for fair cold-start.",
            "- **CLCRec:** Different split/protocol if cited; note in footnote if `source=cited`.",
            "- **Beauty cold:** Content baseline may exceed COAST — discuss homogeneous product text.",
            "- **Electronics cold:** COAST often beats content baseline strongly.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python scripts/run_publication_overnight.py --device cuda --movies_only",
            "```",
            "",
        ]
    )

    out = RESULTS / "PUBLICATION_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
