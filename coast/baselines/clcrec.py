
import argparse
import json

from coast.config.datasets import REPO_ROOT

CITED = REPO_ROOT / "baselines" / "CLCRec" / "cited_results.json"
RESULTS = REPO_ROOT / "results"

def load_coast_results(dataset):
    path = RESULTS / f"{dataset}.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(description="CLCRec cited baseline comparison")
    p.add_argument("--dataset", choices=["beauty", "electronics", "movielens", "amazon"], default="amazon")
    args = p.parse_args()

    with open(CITED) as f:
        cited = json.load(f)

    clcrec_key = "amazon" if args.dataset in ("beauty", "electronics", "amazon") else "movielens"
    row = cited[clcrec_key]

    print(f"\n=== CLCRec (cited, {clcrec_key}) ===")
    print(f"Source: {cited['source']}")
    print(f"Note: {cited['note']}\n")
    print(f"{'Setting':<10} {'NDCG@10':>10} {'Recall@10':>12}")
    print("-" * 34)
    for setting in ("cold", "warm", "all"):
        m = row[setting]
        print(f"{setting:<10} {m['ndcg']:>10.4f} {m['recall']:>12.4f}")

    coast = load_coast_results(args.dataset if args.dataset != "amazon" else "beauty")
    if coast:
        print(f"\n=== COAST ({args.dataset}) from results/{args.dataset}.json ===")
        print(f"{'Method':<15} {'Warm NDCG':>10} {'Warm HR':>10} {'Cold NDCG':>10} {'Cold HR':>10}")
        print("-" * 58)
        for method, m in coast.get("methods", {}).items():
            print(
                f"{method:<15} "
                f"{m.get('warm_ndcg', 0):>10.4f} {m.get('warm_hr', 0):>10.4f} "
                f"{m.get('cold_ndcg', 0):>10.4f} {m.get('cold_hr', 0):>10.4f}"
            )
        print(
            "\nCompare COAST cold NDCG/HR to CLCRec cold row above "
            "(different protocol — discuss in paper)."
        )
    else:
        print(f"\nNo COAST results at results/{args.dataset}.json")
        print("Run: python scripts/run_dataset.py --dataset", args.dataset, "--phase eval")

if __name__ == "__main__":
    main()
