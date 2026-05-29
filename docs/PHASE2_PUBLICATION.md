# Phase 2 — Publication (implemented)

Run from repo root:

```bash
# Single dataset (prep + train + eval + baselines → results/{dataset}.json)
python scripts/run_dataset.py --dataset beauty --phase all --device cuda --skip_download
python scripts/run_dataset.py --dataset electronics --phase all --device cuda
python scripts/run_dataset.py --dataset movielens --phase all --device cuda

# All three datasets
python scripts/run_phase2.py --device cuda --phase all

# CLCRec cited comparison (after results/*.json exist)
python eval_clcrec.py --dataset beauty
python eval_clcrec.py --dataset movielens
```

## Step 1 — MovieLens-1M ✅

- Pipeline: `download_data.py`, `fetch_tmdb.py`, `prepare_dataset.py`
- TMDB: set `TMDB_API_KEY` or `--movies_only`
- Defaults: 50 epochs, dropout 0.3

## Step 2 — CLCRec ✅

- Cited numbers: `baselines/CLCRec/cited_results.json` (Table 2, Wei et al. MM 2021)
- Comparison script: `eval_clcrec.py`
- Full reproduction: `bash scripts/setup_clcrec.sh` + `baselines/CLCRec/README.md`

## Step 3 — Early stopping ✅

- Validation NDCG each epoch; saves `coast_hybrid_best.pt`
- Training log: `checkpoints/{dataset}/train_log.json`
- Eval auto-loads best checkpoint (`--checkpoint auto`)

See [COLAB.md](../COLAB.md) for notebook cells.
