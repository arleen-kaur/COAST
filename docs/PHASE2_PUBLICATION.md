# Phase 2 — Publication strengthening (post CS228)

Three tracks added in the codebase:

1. **MovieLens-1M** — third domain (movies vs Amazon products)
2. **Early stopping** — validation NDCG, `coast_hybrid_best.pt`
3. **CLCRec** — dedicated cold-start baseline (docs + clone script)

---

## Step 1: MovieLens-1M + TMDB text

### Prerequisites

- Free TMDB API key: https://www.themoviedb.org/settings/api  
  Set in Colab: `os.environ["TMDB_API_KEY"] = "..."`  
- Without TMDB, use `--movies_only` (title + genres from `movies.dat` only).

### Colab pipeline

```python
%cd /content/COAST
!pip install -q -r requirements.txt requests

import os
os.environ["TMDB_API_KEY"] = "YOUR_KEY_HERE"  # optional but recommended

# 1) Download ML-1M + preprocess
!python download_data.py --dataset movielens
!python preprocess.py --dataset movielens

# 2) Metadata (TMDB plots) + SASRec file + embeddings
!python download_meta.py --dataset movielens
# Or without API: !python download_meta.py --dataset movielens --movies_only

!python prepare_sasrec.py --dataset movielens
!python encode_items.py --dataset movielens --device cuda --batch_size 512

# 3) Train COAST with early stopping (saves coast_hybrid_best.pt)
!python main.py --dataset movielens --mode train --device cuda \
  --num_epochs 50 --maxlen 50 --batch_size 512 --dropout_rate 0.3 --seed 42

# 4) Eval (loads best checkpoint by default)
!python main.py --dataset movielens --mode warm --device cuda --checkpoint best --seed 42
!python main.py --dataset movielens --mode cold_start --device cuda --checkpoint best --seed 42

# Baselines
!python content_baseline.py --dataset movielens --mode cold_start --seed 42
%cd baselines/SASRec.pytorch/python
!python main.py --dataset=movielens --train_dir=default --maxlen=50 --device cuda \
  --num_epochs=20 --batch_size=512 --hidden_units=50
%cd /content/COAST
!python eval_sasrec.py --dataset movielens --mode warm --device cuda --maxlen 50 --seed 42
!python eval_sasrec.py --dataset movielens --mode cold_start --device cuda --maxlen 50 --seed 42
```

**Note:** TMDB fetch for ~3–6k items takes ~15–30 min (rate-limited). Cache: `data/movielens/tmdb_cache.json`.

---

## Step 2: Early stopping (all datasets)

Training now:

- Evaluates **validation** NDCG each epoch (predict `valid` item from `train` only).
- Saves `checkpoints/{dataset}/coast_hybrid_best.pt` on improvement.
- Stops after `--early_stop_patience` (default 5) epochs without gain, after `--min_epochs` (default 2).

Flags:

```bash
python main.py --mode train --no_early_stop          # old behavior (all epochs)
python main.py --mode warm                           # auto: best ckpt, else --num_epochs
python main.py --mode warm --checkpoint last --num_epochs 20
```

**Reviewer note:** Report both best-val epoch and test metrics; mention `--dropout_rate 0.3` if overfitting persists on MovieLens.

---

## Step 3: CLCRec baseline

See [baselines/CLCRec/README.md](../baselines/CLCRec/README.md).

```bash
bash scripts/setup_clcrec.sh
```

Use **cited paper numbers** for a quick comparison, or re-run their Amazon config for reproduction.

---

## Current COAST results (Amazon, epoch checkpoints)

| Dataset | COAST warm NDCG | SASRec warm | COAST cold | SASRec cold |
|---------|-----------------|-------------|------------|-------------|
| Beauty | 0.3235 | 0.3069 | 0.0466 | 0.0000 |
| Electronics | 0.3652 | 0.3366 | 0.2082 | 0.0000 |

Re-run Beauty/Electronics with `--checkpoint best` after retraining with early stopping to replace manual epoch-2 selection.

---

## Venue checklist

| Item | Status |
|------|--------|
| 2+ datasets | Beauty + Electronics ✅ |
| MovieLens (diverse domain) | Pipeline ready — run experiments |
| SASRec + content baselines | ✅ |
| CLCRec | Docs + clone script |
| Early stopping | ✅ |
| Architecture diagram | TODO (draw.io) |
