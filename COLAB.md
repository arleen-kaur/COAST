# Google Colab — Phase 2 full pipeline

Run **Cell 1** every new session. Then **Cell 2** (one dataset) or **Cell 3** (all three).

---

## Cell 1 — Clone, install, verify

```python
%cd /content
!git clone https://github.com/arleen-kaur/COAST.git 2>/dev/null || true
%cd /content/COAST
!git pull origin main
!pip install -q -r requirements.txt
!python scripts/verify_setup.py
```

---

## Cell 2 — One dataset (prep + train + eval + baselines)

**Beauty** (skip download if `data/train.csv` exists):

```python
%cd /content/COAST
!python scripts/run_dataset.py --dataset beauty --phase all --device cuda --skip_download
```

**Electronics:**

```python
!python scripts/run_dataset.py --dataset electronics --phase all --device cuda
```

**MovieLens** (set TMDB key or use `--movies_only`):

```python
import os
os.environ["TMDB_API_KEY"] = "YOUR_KEY"  # https://www.themoviedb.org/settings/api

!python scripts/run_dataset.py --dataset movielens --phase all --device cuda
# Without TMDB: add --movies_only
```

Results saved to `results/{dataset}.json`. Early stopping picks best checkpoint automatically.

---

## Cell 3 — All Phase 2 datasets (overnight run)

```python
import os
os.environ["TMDB_API_KEY"] = "YOUR_KEY"  # optional for movielens

%cd /content/COAST
!python scripts/run_phase2.py --device cuda --phase all --skip_download
# Beauty assumes train.csv exists; electronics + movielens download fresh
```

---

## Cell 4 — CLCRec cited comparison

```python
!python eval_clcrec.py --dataset beauty
!python eval_clcrec.py --dataset movielens
```

---

## Cell 5 — Pack to Drive (once)

```python
from google.colab import drive
drive.mount('/content/drive')

!python scripts/pack_data.py --all --include_checkpoints
!cp artifacts/coast_data_bundle.tar.gz /content/drive/MyDrive/
```

---

## Next session (fast)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/COAST
!git pull && !pip install -q -r requirements.txt
!python scripts/restore_data.py --archive /content/drive/MyDrive/coast_data_bundle.tar.gz
!python scripts/run_dataset.py --dataset beauty --phase eval --device cuda
```

---

## What Phase 2 implements

| Step | Status |
|------|--------|
| MovieLens-1M + TMDB | `run_dataset.py --dataset movielens` |
| Early stopping | Default in training; `train_log.json` + `coast_hybrid_best.pt` |
| CLCRec baseline | `eval_clcrec.py` + cited Table 2 numbers |
| One-command suite | `scripts/run_dataset.py` / `scripts/run_phase2.py` |
