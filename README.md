# COAST

Hybrid sequential recommender: SASRec behavior tower + MiniLM content embeddings.

## Datasets

| `--dataset` | Category | Metadata source |
|-------------|----------|-----------------|
| `beauty` (default) | Beauty & Personal Care | smartcat HF hub |
| `electronics` | Electronics | McAuley meta JSONL (filtered to train/test items) |

Legacy Beauty paths (`data/train.csv`, `data/beauty_meta.csv`, etc.) still work if you haven't re-run preprocess.

---

## Colab: Amazon Electronics (second dataset)

```python
%cd /content/COAST
!git pull
!pip install -q -r requirements.txt datasets huggingface_hub pandas

# 1) Reviews sample (2M rows) + preprocess
!python download_data.py --dataset electronics
!python preprocess.py --dataset electronics

# 2) Metadata for items in split + SASRec file + embeddings
!python download_meta.py --dataset electronics
!python prepare_sasrec.py --dataset electronics
!python encode_items.py --dataset electronics --device cuda --batch_size 512

# 3) Hybrid COAST (use best epoch from log for eval, often ~2–5)
!python main.py --dataset electronics --mode train --device cuda \
  --num_epochs 20 --maxlen 50 --batch_size 512 --seed 42

!python main.py --dataset electronics --mode warm --device cuda --num_epochs 2 --maxlen 50 --seed 42
!python main.py --dataset electronics --mode cold_start --device cuda --num_epochs 2 --maxlen 50 --seed 42

# 4) Baselines
!python content_baseline.py --dataset electronics --mode cold_start --seed 42
!python content_baseline.py --dataset electronics --mode warm --seed 42

# SASRec (train once per Colab session — checkpoints gitignored)
%cd baselines/SASRec.pytorch/python
!python main.py --dataset=electronics --train_dir=default --maxlen=50 --device cuda \
  --num_epochs=20 --batch_size=512 --hidden_units=50
%cd /content/COAST
!python eval_sasrec.py --dataset electronics --mode warm --device cuda --maxlen 50 --seed 42
!python eval_sasrec.py --dataset electronics --mode cold_start --device cuda --maxlen 50 --seed 42
```

---

## Beauty (hybrid v2) — quick eval after pull

```python
!python main.py --mode warm --device cuda --num_epochs 2 --maxlen 50 --seed 42
!python main.py --mode cold_start --device cuda --num_epochs 2 --maxlen 50 --seed 42
!python content_baseline.py --mode cold_start --seed 42
!python eval_sasrec.py --mode warm --device cuda --maxlen 50 --seed 42
```

---

## Ablations

```bash
python main.py --content_only --mode train --device cuda --num_epochs 20
python content_baseline.py --mode cold_start   # untrained content cosine baseline
```

**Cold-start eval (hybrid):** all 101 candidates scored with **content-only** vectors (fair vs ID-heavy negatives).

Checkpoints: `checkpoints/{dataset}/coast_hybrid_epoch*.pt`
