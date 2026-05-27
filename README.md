# COAST

Hybrid sequential recommender: SASRec behavior tower + MiniLM content embeddings.

## Colab pipeline (hybrid v2)

```python
!git clone https://github.com/arleen-kaur/COAST.git
%cd COAST
!pip install -q -r requirements.txt datasets huggingface_hub pandas

!python download_meta.py
!python prepare_sasrec.py
!python encode_items.py --device cuda --batch_size 512

# Hybrid COAST (default): ID embedding + content projection
!python main.py --mode train --device cuda --num_epochs 20 --maxlen 50 --batch_size 512

# Eval (same seed for reproducible tables)
!python main.py --mode warm --device cuda --num_epochs 20 --maxlen 50 --seed 42
!python main.py --mode cold_start --device cuda --num_epochs 20 --maxlen 50 --seed 42

# SASRec warm/cold baseline (same protocol)
!python eval_sasrec.py --mode warm --device cuda --maxlen 50 --seed 42
!python eval_sasrec.py --mode cold_start --device cuda --maxlen 50 --seed 42
```

SASRec `.pth` checkpoints are **gitignored** (`baselines/SASRec.pytorch/python/beauty_default/`). On a fresh clone, train once then eval:

```python
%cd /content/COAST
!python prepare_sasrec.py  # ensure baselines/SASRec.pytorch/python/data/beauty.txt

%cd /content/COAST/baselines/SASRec.pytorch/python
!python main.py --dataset=beauty --train_dir=default --maxlen=50 --device cuda \\
  --num_epochs=20 --batch_size=512 --hidden_units=50 --num_blocks=2 --num_heads=1 --lr=0.001

%cd /content/COAST
!python eval_sasrec.py --mode warm --device cuda --maxlen 50 --seed 42
```

## Ablations

```bash
# COAST v1 content-only (old checkpoints: coast_epoch*.pt)
python main.py --mode train --content_only --device cuda --num_epochs 20
python main.py --mode warm --content_only --num_epochs 20
```

Checkpoints: `checkpoints/coast_hybrid_epoch*.pt` (hybrid) or `checkpoints/coast_epoch*.pt` (content-only).

**Cold-start eval (hybrid):** candidate items are scored **content-only** so the cold target does not compete with 100 negatives that still have full ID embeddings (that protocol always ranked the target last).
