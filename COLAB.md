# Google Colab — always run this first

Colab does **not** auto-update when you push to GitHub. Run this **every new session** before experiments:

```python
%cd /content
!rm -rf COAST  # optional: fresh clone
!git clone https://github.com/arleen-kaur/COAST.git
%cd /content/COAST

!pip install -q -r requirements.txt
!python scripts/verify_setup.py --pull
```

If `verify_setup.py` fails, your notebook is on stale code — do **not** run eval until pull succeeds.

## Data prep (required once per session — embeddings are not in git)

```python
!python scripts/prepare_dataset.py --dataset beauty --device cuda --from_hub
# electronics:
# !python scripts/prepare_dataset.py --dataset electronics --device cuda
```

## Train then eval

```python
!python main.py --dataset beauty --mode train --device cuda --num_epochs 20 --seed 42
```

## Eval (no epoch number needed)

After training, evaluation **auto-loads** the best validation checkpoint:

```python
!python main.py --dataset beauty --mode warm --device cuda --seed 42
!python main.py --dataset beauty --mode cold_start --device cuda --seed 42
```

Optional: `--checkpoint last --num_epochs 20` to force a specific epoch file.

## Version

Check installed version:

```python
!cat VERSION
!python scripts/verify_setup.py
```

Current release: see `VERSION` file (e.g. `0.2.0`).

---

## Save data once (skip re-download on every session)

Preprocessed files are **not in git**. After running `prepare_dataset.py` once, pack and upload to **Google Drive**:

### On Colab (after prep finishes)

```python
%cd /content/COAST
!python scripts/pack_data.py --dataset beauty --include_checkpoints
# Or all datasets: !python scripts/pack_data.py --all --include_checkpoints

from google.colab import drive
drive.mount('/content/drive')
!cp artifacts/coast_data_bundle.tar.gz /content/drive/MyDrive/
```

Typical sizes: **Beauty ~80–150 MB**, **Electronics ~200–400 MB** (embeddings dominate).  
Do **not** upload raw `*_reviews.csv` (multi-GB) — only the packed bundle.

### Next Colab sessions

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/COAST
!git pull origin main
!pip install -q -r requirements.txt

!python scripts/restore_data.py --archive /content/drive/MyDrive/coast_data_bundle.tar.gz
!python scripts/verify_setup.py --check-data

# Train / eval — no download or encode needed
!python main.py --dataset beauty --mode warm --device cuda --seed 42
```

### What gets saved

| File | Purpose |
|------|---------|
| `train.csv`, `test.csv` | Splits |
| `meta.csv` | Item text |
| `item_embeddings.npy` | MiniLM vectors |
| `asin2id.json` | Item ID map |
| `data/{dataset}.txt` | SASRec format |
| `checkpoints/` (optional) | COAST `.pt` files |
| `{dataset}_default/*.pth` (optional) | SASRec weights |
