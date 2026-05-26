# Running COAST on UCR CS228 NRP JupyterHub (GPU)

Use NRP for **GPU encoding**, **COAST training**, and **full-scale experiments** for the paper. Keep coding on your laptop; run heavy jobs on the hub.

## 0. One-time access

1. Log in at **https://nrp.ai/** with UCR credentials (required once).
2. Open **https://ucr-cs-228-s26-hub.nrp-nautilus.io/** and sign in.
3. Start a server with a **GPU** profile (name varies: “GPU”, “CUDA”, etc.).
4. Issues → **systems@cs.ucr.edu**.

## 1. Get code on the hub

**Option A — Git (best)**

```bash
cd ~
git clone <YOUR_REPO_URL> COAST
cd COAST
```

**Option B — Upload**  
Zip the project (without `data/beauty_reviews.csv` if huge), upload in Jupyter, unzip.

## 2. Put data on the hub

You need these under `COAST/data/`:

| File | Needed for |
|------|------------|
| `train.csv`, `test.csv` | COAST + SASRec splits |
| `beauty_meta.csv` | Item text → embeddings |
| `beauty.txt` (optional) | Only if re-running SASRec from scratch |

**Do not upload** `beauty_reviews.csv` (11GB) if you already have `train.csv` / `test.csv` from local `preprocess.py`.

Upload via Jupyter file browser, or `scp`/`rsync` if your course provides SSH.

## 3. Environment (terminal in Jupyter)

```bash
cd ~/COAST   # your clone path

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Hub images often ship CUDA PyTorch; if encode fails, install a CUDA wheel from pytorch.org
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 4. GPU pipeline (paper experiments)

Run from repo root:

```bash
source .venv/bin/activate

# Step 1 — content vectors (~few min on GPU vs ~1hr CPU)
python encode_items.py --device cuda --batch_size 512

# Step 2 — COAST train + eval each epoch
python main.py --mode train --device cuda --num_epochs 20 --maxlen 50 --batch_size 512

# Step 3 — report metrics
python main.py --mode evaluate --device cuda --num_epochs 20
python main.py --mode warm --device cuda --num_epochs 20
python main.py --mode cold_start --device cuda --num_epochs 20
```

Or use the helper script:

```bash
bash scripts/nrp_pipeline.sh
```

## 5. SASRec baseline on GPU (optional, for paper table)

```bash
# from repo root, if beauty.txt is missing:
python prepare_sasrec.py

cd baselines/SASRec.pytorch/python
python main.py --dataset=beauty --train_dir=default --maxlen=50 --device cuda --num_epochs 20 --batch_size 512
python main.py --dataset=beauty --train_dir=default --maxlen=50 --device cuda \
  --state_dict_path=beauty_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth \
  --inference_only=true
```

## 6. Download results to your laptop

Copy back:

- `data/item_embeddings.npy`, `data/asin2id.json`
- `checkpoints/coast_epoch*.pt`
- `baselines/SASRec.pytorch/python/beauty_default/log.txt`

Use Jupyter “Download” or `tar czf results.tgz data/item_embeddings.npy data/asin2id.json checkpoints &&` download.

## 7. Paper-oriented roadmap on NRP

| Phase | What | Command / note |
|-------|------|----------------|
| **Baselines** | Popularity + SASRec | Already have local numbers; re-run SASRec on GPU for fair match |
| **COAST v1** | Train + warm/cold eval | `main.py` as above |
| **Scale-up** | Full 20M preprocess | Run `preprocess.py` without sample cap on hub (needs reviews CSV or pre-built splits) |
| **Ablations** | No content / frozen vs fine-tune | Code changes + short GPU runs |
| **Analysis** | Cold-start % of test, text length vs HR | Notebook on hub |

Document in the paper: **Amazon Beauty 2023**, **2M interaction sample** (milestone) vs **full corpus** (if you scale), **leave-one-out / SASRec-style eval**, **HR@10 / NDCG@10**.

## 8. Session tips

- GPU sessions **time out** — save checkpoints every epoch (COAST already does).
- Log experiments: `checkpoints/`, `beauty_default/log.txt`, and copy console output.
- Pin versions in `requirements.txt` for reproducibility.
