# COAST: Content-Augmented Sequential Recommendation for Cold-Start

CS228 course project. COAST tackles the **item cold-start** problem in sequential
recommendation: standard models like SASRec learn item embeddings from clicks, so a brand
new item with no interactions gets a zero embedding and is never recommended. COAST adds a
content signal so new items can be recommended from their text alone.

## Method

Each item is represented as the sum of two parts:

- a **learned ID embedding** (collaborative signal, like SASRec), and
- a **content embedding** from the item's title + features + description, encoded with a
  frozen MiniLM sentence encoder.

A user's click history is read by a Transformer (SASRec-style self-attention). For a cold
item the ID embedding is zero (it was never trained), so the content embedding carries the
full representation. This lets COAST recommend items it has never seen while keeping warm
performance on par with SASRec.

## Datasets and metrics

- **Amazon Beauty** and **Amazon Electronics** (Amazon Reviews 2023), 5-core filtered.
- Leave-one-out split: each user's last interaction is the test item.
- Metrics: **HR@10** and **NDCG@10**, reported separately for *warm* items (seen in
  training) and *cold-start* items (never seen in training).

## Setup

```bash
pip install -r requirements.txt
```

## Run

Prepare data, train, and evaluate COAST plus baselines for one dataset:

```bash
python scripts/run_dataset.py --dataset beauty --device cuda --ablations
```

`--ablations` also trains/evaluates a **content-only** variant (ID embeddings removed) to
isolate the contribution of each signal. Run both datasets and write a summary table:

```bash
python scripts/run_experiments.py --datasets beauty electronics --device cuda --ablations
python scripts/generate_report.py   # writes results/RESULTS.txt
```

Phases can be run individually with `--phase {prep,train,eval,baselines,all}`.

## Repository layout

```
coast/
  config/      dataset configs and paths
  preprocess/  download, filter (5-core), build SASRec file, encode items with MiniLM
  core/        model, data loading, evaluation, checkpointing
  train/       training loop with validation-based early stopping
  cli/         train/eval entry point (python -m coast.cli.main)
  baselines/   content-only baseline and SASRec wrapper
scripts/       end-to-end runners, reporting, and data pack/restore helpers
baselines/     vendored SASRec.pytorch baseline
```

## Baselines

- **SASRec** — ID-only sequential recommender. Strong on warm items, scores ~0 on
  cold-start since cold items have no learned embedding.
- **Content-only** — text embeddings without the sequential ID signal.
- **COAST (hybrid)** — both signals; matches SASRec on warm items and is the only method
  that recommends cold items meaningfully.
