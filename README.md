# COAST

Arleen Kaur

COAST is a sequential recommender for item cold-start. Each item is a learned ID embedding plus a learned projection of a frozen text embedding. Items that appear in training use both. An item that never appears in training has no trained ID vector, so COAST scores it from text alone.

The sequence model is SASRec (Kang & McAuley, 2018): a causal Transformer over the user’s recent items, trained with binary cross-entropy to predict the next item. Text comes from `all-MiniLM-L6-v2` on the item title, features, and description.

## Method

Text is encoded once and kept frozen. A linear layer maps it to the Transformer hidden size (128):

```
e(i) = E_id[i] + W · MiniLM(text_i)
```

At evaluation, any item id absent from training has `E_id[i]` set to zero. In cold-start mode, candidates are scored from the text term only. The content-only ablation drops `E_id` entirely (`--content_only`).

Training uses Adam (`lr = 1e-3`), batch size 512, sequence length 50, two blocks, and one attention head, for up to 30 epochs. The checkpoint with the best validation NDCG@10 is kept. Beauty uses dropout 0.3 (patience 3); Electronics uses dropout 0.25 (patience 4).

## Data and evaluation

Experiments use Beauty and Electronics from Amazon Reviews 2023 (`McAuley-Lab/Amazon-Reviews-2023`). Reviews are 5-core filtered three times (users and items with at least five interactions). Sequences are ordered by time: the last item is the test target, the second-to-last is validation, and the rest is training. Users with fewer than four interactions are not scored. The filtered data has about 81k users, 49k items, and 854k interactions.

Metrics are HR@10 and NDCG@10. The ground-truth next item is ranked against 100 random items the user has not seen. If there are more than 10,000 users, 10,000 are sampled (seed 42).

A test case is **cold** when the ground-truth item never appears in any training sequence, and **warm** otherwise. SASRec scores zero on cold items because it has no trained embedding for them. The content baseline ranks by cosine similarity to the mean MiniLM embedding of the user’s history, with no sequential model.

## Results

**Beauty.** COAST matches SASRec on warm items and is the one that recalls cold items.

| Method | Warm HR@10 | Cold HR@10 |
| --- | ---: | ---: |
| SASRec (ID only) | 0.488 | 0.000 |
| COAST (ID + text) | 0.517 | 0.171 |

**Electronics.** The sequential model adds more on top of raw text here, where item descriptions vary more.

| Method | Cold HR@10 |
| --- | ---: |
| Content similarity | 0.30 |
| COAST (ID + text) | 0.34 |

**Ablation.** ID-only is strong on warm items and zero on cold items. Content-only handles cold-start and is weaker on warm items, because it misses the collaborative signal. The hybrid recovers warm performance and still ranks new items. Neither signal alone does both. The same pattern holds on both categories.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_dataset.py --dataset beauty --phase all --device cuda
python scripts/run_dataset.py --dataset electronics --phase all --device cuda
```

`--phase all` downloads data (at most 2M review rows per category), trains the hybrid model, and evaluates COAST, SASRec, and the content baseline. Add `--ablations` to include content-only COAST. `scripts/run_experiments.py` runs both categories and writes `results/RESULTS.txt`.

```bash
python -m coast.cli.main --dataset beauty --mode train --device cuda
python -m coast.cli.main --dataset beauty --mode warm --checkpoint best
python -m coast.cli.main --dataset beauty --mode cold_start --checkpoint best
python -m coast.cli.main --dataset beauty --mode train --content_only --device cuda
```

| Path | Role |
| --- | --- |
| `coast/core/model.py` | Hybrid item encoder and causal Transformer |
| `coast/core/evaluate.py` | Sampled HR@10 and NDCG@10, warm and cold |
| `coast/preprocess/` | Download, 5-core filter, MiniLM encoding |
| `coast/baselines/` | SASRec and mean-pooled content similarity |
| `baselines/SASRec.pytorch/` | Vendored SASRec implementation |

## References

- Kang and McAuley. Self-Attentive Sequential Recommendation. ICDM 2018.
- Hou, Li, He, Yan, and McAuley. Bridging Language and Items for Retrieval and Recommendation. Amazon Reviews 2023.
- Reimers and Gurevych. Sentence-BERT. EMNLP 2019. Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
