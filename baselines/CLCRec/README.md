# CLCRec baseline (Contrastive Learning for Cold-Start Recommendation)

**Paper:** Wei et al., *Contrastive Learning for Cold-Start Recommendation*, ACM MM 2021.  
**Official code:** https://github.com/iLearn-Lab/MM21-CLCRec (mirror: https://github.com/weiyinwei/CLCRec)

CLCRec is the standard dedicated cold-start baseline: it maximizes mutual information between item content and collaborative signals via contrastive learning. Use it to compare against COAST beyond SASRec (which is not designed for cold-start).

---

## Option A — Cite published numbers (fastest)

The original paper reports results on **Amazon** and **MovieLens** with their own split protocol. Numbers may not match COAST’s leave-last-out + 100-negative eval exactly. In your paper, cite their table and note the protocol difference, or run Option B.

| Dataset (paper) | Setting | Metric (typical in paper) | Reported trend |
|-----------------|---------|---------------------------|----------------|
| Amazon | Cold items | HR / NDCG | CLCRec > content-only / MF variants |
| MovieLens | Cold items | HR / NDCG | CLCRec > baselines |

**Action:** Open the paper PDF Table/Figure for exact HR@K and NDCG@K on Amazon Beauty and copy into your comparison table with a footnote: *“CLCRec from Wei et al. (2021); different split protocol.”*

---

## Option B — Run CLCRec locally (1–2 days)

### 1. Clone and install

```bash
cd baselines
git clone https://github.com/iLearn-Lab/MM21-CLCRec.git CLCRec
cd CLCRec
pip install -r requirements.txt  # if present; else torch, numpy, scipy
```

Or run from repo root:

```bash
bash scripts/setup_clcrec.sh
```

### 2. Amazon (their preprocessed format)

Their repo ships **preprocessed** Amazon and MovieLens pickles. Training:

```bash
python main.py --model_name='CLCRec' --data_path=amazon \
  --l_r=0.001 --reg_weight=0.001 --num_workers=4 --num_neg=512 \
  --has_v=True --lr_lambda=0.9 --num_sample=0.5
```

### 3. Aligning with COAST splits

COAST uses:
- McAuley Amazon Beauty/Electronics or MovieLens-1M
- 5-core filter, leave-last-out
- HR@10 / NDCG@10, 100 random negatives, seed 42

CLCRec uses its own preprocessing. For a **fair** comparison you would need to either:
- Export COAST `train.csv` / `test.csv` into their data loader format, or
- Reimplement their contrastive objective on COAST splits (larger effort).

For a **course / short paper**, Option A (cited numbers + protocol note) is acceptable. For **RecSys full paper**, invest in Option B or a unified split via [ColdRec](https://github.com/YuanchenBei/ColdRec).

---

## How to cite in your COAST paper

> We compare to CLCRec (Wei et al., 2021), a contrastive cold-start method. SASRec fails on cold items under our protocol (HR@10 = 0). COAST enables cold-start ranking via hybrid ID+content representations; on Electronics, COAST outperforms a cosine content baseline. CLCRec results are from the original paper / [or: our reproduction on their split].

---

## References

```bibtex
@inproceedings{wei2021clcrec,
  title={Contrastive Learning for Cold-Start Recommendation},
  author={Wei, Yinwei and others},
  booktitle={ACM MM},
  year={2021}
}
```
