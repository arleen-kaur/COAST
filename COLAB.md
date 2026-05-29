# Google Colab — overnight publication run

## Keep Colab alive overnight (important on free tier)

Free Colab can **disconnect after ~90 minutes** even while code runs. Use **Colab Pro** if you have it, or run this in a **separate cell** before the overnight script:

```javascript
%%javascript
function ClickConnect(){
    console.log("Keeping alive...");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

Also: **Runtime → Change runtime type → GPU** (T4 is fine).

---

## One cell to run everything (leave overnight)

```python
%cd /content
!git clone https://github.com/arleen-kaur/COAST.git 2>/dev/null || true
%cd /content/COAST
!git pull origin main
!pip install -q -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Full run: Beauty + Electronics + MovieLens + CLCRec (cited) + report
!python scripts/run_publication_overnight.py --device cuda --movies_only \
  --drive_out /content/drive/MyDrive/coast_data_bundle.tar.gz
```

**If you already have Beauty `train.csv` on disk:**

```python
!python scripts/run_publication_overnight.py --device cuda --movies_only --skip_download \
  --drive_out /content/drive/MyDrive/coast_data_bundle.tar.gz
```

**If you already packed data to Drive (fast — eval only):**

```python
!python scripts/run_publication_overnight.py --device cuda --restore /content/drive/MyDrive/coast_data_bundle.tar.gz --phase eval
```

---

## What you get in the morning

| Output | Purpose |
|--------|---------|
| `results/PUBLICATION_REPORT.md` | **Copy into your report** — full tables |
| `results/beauty.json` | All metrics per method |
| `results/electronics.json` | |
| `results/movielens.json` | |
| `results/clcrec_*.json` | CLCRec cited or reproduced |
| `results/overnight_log.json` | What succeeded/failed |
| `artifacts/coast_data_bundle.tar.gz` | Reuse next session |

---

## CLCRec note

- **Always included:** paper Table-2 numbers (`source: cited`) merged into results.
- **Optional full train:** add `--clcrec_train` (needs `baselines/CLCRec/src/Data/amazon/` from authors — not on GitHub).

---

## First-time setup only

```python
!python scripts/verify_setup.py
```

See also [docs/PHASE2_PUBLICATION.md](docs/PHASE2_PUBLICATION.md).
