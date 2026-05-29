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
