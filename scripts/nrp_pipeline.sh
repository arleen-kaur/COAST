#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-20}"
MAXLEN="${MAXLEN:-50}"
BATCH="${BATCH:-512}"

echo "device=$DEVICE epochs=$EPOCHS maxlen=$MAXLEN batch=$BATCH"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python encode_items.py --device "$DEVICE" --batch_size 512

python main.py --mode train --device "$DEVICE" --num_epochs "$EPOCHS" --maxlen "$MAXLEN" --batch_size "$BATCH"

python main.py --mode evaluate --device "$DEVICE" --num_epochs "$EPOCHS" --maxlen "$MAXLEN"
python main.py --mode warm --device "$DEVICE" --num_epochs "$EPOCHS" --maxlen "$MAXLEN"
python main.py --mode cold_start --device "$DEVICE" --num_epochs "$EPOCHS" --maxlen "$MAXLEN"

echo "done — checkpoints in checkpoints/"
