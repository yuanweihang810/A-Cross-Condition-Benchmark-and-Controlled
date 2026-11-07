#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data/CALE/data_aligned}"
OUT="${2:-runs/cale_u}"
python autodl_rafdb/train_cale.py --root "$ROOT" --variant tpr --seed 0 --epochs 30 --train_conditions standard --val_conditions standard --batch_size 128 --workers 8 --out_dir "$OUT"
