#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data/RAF-DB}"
OUT="${2:-runs/rafdb}"
python autodl_rafdb/train_rafdb.py --root "$ROOT" --variant tpr --seed 0 --epochs 30 --batch_size 64 --workers 4 --out_dir "$OUT"
