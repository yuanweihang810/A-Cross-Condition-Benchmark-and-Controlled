# -*- coding: utf-8 -*-
"""Check RAF-DB paths and label file before training.

Example:
    python check_data.py --root /root/autodl-tmp/IEEE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rafdb_data import RAFDB_CLASS_NAMES, scan_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/IEEE")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--label_file", type=str, default="")
    p.add_argument("--out", type=str, default="data_report.json")
    args = p.parse_args()

    root = Path(args.root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "Image/aligned"
    label_file = Path(args.label_file) if args.label_file else root / "EmoLabel/list_patition_label.txt"

    print(f"root       : {root}")
    print(f"image_dir  : {image_dir}")
    print(f"label_file : {label_file}")
    if not image_dir.is_dir():
        raise SystemExit(f"image_dir does not exist: {image_dir}")
    if not label_file.is_file():
        raise SystemExit(f"label_file does not exist: {label_file}")

    report = scan_dataset(image_dir, label_file)
    report["class_names"] = RAFDB_CLASS_NAMES

    print("\nSplit summary")
    for split, info in report["splits"].items():
        print(f"  {split:5s}: total={info['count']:5d} found={info['found']:5d} "
              f"missing={info['missing']:5d}")
        print("         " + ", ".join(f"{k}={v}" for k, v in info["class_counts"].items()))

    print("\nResolved examples")
    for pth in report["resolved_examples"]:
        print("  " + pth)

    missing_total = sum(v["missing"] for v in report["splits"].values())
    if missing_total:
        print("\nFirst missing files")
        for split, names in report["missing"].items():
            if names:
                print(f"  [{split}] " + ", ".join(names))

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    if missing_total:
        print(f"WARNING: {missing_total} images missing. Training will fail unless --skip_missing is used.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
