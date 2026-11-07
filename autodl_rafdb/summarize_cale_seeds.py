# -*- coding: utf-8 -*-
"""Aggregate CALE cross-condition runs over seeds."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ["standard", "dark", "bright", "lowres"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs/cale_all")
    args = p.parse_args()
    root = Path(args.out_dir)
    rows = []
    for d in sorted(root.iterdir()):
        mpath = d / "test_metrics.json"
        if not mpath.is_file() or "_seed" not in d.name:
            continue
        variant, seed = d.name.split("_standard")[0], d.name.rsplit("_seed", 1)[-1]
        m = json.loads(mpath.read_text(encoding="utf-8"))
        row = {"variant": variant, "seed": int(seed)}
        for c in CONDS:
            row[c] = m[c]["acc"]
        row["mean"] = m["mean_over_conditions"]["acc"]
        row["uar"] = m["mean_over_conditions"]["uar"]
        row["ece"] = m["mean_over_conditions"]["ece"]
        rows.append(row)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["variant"]].append(r)

    lines = ["# CALE cross-condition, multi-seed\n",
             "| Method | seeds | standard | dark | bright | lowres | mean | UAR | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v, rs in sorted(grouped.items(), key=lambda kv: -np.mean([r["mean"] for r in kv[1]])):
        def m(k):
            a = np.array([r[k] for r in rs], dtype=float)
            return f"{a.mean():.4f} +/- {a.std(ddof=0):.4f}"
        lines.append(f"| {v} | {len(rs)} | " + " | ".join(m(c) for c in CONDS)
                     + f" | {m('mean')} | {m('uar')} | {m('ece')} |")

    text = "\n".join(lines) + "\n"
    print(text)
    (root / "summary_seeds.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

