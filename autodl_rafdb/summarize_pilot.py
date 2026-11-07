# -*- coding: utf-8 -*-
"""Aggregate RAF-DB pilot results and compute the 2x2 interaction.

Usage:
    python summarize_pilot.py --out_dir /root/autodl-tmp/TPR_FER/runs/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ORDER = ["euler", "orth", "heun", "tpr", "tpr_k3", "tpr_local", "tpr_nofinal",
         "tpr_learnh", "tpr_stage4", "tpr_rho50", "shared2", "resnet34"]
METRICS = ["acc", "uar", "f1", "ece"]


def parse_run_dir(path: Path):
    name = path.name
    if "_seed" not in name:
        return None, None
    variant, seed = name.rsplit("_seed", 1)
    try:
        seed = int(seed)
    except ValueError:
        return None, None
    return variant, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs/pilot")
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--md", type=str, default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    for run_dir in sorted(out_dir.glob("*_seed*")):
        variant, seed = parse_run_dir(run_dir)
        if variant is None:
            continue
        # Prefer the one-time test metrics; fall back to best validation.
        test_path = run_dir / "test_metrics.json"
        val_path = run_dir / "best_metrics.json"
        if test_path.is_file():
            mpath, source = test_path, "test"
        elif val_path.is_file():
            mpath, source = val_path, "val"
        else:
            continue
        with mpath.open("r", encoding="utf-8") as f:
            m = json.load(f)
        row = {"variant": variant, "seed": seed, "source": source}
        for k in METRICS:
            row[k] = float(m.get(k, float("nan")))
        row["n"] = int(m.get("n", 0))
        row["epoch"] = int(m.get("epoch", 0))
        rows.append(row)

    if not rows:
        raise SystemExit(f"no best_metrics.json found under {out_dir}")

    csv_path = Path(args.csv) if args.csv else out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "seed", "source"] + METRICS + ["n", "epoch"])
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["variant"]].append(r)

    variants = [v for v in ORDER if v in grouped] + [v for v in grouped if v not in ORDER]

    lines = []
    lines.append("# RAF-DB pilot summary\n")
    sources = sorted(set(r["source"] for r in rows))
    lines.append(f"runs: {len(rows)}  seeds per variant: "
                 + ", ".join(f"{v}={len(grouped[v])}" for v in variants) + "\n")
    lines.append(f"metric source: {sources}. test preferred; val fallback means that run "
                 "did not finish its one-time test evaluation.\n")
    lines.append("| variant | seeds | acc | uar | f1 | ece |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for v in variants:
        rs = grouped[v]
        acc = np.array([r["acc"] for r in rs], dtype=float)
        uar = np.array([r["uar"] for r in rs], dtype=float)
        f1 = np.array([r["f1"] for r in rs], dtype=float)
        ece = np.array([r["ece"] for r in rs], dtype=float)
        lines.append(
            f"| {v} | {len(rs)} | {acc.mean():.4f} +/- {acc.std(ddof=0):.4f} | "
            f"{uar.mean():.4f} +/- {uar.std(ddof=0):.4f} | "
            f"{f1.mean():.4f} +/- {f1.std(ddof=0):.4f} | "
            f"{ece.mean():.4f} +/- {ece.std(ddof=0):.4f} |"
        )

    # 2x2 interaction, aligned by seed.
    required = {"euler", "orth", "heun", "tpr"}
    seed_maps = {v: {r["seed"]: r["acc"] for r in grouped.get(v, [])} for v in required}
    common_seeds = sorted(set.intersection(*(set(seed_maps[v]) for v in required))) if required <= set(grouped) else []
    if common_seeds:
        inter = []
        tpr_vs_best = []
        for s in common_seeds:
            e = seed_maps["euler"][s]
            o = seed_maps["orth"][s]
            h = seed_maps["heun"][s]
            t = seed_maps["tpr"][s]
            inter.append((t - o) - (h - e))
            tpr_vs_best.append(t - max(o, h))
        lines.append("\n## 2x2 interaction (acc)\n")
        lines.append("interaction = (TPR - Orth) - (Heun - Euler), per seed. "
                     "Positive means the coupling is super-additive.\n")
        lines.append(f"- seeds: {common_seeds}")
        lines.append(f"- interaction mean: {np.mean(inter):+.4f}  std: {np.std(inter):.4f}")
        lines.append(f"- TPR - max(Orth, Heun): mean {np.mean(tpr_vs_best):+.4f}  "
                     f"std {np.std(tpr_vs_best):.4f}")
        lines.append("")
        lines.append("| seed | Euler | Orth | Heun | TPR | interaction | TPR-best_parent |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for s, a, b, c, d, ii, bb in zip(common_seeds, [seed_maps["euler"][s] for s in common_seeds],
                                         [seed_maps["orth"][s] for s in common_seeds],
                                         [seed_maps["heun"][s] for s in common_seeds],
                                         [seed_maps["tpr"][s] for s in common_seeds],
                                         inter, tpr_vs_best):
            lines.append(f"| {s} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {ii:+.4f} | {bb:+.4f} |")
    else:
        lines.append("\n2x2 interaction not computed: need euler/orth/heun/tpr with common seeds.")

    text = "\n".join(lines) + "\n"
    print(text)
    md_path = Path(args.md) if args.md else out_dir / "summary.md"
    md_path.write_text(text, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
