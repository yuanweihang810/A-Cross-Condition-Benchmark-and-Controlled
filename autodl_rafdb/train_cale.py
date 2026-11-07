# -*- coding: utf-8 -*-
"""Cross-condition CALE pilot: train on standard, test on all four conditions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from tpr import set_tpr_gamma, count_parameters  # noqa: E402
from train_rafdb import build_model, expected_calibration_error, uses_tpr  # noqa: E402
from cale_data import (CLASSES, CONDITIONS, COND_TO_ID, CALEDataset,  # noqa: E402
                       build_condition_map, condition_counts, scan_instances,
                       split_instances)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_tf():
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])
    return train_tf, test_tf


@torch.inference_mode()
def eval_by_condition(model, loader, device):
    model.eval()
    probs_all, labels_all, conds_all = [], [], []
    for x, y, c, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
        probs_all.append(F.softmax(logits.float(), dim=1).cpu())
        labels_all.append(y)
        conds_all.append(c)
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    conds = torch.cat(conds_all).numpy()
    out = {}
    for name, cid in COND_TO_ID.items():
        m = conds == cid
        if m.sum() == 0:
            continue
        p = probs[m]
        y = labels[m]
        pred = p.argmax(axis=1)
        out[name] = {
            "acc": float(accuracy_score(y, pred)),
            "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ece": expected_calibration_error(p, y),
            "n": int(m.sum()),
        }
    out["mean_over_conditions"] = {
        k: float(np.mean([out[c][k] for c in out]))
        for k in ("acc", "uar", "f1", "ece")
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/IEEE/data_aligned")
    p.add_argument("--cache", type=str, default="/root/autodl-tmp/TPR_FER/cale_conditions.json")
    p.add_argument("--variant", type=str, default="euler")
    p.add_argument("--train_conditions", type=str, default="standard")
    p.add_argument("--val_conditions", type=str, default="standard")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="runs/cale")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    instances = scan_instances(args.root)
    cond_map = build_condition_map(instances, args.cache)
    train, val, test = split_instances(instances, seed=0)
    train_tf, test_tf = make_tf()

    train_conds = [c.strip() for c in args.train_conditions.split(",") if c.strip()]
    val_conds = [c.strip() for c in args.val_conditions.split(",") if c.strip()]
    train_ds = CALEDataset(train, cond_map, train_conds, train_tf)
    val_ds = CALEDataset(val, cond_map, val_conds, test_tf)
    test_ds = CALEDataset(test, cond_map, CONDITIONS, test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    model = build_model(args.variant, len(CLASSES), pretrained=True).to(device)
    out_dir = Path(args.out_dir) / f"{args.variant}_{'-'.join(train_conds)}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"variant={args.variant} train={train_conds} val={val_conds}")
    print(f"instances train/val/test={len(train)}/{len(val)}/{len(test)}")
    print(f"images train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"train conditions={condition_counts(train_ds)}")
    print(f"params={count_parameters(model)/1e6:.2f}M device={device}")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = {"acc": -1.0}
    for epoch in range(args.epochs):
        if uses_tpr(args.variant):
            if epoch < 5:
                gamma = 0.0
            elif epoch < 10:
                gamma = (epoch - 5 + 1) / 5
            else:
                gamma = 1.0
            set_tpr_gamma(model, gamma)
        else:
            gamma = 0.0

        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        if uses_tpr(args.variant):
            set_tpr_gamma(model, 1.0)
        val_metrics = eval_by_condition(model, val_loader, device)
        key = val_metrics["mean_over_conditions"]["acc"]
        print(f"epoch {epoch+1:03d}/{args.epochs} gamma={gamma:.2f} "
              f"loss={running/max(1,seen):.4f} val_acc={key:.4f} "
              f"({time.time()-t0:.1f}s)")

        if key > best["acc"]:
            best = {"acc": key, "epoch": epoch + 1, "val": val_metrics}
            torch.save({"model": model.state_dict(), "variant": args.variant,
                        "args": vars(args), "classes": CLASSES}, out_dir / "best.pth")
            (out_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    if uses_tpr(args.variant):
        set_tpr_gamma(model, 1.0)
    test_metrics = eval_by_condition(model, test_loader, device)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
