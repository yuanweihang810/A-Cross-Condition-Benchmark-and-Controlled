# -*- coding: utf-8 -*-
"""Train TPR / baselines on RAF-DB from the official label file.

AutoDL example:

    cd /root/autodl-tmp/TPR_FER
    python autodl_rafdb/check_data.py --root /root/autodl-tmp/IEEE

    python autodl_rafdb/train_rafdb.py \
        --root /root/autodl-tmp/IEEE \
        --variant tpr --seed 0 --epochs 30 \
        --out_dir /root/autodl-tmp/TPR_FER/runs/pilot

Four core cells:
    euler, orth, heun, tpr
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from torchvision.models import resnet18, resnet34

# Make "from tpr import ..." work when this file is run from any cwd.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tpr import convert_to_tpr, count_parameters, set_tpr_gamma  # noqa: E402
from nrr import convert_to_nrr  # noqa: E402
from baselines import convert_to_attention  # noqa: E402
from rafdb_data import RAFDB_CLASS_NAMES, RAFDBDataset, class_counts, stratified_split  # noqa: E402


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic but a bit slower.  Comment out if you need speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TwoStepBlock(nn.Module):
    """Shared-weight two-step Euler baseline: x -> B(x) -> B(B(x)).

    Same parameter count as one block, 2x compute.  It is the strongest
    compute/parameter-matched baseline for TPR.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(self.block(x))


def convert_shared2(model: nn.Module) -> nn.Module:
    for name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, name)
        for i, block in enumerate(layer):
            layer[i] = TwoStepBlock(block)
    return model


def build_model(variant: str, num_classes: int, pretrained: bool):
    weights = "IMAGENET1K_V1" if pretrained else None
    if variant == "resnet34":
        model = resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if variant == "euler":
        return model
    if variant == "se":
        return convert_to_attention(model, "se")
    if variant == "cbam":
        return convert_to_attention(model, "cbam")
    if variant == "shared2":
        return convert_shared2(model)
    if variant == "orth":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=1, h=1.0)
        return model
    if variant == "heun":
        convert_to_tpr(model, stages=(3, 4), projection="none", K=2, h=1.0)
        return model
    if variant == "tpr":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_k3":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=3, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_local":
        convert_to_tpr(model, stages=(3, 4), projection="local", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_nofinal":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=1.0,
                       final_projection=False)
        return model
    if variant == "tpr_learnh":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=1.0,
                       final_projection=True, learn_h=True)
        return model
    if variant == "tpr_stage4":
        convert_to_tpr(model, stages=(4,), projection="global", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_rho50":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=1.0,
                       final_projection=True, rho=0.5)
        return model
    if variant == "tpr_mean":
        convert_to_tpr(model, stages=(3, 4), projection="mean", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "orth_mean":
        convert_to_tpr(model, stages=(3, 4), projection="mean", K=1, h=1.0)
        return model
    if variant == "tpr_h05":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=0.5,
                       final_projection=True)
        return model
    if variant == "tpr_nofinal_h05":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=0.5,
                       final_projection=False)
        return model
    if variant == "tpr_nofinal_h025":
        convert_to_tpr(model, stages=(3, 4), projection="global", K=2, h=0.25,
                       final_projection=False)
        return model
    if variant == "tpr_stage2":
        convert_to_tpr(model, stages=(2,), projection="global", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_stage12":
        convert_to_tpr(model, stages=(1, 2), projection="global", K=2, h=1.0,
                       final_projection=True)
        return model
    if variant == "tpr_nofinal_stage2":
        convert_to_tpr(model, stages=(2,), projection="global", K=2, h=1.0,
                       final_projection=False)
        return model
    if variant == "nrr":
        convert_to_nrr(model, stages=(3, 4), lam=0.5)
        return model
    if variant == "nrr_lam10":
        convert_to_nrr(model, stages=(3, 4), lam=1.0)
        return model
    if variant == "nrr_lam025":
        convert_to_nrr(model, stages=(3, 4), lam=0.25)
        return model
    raise ValueError(f"unknown variant: {variant}")


def uses_tpr(variant: str) -> bool:
    # heun also uses TPRBlock (projection="none"), so it needs the same
    # gamma warm-up schedule.  Missing "heun" here made it start at gamma=1
    # and collapse in the first pilot.
    return (variant.startswith("tpr") or variant.startswith("nrr")
            or variant in ("orth", "orth_mean", "heun"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, test_tf


def build_loaders(args):
    root = Path(args.root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "Image/aligned"
    label_file = Path(args.label_file) if args.label_file else root / "EmoLabel/list_patition_label.txt"
    train_tf, test_tf = build_transforms()

    max_samples = args.max_samples if args.max_samples > 0 else None
    train_full = RAFDBDataset(
        image_dir, label_file, "train", train_tf,
        skip_missing=args.skip_missing, max_samples=max_samples,
    )
    test_ds = RAFDBDataset(
        image_dir, label_file, "test", test_tf,
        skip_missing=args.skip_missing, max_samples=max_samples,
    )

    # Model selection must NOT use the test set.  Hold out a stratified
    # fraction of the official training split as validation.
    if args.val_fraction > 0.0 and len(train_full) > 7:
        labels = [s[1] for s in train_full.samples]
        tr_idx, va_idx = stratified_split(labels, args.val_fraction, args.seed)
        train_ds = RAFDBDataset.from_resolved(
            [train_full.samples[i] for i in tr_idx], transform=train_tf
        )
        val_ds = RAFDBDataset.from_resolved(
            [train_full.samples[i] for i in va_idx], transform=test_tf
        )
    else:
        print("WARNING: val_fraction=0; using the test set as validation. "
              "Do not use this mode for paper numbers.")
        train_ds = train_full
        val_ds = test_ds

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@torch.inference_mode()
def evaluate(model, loader, device, amp: bool, max_batches: int = 0, return_preds: bool = False):
    model.eval()
    logits_all = []
    labels_all = []
    indices_all = []
    for batch_idx, (x, y, idx) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        labels_all.append(y)
        indices_all.append(idx)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    indices = torch.cat(indices_all).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    pred = probs.argmax(axis=1)

    metrics = {
        "acc": float(accuracy_score(labels, pred)),
        "uar": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, labels),
        "n": int(len(labels)),
    }
    if return_preds:
        return metrics, logits.numpy(), labels, indices, probs
    return metrics


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def tpr_gamma_for_epoch(epoch: int, args) -> float:
    if not uses_tpr(args.variant):
        return 0.0
    if epoch < args.warmup_epochs:
        return 0.0
    if epoch < args.warmup_epochs + args.ramp_epochs:
        return (epoch - args.warmup_epochs + 1) / max(1, args.ramp_epochs)
    return 1.0


def make_scheduler(optimizer, args):
    def lr_lambda(epoch: int) -> float:
        if epoch < args.lr_warmup_epochs:
            return (epoch + 1) / max(1, args.lr_warmup_epochs)
        progress = (epoch - args.lr_warmup_epochs) / max(1, args.epochs - args.lr_warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/IEEE")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--label_file", type=str, default="")
    p.add_argument("--variant", type=str, default="tpr")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--lr_warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="epochs of standard residual before TPR ramp")
    p.add_argument("--ramp_epochs", type=int, default=5,
                   help="epochs to ramp TPR gamma from 0 to 1")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="stratified fraction of the official training split used for model selection")
    p.add_argument("--out_dir", type=str, default="runs/pilot")
    p.add_argument("--max_samples", type=int, default=0,
                   help="debug only: take first N samples per split")
    p.add_argument("--max_batches", type=int, default=0,
                   help="debug only: limit eval/train batches")
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--random_gamma", action="store_true",
                   help="stochastic tangent-path training: sample gamma=0/1 per batch")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_samples = min(args.max_samples or 64, 64)
        args.max_batches = 5
        args.workers = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)
    model = build_model(args.variant, len(RAFDB_CLASS_NAMES), args.pretrained).to(device)

    out_dir = Path(args.out_dir) / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"variant   : {args.variant}")
    print(f"device    : {device}  amp={amp}")
    print(f"params    : {count_parameters(model)/1e6:.2f}M")
    print(f"train     : {len(train_ds)}  {class_counts(train_ds)}")
    print(f"val       : {len(val_ds)}  {class_counts(val_ds)}")
    print(f"test      : {len(test_ds)}  {class_counts(test_ds)}")
    print(f"out_dir   : {out_dir}")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best = {"acc": -1.0}
    history = []
    for epoch in range(args.epochs):
        gamma = tpr_gamma_for_epoch(epoch, args)
        if args.random_gamma and uses_tpr(args.variant):
            gamma = 0.5
        if uses_tpr(args.variant):
            set_tpr_gamma(model, gamma)

        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for batch_idx, (x, y, _) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            if args.random_gamma and uses_tpr(args.variant):
                set_tpr_gamma(model, 1.0 if random.random() < 0.5 else 0.0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step()
        train_loss = running / max(1, seen)
        if uses_tpr(args.variant):
            set_tpr_gamma(model, 1.0)
        metrics = evaluate(model, val_loader, device, amp, args.max_batches)
        metrics.update({"epoch": epoch + 1, "gamma": gamma, "train_loss": train_loss,
                        "lr": optimizer.param_groups[0]["lr"], "split": "val"})
        history.append(metrics)
        dt = time.time() - t0
        print(f"epoch {epoch+1:03d}/{args.epochs} gamma={gamma:.2f} lr={metrics['lr']:.2e} "
              f"loss={train_loss:.4f} val_acc={metrics['acc']:.4f} "
              f"val_uar={metrics['uar']:.4f} val_f1={metrics['f1']:.4f} "
              f"val_ece={metrics['ece']:.4f} ({dt:.1f}s)")

        if metrics["acc"] > best["acc"]:
            best = dict(metrics)
            torch.save({
                "model": model.state_dict(),
                "variant": args.variant,
                "args": vars(args),
                "class_names": RAFDB_CLASS_NAMES,
                "metrics": best,
            }, out_dir / "best.pth")
            (out_dir / "best_metrics.json").write_text(
                json.dumps(best, indent=2), encoding="utf-8"
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Reload best and save test predictions for later analysis.
    if (out_dir / "best.pth").is_file():
        ckpt = torch.load(out_dir / "best.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        if uses_tpr(args.variant):
            set_tpr_gamma(model, 1.0)
        test_metrics, logits, labels, indices, probs = evaluate(
            model, test_loader, device, amp, args.max_batches, return_preds=True
        )
        test_metrics["best_val"] = best
        (out_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        paths = [str(test_ds.samples[i][0]) for i in indices]
        np.savez_compressed(
            out_dir / "predictions.npz",
            logits=logits,
            probs=probs,
            labels=labels,
            indices=indices,
            paths=np.asarray(paths),
            class_names=np.asarray(RAFDB_CLASS_NAMES),
        )
        print("BEST VAL", json.dumps(best, indent=2))
        print("FINAL TEST", json.dumps(test_metrics, indent=2))
    else:
        print("No checkpoint saved (max_batches too small?).")


if __name__ == "__main__":
    main()
