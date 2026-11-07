# -*- coding: utf-8 -*-
"""Evaluate a trained TPR checkpoint at gamma=0 (Euler inference) and gamma=1.

This tests whether tangent-path *training* helps even when inference collapses
back to the standard residual (zero inference overhead).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tpr import set_tpr_gamma  # noqa: E402
from train_rafdb import build_model, build_transforms, evaluate  # noqa: E402
from rafdb_data import RAFDBDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/IEEE")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image_subdir", type=str, default="aligned",
                   help="aligned or original")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--corruption", type=str, default="none",
                   choices=["none", "occlude", "noise", "blur"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    variant = ckpt["variant"]
    model = build_model(variant, 7, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    root = Path(args.root)
    _, test_tf = build_transforms()

    if args.corruption != "none":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ops = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC)]
        if args.corruption == "blur":
            ops.append(transforms.GaussianBlur(kernel_size=7, sigma=2.0))
        ops.append(transforms.CenterCrop(224))
        ops.append(transforms.ToTensor())
        if args.corruption == "occlude":
            ops.append(transforms.RandomErasing(
                p=1.0, scale=(0.08, 0.18), ratio=(0.3, 3.3), value=0.0))
        if args.corruption == "noise":
            ops.append(transforms.Lambda(
                lambda x: torch.clamp(x + 0.08 * torch.randn_like(x), 0.0, 1.0)))
        ops.append(normalize)
        test_tf = transforms.Compose(ops)
        torch.manual_seed(0)

    test_ds = RAFDBDataset(
        root / "Image" / args.image_subdir,
        root / "EmoLabel/list_patition_label.txt",
        "test",
        test_tf,
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    results = {}
    gamma = float(args.gamma)
    set_tpr_gamma(model, gamma)
    metrics = evaluate(model, loader, device, amp=True, return_preds=False)
    results[f"gamma_{gamma:g}"] = metrics
    print(f"image_subdir={args.image_subdir} corruption={args.corruption} "
          f"gamma={gamma:g}  {json.dumps(metrics)}")

    out = Path(args.ckpt).with_name(
        f"gamma_eval_{args.image_subdir}_{args.corruption}_g{gamma:g}.json")
    out.write_text(json.dumps({"variant": variant, "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
