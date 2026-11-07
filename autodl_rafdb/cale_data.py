# -*- coding: utf-8 -*-
"""CALE-style self-built dataset: 7 classes, 216 instances/class, 20 views each.

The 20 views = 5 geometric transforms x 4 imaging conditions.  The suffix order
is shuffled, so we infer the condition from simple image statistics per instance:

    dark    : 5 lowest mean brightness
    bright  : 5 highest mean brightness
    lowres  : among the remaining 10, 5 lowest Laplacian energy
    standard: the other 5

This is only used to define splits/evaluation conditions; it is not a label.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "natural"]
CONDITIONS = ["dark", "bright", "lowres", "standard"]
COND_TO_ID = {c: i for i, c in enumerate(CONDITIONS)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def instance_of(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def _stats(path: Path) -> Tuple[float, float]:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    mean = float(g.mean())
    lap = float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                       - g[1:-1, :-2] - g[1:-1, 2:]).mean())
    return mean, lap


def scan_instances(root: str | Path):
    root = Path(root)
    instances = []
    for ci, cls in enumerate(CLASSES):
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(d)
        groups = defaultdict(list)
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                groups[instance_of(p)].append(p)
        for inst, files in groups.items():
            if len(files) != 20:
                raise RuntimeError(f"{inst}: expected 20 files, got {len(files)}")
            instances.append({"instance": inst, "class": cls, "label": ci, "files": files})
    return instances


def build_condition_map(instances, cache_path: str | Path | None = None) -> Dict[str, str]:
    if cache_path is not None and Path(cache_path).is_file():
        return json.loads(Path(cache_path).read_text(encoding="utf-8"))

    cond = {}
    for i, item in enumerate(instances):
        stats = [(*_stats(p), p) for p in item["files"]]
        original = [s for s in stats if "_original" in s[2].name]
        if len(original) != 1:
            raise RuntimeError(f"{item['instance']}: expected one _original file")
        rest = [s for s in stats if "_original" not in s[2].name]
        # 19 generated views: 5 dark, 5 bright, 9 middle (5 lowres + 4 standard)
        by_mean = sorted(rest, key=lambda x: x[0])
        dark = by_mean[:5]
        bright = by_mean[-5:]
        middle = by_mean[5:14]
        by_lap = sorted(middle, key=lambda x: x[1])
        lowres = by_lap[:5]
        standard = by_lap[5:] + original
        if len(dark) != 5 or len(bright) != 5 or len(lowres) != 5 or len(standard) != 5:
            raise RuntimeError(f"bad condition split for {item['instance']}")
        for group, name in ((dark, "dark"), (bright, "bright"),
                            (lowres, "lowres"), (standard, "standard")):
            for _, _, p in group:
                cond[str(p)] = name
        if (i + 1) % 100 == 0:
            print(f"  conditions {i+1}/{len(instances)}")

    if cache_path is not None:
        Path(cache_path).write_text(json.dumps(cond, indent=2), encoding="utf-8")
    return cond


def split_instances(instances, seed: int = 0, train_ratio=0.70, val_ratio=0.15):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for x in instances:
        by_class[x["label"]].append(x)
    train, val, test = [], [], []
    for cls, xs in by_class.items():
        rng.shuffle(xs)
        n = len(xs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train.extend(xs[:n_train])
        val.extend(xs[n_train:n_train + n_val])
        test.extend(xs[n_train + n_val:])
    return train, val, test


class CALEDataset(Dataset):
    def __init__(self, instances, cond_map, conditions: Sequence[str],
                 transform=None):
        self.transform = transform
        self.conditions = set(conditions)
        self.samples = []
        for item in instances:
            for p in item["files"]:
                c = cond_map[str(p)]
                if c in self.conditions:
                    self.samples.append((p, item["label"], COND_TO_ID[c],
                                         item["instance"]))
        if not self.samples:
            raise RuntimeError("empty CALEDataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, cond, inst = self.samples[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
        return im, label, cond, idx

    @property
    def instances(self):
        return sorted({s[3] for s in self.samples})


def condition_counts(dataset: CALEDataset):
    out = {c: 0 for c in CONDITIONS}
    for _, _, cid, _ in dataset.samples:
        out[CONDITIONS[cid]] += 1
    return out


__all__ = [
    "CLASSES", "CONDITIONS", "COND_TO_ID", "scan_instances",
    "build_condition_map", "split_instances", "CALEDataset", "condition_counts",
]
