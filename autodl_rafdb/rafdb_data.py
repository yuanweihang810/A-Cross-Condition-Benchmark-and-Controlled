# -*- coding: utf-8 -*-
"""RAF-DB official label-file dataset for AutoDL.

Expected layout (the paths you gave):

    /root/autodl-tmp/IEEE/
      Image/aligned/train_00001_aligned.jpg
      Image/aligned/test_00001_aligned.jpg
      EmoLabel/list_patition_label.txt

The official label file lines look like:

    train_00001.jpg 1
    test_00001.jpg 1

Labels are 1..7.  We map them to 0..6 for PyTorch.  The class-name order
below follows the RAF-DB README (1 surprise, 2 fear, 3 disgust, 4 happy,
5 sad, 6 angry, 7 neutral).  If your copy uses a different order, accuracy
is unaffected; only the printed class names / confusion matrix would change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


# RAF-DB official order: label 1..7 -> index 0..6.
RAFDB_CLASS_NAMES = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral",
]


def _candidate_paths(image_dir: Path, name: str) -> List[Path]:
    """RAF-DB aligned images usually have an _aligned suffix."""
    p = Path(name)
    stem, suffix = p.stem, p.suffix
    candidates = [
        image_dir / name,
        image_dir / f"{stem}_aligned{suffix}",
        image_dir / f"{stem}_aligned.jpg",
        image_dir / f"{stem}_aligned.png",
    ]
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            out.append(c)
    return out


def resolve_image(image_dir: Path, name: str) -> Optional[Path]:
    for c in _candidate_paths(image_dir, name):
        if c.is_file():
            return c
    return None


def parse_label_file(label_file: str | Path) -> List[Tuple[str, int, str]]:
    """Return [(image_name, label_0_6, split), ...] sorted by file order."""
    label_file = Path(label_file)
    samples: List[Tuple[str, int, str]] = []
    with label_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                label = int(parts[1])
            except ValueError:
                continue
            if not (1 <= label <= 7):
                raise ValueError(
                    f"label out of range at {label_file}:{line_no}: {line!r}"
                )
            lower = name.lower()
            if lower.startswith("train"):
                split = "train"
            elif lower.startswith("test"):
                split = "test"
            else:
                raise ValueError(
                    f"cannot infer split from filename at {label_file}:{line_no}: {name!r}. "
                    "Expected train_*.jpg or test_*.jpg."
                )
            samples.append((name, label - 1, split))
    if not samples:
        raise RuntimeError(f"no samples parsed from {label_file}")
    return samples


class RAFDBDataset(Dataset):
    """RAF-DB dataset from the official label file.

    Returns:
        image tensor, label (0..6), sample index
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_file: str | Path,
        split: str,
        transform=None,
        skip_missing: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)
        self.split = split
        self.transform = transform
        self.class_names = list(RAFDB_CLASS_NAMES)

        all_samples = parse_label_file(self.label_file)
        selected = [s for s in all_samples if s[2] == split]
        if max_samples is not None:
            selected = selected[: int(max_samples)]

        self.samples: List[Tuple[Path, int, str]] = []
        missing: List[str] = []
        for name, label, _ in selected:
            path = resolve_image(self.image_dir, name)
            if path is None:
                missing.append(name)
                if skip_missing:
                    continue
                continue
            self.samples.append((path, label, name))

        if missing and not skip_missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} / {len(selected)} images in split={split} were not found "
                f"under {self.image_dir}. First missing:\n{preview}\n"
                "Check the _aligned suffix or run check_data.py."
            )
        if not self.samples:
            raise RuntimeError(f"no images found for split={split}")

    @classmethod
    def from_resolved(cls, samples, transform=None, class_names=None):
        """Build a dataset from already-resolved (path, label, name) tuples.

        Used for deterministic train/val splits without re-scanning the disk.
        """
        obj = cls.__new__(cls)
        obj.image_dir = None
        obj.label_file = None
        obj.split = "custom"
        obj.transform = transform
        obj.class_names = list(class_names or RAFDB_CLASS_NAMES)
        obj.samples = list(samples)
        if not obj.samples:
            raise RuntimeError("from_resolved received an empty sample list")
        return obj

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, name = self.samples[index]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
        return im, label, index


def class_counts(dataset: RAFDBDataset) -> Dict[str, int]:
    counts = {name: 0 for name in dataset.class_names}
    for _, label, _ in dataset.samples:
        counts[dataset.class_names[label]] += 1
    return counts

def stratified_split(labels: Sequence[int], val_fraction: float, seed: int = 0):
    """Return (train_indices, val_indices), stratified by class."""
    import numpy as np

    labels = np.asarray(list(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * float(val_fraction)))
        if val_fraction > 0 and len(idx) > 1:
            n_val = max(1, min(n_val, len(idx) - 1))
        else:
            n_val = 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def scan_dataset(
    image_dir: str | Path,
    label_file: str | Path,
    max_report: int = 10,
) -> dict:
    """Lightweight integrity report used by check_data.py."""
    image_dir = Path(image_dir)
    label_file = Path(label_file)
    samples = parse_label_file(label_file)
    report = {
        "image_dir": str(image_dir),
        "label_file": str(label_file),
        "total_lines": len(samples),
        "splits": {},
        "missing": {},
        "resolved_examples": [],
    }
    for split in ("train", "test"):
        subset = [s for s in samples if s[2] == split]
        missing = []
        counts = {name: 0 for name in RAFDB_CLASS_NAMES}
        for name, label, _ in subset:
            path = resolve_image(image_dir, name)
            if path is None:
                missing.append(name)
            else:
                counts[RAFDB_CLASS_NAMES[label]] += 1
                if len(report["resolved_examples"]) < max_report:
                    report["resolved_examples"].append(str(path))
        report["splits"][split] = {
            "count": len(subset),
            "found": len(subset) - len(missing),
            "missing": len(missing),
            "class_counts": counts,
        }
        report["missing"][split] = missing[:max_report]
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/root/autodl-tmp/IEEE")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--label_file", type=str, default="")
    args = p.parse_args()
    root = Path(args.root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "Image/aligned"
    label_file = Path(args.label_file) if args.label_file else root / "EmoLabel/list_patition_label.txt"
    print(json.dumps(scan_dataset(image_dir, label_file), indent=2, ensure_ascii=False))
