# CC-FER + RAF-DB results (current)

Date: 2026-11-07  
Status: RAF-DB 5 seeds; CC-FER-M 3 seeds; CC-FER-U 3 seeds (SE: 2 seeds); ResNet-34/CBAM 1 seed (screening). No jobs running.

---

## 1. RAF-DB, 5 seeds, official split

| Method | Params | Acc | UAR | ECE |
|---|---:|---:|---:|---:|
| Euler (ResNet-18) | 11.18M | **86.49 ± 0.34** | 76.49 ± 0.87 | 5.91 ± 0.49 |
| OrthRes | 11.18M | 85.74 ± 0.73 | 74.86 ± 1.14 | 5.21 ± 0.39 |
| HeunNet | 11.18M | 85.61 ± 0.90 | 75.77 ± 1.67 | **4.85 ± 0.33** |
| TPR | 11.18M | 86.11 ± 0.54 | 76.48 ± 1.00 | 5.06 ± 0.56 |
| TPR-nofinal | 11.18M | 85.91 ± 0.54 | **76.59 ± 1.06** | 4.84 ± 0.41 |
| ResNet-34 (1 seed) | 21.30M | 86.60 | **78.01** | 4.97 |
| SE-ResNet-18 (1 seed) | 11.27M | 86.41 | 77.15 | 5.27 |
| CBAM-ResNet-18 (1 seed) | 11.27M | 85.27 | 75.23 | 5.57 |

2x2 interaction: **+1.25 ± 1.05** (positive across 5 seeds), but TPR - max(Orth, Heun) = **+0.15 ± 0.52** — not a net win.

**RAF-DB conclusion:** standard Euler is the strongest clean baseline. TPR is second in the geometry family, but below Euler. No method dominates all metrics: Heun/TPR-nofinal have the best ECE; ResNet-34 has the best UAR at 2x cost.

---

## 2. CALE/CC-FER, 1 seed, train all 4 conditions, instance-disjoint test

Protocol:

- 1,512 instances = 216/class × 7 classes.
- 20 views/instance = 5 geometric × 4 conditions (dark / bright / lowres / standard).
- Train/val/test = 1,057 / 224 / 231 instances, split by instance.
- Train on all 4 conditions; test on held-out instances, report each condition.

| Method | seeds | standard | dark | bright | lowres | mean | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TPR** | 3 | **73.02 ± 0.90** | 72.61 ± 0.61 | **72.87 ± 0.84** | 71.83 ± 0.57 | **72.58 ± 0.46** | 10.46 ± 0.66 |
| Euler | 3 | 72.47 ± 1.10 | **73.42 ± 0.72** | 71.83 ± 0.83 | **71.86 ± 0.88** | 72.40 ± 0.57 | **5.82 ± 1.25** |
| SE | 3 | 71.92 ± 0.43 | 71.80 ± 0.96 | 71.00 ± 0.72 | 71.92 ± 0.90 | 71.66 ± 0.64 | 10.33 ± 1.53 |
| ResNet-34 | 1 | 72.03 | 71.00 | 70.48 | 71.43 | 71.23 | 13.22 |
| CBAM | 1 | 70.82 | 71.34 | 71.52 | 68.83 | 70.63 | 14.28 |

**Who wins what (3 seeds for TPR/Euler/SE):**

- **TPR**: standard (+0.55), bright (+1.04), mean (+0.18), same params as ResNet-18, no gate.
- **Euler**: dark (+0.81), calibration (ECE 5.82 vs 10.46).
- **SE**: no condition win; lowres ties TPR/Euler.
- **ResNet-34 / CBAM**: no condition win; deeper/gated models overfit this dataset.

**CALE conclusion:** TPR is numerically best on mean (72.58 vs 72.40), standard and bright; Euler is better on dark and calibration. The mean gap (0.18) is within seed noise, so the honest claim is **statistical tie overall + condition-specific wins**, not a significant SOTA gain.

### CC-FER-U: unseen-condition generalization (train standard only)

| Method | Seeds | standard | dark | bright | lowres | mean | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPR | 3 | **72.41 ± 1.51** | 61.50 ± 0.94 | **60.35 ± 1.31** | 56.97 ± 1.59 | 62.81 ± 0.86 | 10.66 ± 0.83 |
| Euler | 3 | 72.38 ± 0.90 | **63.46 ± 0.83** | 57.78 ± 1.10 | **57.72 ± 1.35** | **62.84 ± 0.37** | **5.00 ± 0.01** |
| SE | 2 | 71.00 ± 0.78 | 58.70 ± 0.69 | 51.73 ± 3.25 | 59.83 ± 0.52 | 60.31 ± 1.31 | 7.02 ± 0.06 |

In the true unseen-condition setting, TPR and Euler are tied on the mean (62.81 vs 62.84). TPR is **+2.57 on bright** and +0.03 on standard, while Euler is +1.96 on dark and +0.75 on lowres. This is the clearest condition-specific result in the paper: TPR specializes in bright/standard illumination; Euler is stronger in dark/low-resolution images and calibration. SE is far behind on the mean.

---

## 3. Paper claim (honest)

> Under a controlled RAF-DB protocol and a new cross-condition protocol, no residual family dominates all metrics. TPR is numerically best on the cross-condition mean (72.58 vs 72.40), standard and bright, but the mean gap is within seed noise; Euler is better on dark images and calibration. On clean RAF-DB, Euler is ahead (86.49 vs 86.11). The result is a condition-dependent trade-off and motivates a cross-condition benchmark rather than a single “universal” residual module.

This is a **benchmark + controlled study**, not a SOTA method paper.

---

## 4. What is still running

All core runs are finished. ResNet-34 and CBAM remain one-seed screening results; they can be extended to three seeds before submission if desired.
