# AutoDL RAF-DB 运行说明

这份代码按你给的路径写好了：

    /root/autodl-tmp/IEEE/Image/aligned
    /root/autodl-tmp/IEEE/EmoLabel/list_patition_label.txt

注意文件名是官方的 list_patition_label.txt（官方拼写就是 patition，不是 partition）。

---

## 0. 先确认目录

SSH 上服务器后：

    cd /root/autodl-tmp/IEEE
    ls Image/aligned | head
    wc -l EmoLabel/list_patition_label.txt
    head -3 EmoLabel/list_patition_label.txt

你应该看到类似：

    train_00001_aligned.jpg
    ...
    train_00001.jpg 1
    test_00001.jpg 1

如果 aligned 文件名没有 _aligned 后缀，代码也能自动匹配，不用改。

---

## 1. 上传代码

把本地的 TPR_FER 整个目录传到：

    /root/autodl-tmp/TPR_FER

可以用 AutoDL 的 JupyterLab 上传，或者本地 scp：

    scp -r TPR_FER root@<你的实例地址>:/root/autodl-tmp/

然后：

    cd /root/autodl-tmp/TPR_FER
    ls
    ls autodl_rafdb

---

## 2. 环境

AutoDL 的镜像一般已经带 torch 和 torchvision。先检查：

    python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

如果 torch 正常，只装 sklearn：

    pip install -r autodl_rafdb/requirements_autodl.txt

如果终端提示 python: command not found（AutoDL 有时要手动激活 conda），用绝对路径：

    /root/miniconda3/bin/python --version

export PATH=/root/miniconda3/bin:$PATH

或者在运行脚本时指定：

    PYTHON=/root/miniconda3/bin/python bash autodl_rafdb/run_smoke.sh

如果 torch 没有，按 AutoDL 镜像说明装一个 CUDA 版本；不要直接 pip install torch 把镜像里的版本覆盖坏。

---

## 3. 数据检查（必须先跑）

    cd /root/autodl-tmp/TPR_FER
    python autodl_rafdb/check_data.py --root /root/autodl-tmp/IEEE

正常应该看到：

    train: total=12271 found=12271 missing=0
    test : total= 3068 found= 3068 missing=0

并打印 7 类分布。RAF-DB 官方训练集大约 12,271 张，测试集 3,068 张；如果你的数字不同，先检查 label 文件是不是官方版本。

如果报 missing：

- 看它打印的第一个缺失文件名；
- 去 Image/aligned 看实际文件名；
- 代码会自动尝试 name、name_aligned.jpg、name_aligned.png；
- 如果你的文件名规则更特殊，改 rafdb_data.py 里的 _candidate_paths 即可。

---

## 4. Smoke test（1 分钟，只验证接线）

    bash autodl_rafdb/run_smoke.sh

它会：

- 检查数据；
- 用 64 张图、1 epoch、no-pretrained 跑一遍 TPR；
- 输出到 runs/smoke/tpr_seed0。

看到 loss、acc 打印出来，且不报错，就说明数据、模型、AMP、保存都通了。这个结果没有意义，不要写进论文。

---

## 5. 正式 pilot（4 个格子）

先跑 seed 0，4 个变体，30 epoch，确认时间和显存：

    SEEDS="0" EPOCHS=30 bash autodl_rafdb/run_pilot.sh

默认 batch_size=64，workers=4。如果 OOM：

    BATCH_SIZE=32 SEEDS="0" EPOCHS=30 bash autodl_rafdb/run_pilot.sh

如果 DataLoader 报错：

    WORKERS=0 SEEDS="0" EPOCHS=30 bash autodl_rafdb/run_pilot.sh

seed 0 跑通、结果合理后，再跑 3 seeds：

    SEEDS="0 1 2" EPOCHS=30 bash autodl_rafdb/run_pilot.sh

四个变体是：

- euler：标准 ResNet-18
- orth：K=1，切空间单点
- heun：K=2，环境空间两点
- tpr：K=2，切空间路径 + final projection

训练脚本默认从官方 train 切 10% 做 val（--val_fraction 0.1），best 按 val acc 选，test 只在最后评一次。这样不会用测试集调参。

---

## 6. 看结果

跑完会自动执行 summarize_pilot.py，也会在输出目录生成：

    runs/pilot/summary.md
    runs/pilot/summary.csv

手动重跑：

    python autodl_rafdb/summarize_pilot.py --out_dir /root/autodl-tmp/TPR_FER/runs/pilot

summary.md 里有：

- 每个 variant 的 acc / uar / f1 / ece 均值±标准差；
- 2×2 interaction： (TPR - Orth) - (Heun - Euler)；
- TPR - max(Orth, Heun)。

判断标准：

1. TPR 比 Euler 高 >= 0.5%，至少 2/3 seed 更高；
2. TPR 比 max(Orth, Heun) 高 >= 0.2%；
3. 2×2 interaction 为正，说明“投影 × 积分”是超加性的。

满足两条再投全量；只满足第 1 条，按设计报告 Plan B 走；一条都不满足，停。

---

## 7. 预计时间

单卡 3090，RAF-DB 12,271 张，ResNet-18 224×224：

- 1 epoch 大约 1–2 分钟（AMP、batch 64）；
- 1 个 run 30 epoch 大约 45–75 分钟；
- TPR 约 1.25x，约 60–90 分钟；
- 4 变体 × 3 seed = 12 runs，约 10–15 小时。

4090 会更快；如果按小时计费，先跑 SEEDS=0 的 4 个 run（约 3–5 小时），看结果再决定。

---

## 8. 常见问题

### 8.1 pretrained 下载失败

默认会下载 ImageNet 权重。如果服务器不能访问 torchvision 权重：

    python autodl_rafdb/train_rafdb.py ... --no-pretrained

但 no-pretrained 的绝对值会低很多，只适合 smoke；正式实验最好用预训练。

也可以手动把 resnet18 权重放到 ~/.cache/torch/hub/checkpoints/。

### 8.2 显存不够

- BATCH_SIZE=32 或 16；
- 或把 TPR 只放 stage 4：--variant tpr_stage4；
- 或 --no-amp 关闭 AMP（更慢、更占显存，不推荐）。

### 8.3 训练很慢

- 确认 torch.cuda.is_available() 是 True；
- 确认 --amp 默认开启；
- workers 不要超过 CPU 核数；
- 如果数据在普通磁盘而不是 autodl-tmp，先拷贝到 /root/autodl-tmp。

### 8.4 loss 不降 / acc 随机

- 先看 lr：默认 3e-4 AdamW；如果 no-pretrained，可能太大，试 1e-4；
- 确认 label 是 1..7 且已经减 1；
- 确认图像不是全黑/全白；
- 先跑 euler 10 epoch，确认标准 ResNet 能到 80% 左右，再跑 TPR。

### 8.5 断点续跑

当前脚本每次 run 独立，不会自动续跑。如果断了：

- 已经完成的 run 有 best.pth 和 best_metrics.json，summary 会读；
- 重跑时跳过已有 best_metrics.json 的 run，或者手动删掉对应的 runs/pilot/xxx_seedY 再跑。

---

## 9. 文件说明

| 文件 | 作用 |
|---|---|
| rafdb_data.py | 读官方 label 文件、匹配 aligned 图片、Dataset |
| check_data.py | 数据完整性检查 |
| train_rafdb.py | 训练/评估/保存 best 和 predictions |
| summarize_pilot.py | 汇总 4 格结果 + 2×2 interaction |
| run_smoke.sh | 1 分钟接线测试 |
| run_pilot.sh | 4 变体 × 多 seed 批量跑 |

训练输出每个 run 包含：

    best.pth
    best_metrics.json
    history.json
    predictions.npz
    args.json

predictions.npz 里有 logits、probs、labels、paths，后面做 radial/curvature 分析和 Grad-CAM 可以直接用。

---

## 10. 重要提醒

- 先跑 check_data 和 smoke，不要一上来就跑 12 个 run。
- 训练脚本默认从官方 train 切 10% 做 val（--val_fraction 0.1），best 按 val acc 选，test 只在最后评一次。不要用 --val_fraction 0 出论文数字。
- 论文里报告 mean±std、参数量、FLOPs、推理时间；不要只报最好 seed。
- 如果 TPR 没有超过 max(Orth, Heun)，不要硬写；按设计报告第 7 节切 Plan B。
