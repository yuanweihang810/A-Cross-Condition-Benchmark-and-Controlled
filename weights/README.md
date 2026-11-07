# Selected checkpoints

These are the seed-0 checkpoints used for the qualitative model comparison. They are not the full 5-seed/3-seed result set.

| File | Dataset / setting | Variant | Notes |
|---|---|---|---|
| rafdb_tpr_seed0.pth | RAF-DB | tpr | main TPR model |
| rafdb_euler_seed0.pth | RAF-DB | euler | standard residual baseline |
| rafdb_se_seed0.pth | RAF-DB | se | SE-ResNet-18 |
| rafdb_cbam_seed0.pth | RAF-DB | cbam | CBAM-ResNet-18 |
| rafdb_resnet34_seed0.pth | RAF-DB | resnet34 | deeper baseline |
| cale_m_tpr_seed0.pth | CC-FER-M | tpr | all conditions in training |
| cale_m_euler_seed0.pth | CC-FER-M | euler | all conditions in training |
| cale_u_tpr_seed0.pth | CC-FER-U | tpr | train on standard only |
| cale_u_euler_seed0.pth | CC-FER-U | euler | train on standard only |

Each file is a PyTorch checkpoint with keys: model, variant, args, classes.

    import torch
    ckpt = torch.load("weights/rafdb_tpr_seed0.pth", map_location="cpu")
    print(ckpt["variant"], ckpt["args"]["seed"])

For evaluation:

    python autodl_rafdb/eval_checkpoint.py --root data/RAF-DB       --ckpt weights/rafdb_tpr_seed0.pth --gamma 1.0

The CC-FER checkpoints use the same ResNet-18 backbone; pass the matching train_cale.py settings when evaluating.
