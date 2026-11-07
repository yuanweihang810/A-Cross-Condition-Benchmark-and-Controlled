# -*- coding: utf-8 -*-
"""Self-test for the TPR module.  Runs on CPU in a few seconds."""

import torch
import torch.nn as nn
import torchvision.models as tvm

from tpr import (
    TPRBlock,
    convert_to_tpr,
    count_parameters,
    set_tpr_gamma,
    tangent_project,
    tpr_resnet18,
)


def test_projection():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8)
    f = torch.randn(2, 16, 8, 8)
    fp = tangent_project(f, x, mode="global")
    # Global means must be orthogonal.
    gx = x.mean(dim=(2, 3))
    gp = fp.mean(dim=(2, 3))
    dot = (gx * gp).sum(dim=1).abs().max().item()
    assert dot < 1e-5, f"global tangency failed: {dot}"
    # Idempotence.
    fp2 = tangent_project(fp, x, mode="global")
    err = (fp - fp2).abs().max().item()
    assert err < 1e-6, f"idempotence failed: {err}"
    # Mean-only projection: same global tangency, less spatial distortion.
    fm = tangent_project(f, x, mode="mean")
    gm = fm.mean(dim=(2, 3))
    dot_m = (gx * gm).sum(dim=1).abs().max().item()
    assert dot_m < 1e-5, f"mean tangency failed: {dot_m}"
    fm2 = tangent_project(fm, x, mode="mean")
    err_m = (fm - fm2).abs().max().item()
    assert err_m < 1e-6, f"mean idempotence failed: {err_m}"
    # Partial projection.
    f_half = tangent_project(f, x, mode="global", rho=0.5)
    assert not torch.allclose(f_half, fp, atol=1e-7)
    # local projection should be per-pixel orthogonal.
    fl = tangent_project(f, x, mode="local")
    local_dot = (fl * x).sum(dim=1).abs().max().item()
    assert local_dot < 1e-5, f"local tangency failed: {local_dot}"
    print("  projection: PASS")

def test_non_additivity():
    """TPR is not Orth + Heun; the second projection base is x1, not x."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8)
    f1 = torch.randn(2, 16, 8, 8)
    f2 = torch.randn(2, 16, 8, 8)
    u0 = tangent_project(f1, x, mode="global")
    x1 = x + u0
    # TPR: project the second field at x1.
    u1_tpr = tangent_project(f2, x1, mode="global")
    tpr = 0.5 * (u0 + u1_tpr)
    # Naive composition: project the second field at x.
    u1_naive = tangent_project(f2, x, mode="global")
    naive = 0.5 * (u0 + u1_naive)
    diff = (tpr - naive).abs().max().item()
    assert diff > 1e-4, f"expected non-additivity, got {diff}"
    print(f"  non-additivity: PASS (max diff {diff:.4f})")


def test_gamma_zero_equivalence():
    torch.manual_seed(0)
    net_std = tvm.resnet18(weights=None)
    net_tpr = tvm.resnet18(weights=None)
    net_tpr.load_state_dict(net_std.state_dict())
    convert_to_tpr(net_tpr, stages=(3, 4), projection="global", K=2)
    set_tpr_gamma(net_tpr, 0.0)
    net_std.eval()
    net_tpr.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y_std = net_std(x)
        y_tpr = net_tpr(x)
    err = (y_std - y_tpr).abs().max().item()
    assert err < 1e-5, f"gamma=0 equivalence failed: {err}"
    print("  gamma=0 equivalence: PASS")


def test_forward_backward():
    torch.manual_seed(0)
    net = tpr_resnet18(num_classes=7, pretrained=False, stages=(3, 4), K=2)
    net.train()
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    assert y.shape == (2, 7), y.shape
    loss = y.square().mean()
    loss.backward()
    n_grad = sum(1 for p in net.parameters() if p.grad is not None)
    n_param = sum(1 for p in net.parameters() if p.requires_grad)
    assert n_grad == n_param, (n_grad, n_param)
    # All gradients finite.
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    print(f"  forward/backward: PASS ({n_param/1e6:.2f}M params)")


def test_parameter_count():
    std = tvm.resnet18(weights=None)
    tpr = tpr_resnet18(num_classes=1000, pretrained=False, stages=(3, 4), K=2)
    # Same params when h is a fixed buffer.
    n_std = count_parameters(std)
    n_tpr = count_parameters(tpr)
    assert n_std == n_tpr, (n_std, n_tpr)
    # learn_h adds one scalar per converted block (2 blocks in stages 3 and 4
    # have no downsample: layer3[1], layer4[1]).
    tpr_h = tpr_resnet18(num_classes=1000, pretrained=False, stages=(3, 4), K=2, learn_h=True)
    n_h = count_parameters(tpr_h)
    assert n_h - n_std == 2, (n_std, n_h)
    print(f"  parameter count: PASS ({n_std/1e6:.2f}M; learn_h +{n_h-n_std})")


def test_k_variants():
    torch.manual_seed(0)
    for K in (1, 2, 3):
        net = tpr_resnet18(num_classes=7, pretrained=False, stages=(4,), K=K)
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(1, 3, 64, 64))
        assert y.shape == (1, 7)
    print("  K=1/2/3: PASS")


if __name__ == "__main__":
    print("TPR self-test")
    test_projection()
    test_non_additivity()
    test_gamma_zero_equivalence()
    test_forward_backward()
    test_parameter_count()
    test_k_variants()
    print("ALL PASS")
