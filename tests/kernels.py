"""
correctness checks for pluggy.kernels.

compares each fused op against a naive reference under bf16 autocast-style
inputs, including gradients where they matter (moe, softcap).

run: uv run tests/kernels.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pluggy.kernels.moe import moe_expert_ffn, _has_grouped_mm
from pluggy.kernels.rms_norm import rms_norm
from pluggy.kernels.rope import apply_rotary, rotate_half
from pluggy.kernels.softcap import logit_softcap
from pluggy.kernels.swiglu import geglu_mul, swiglu_mul


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a - b).float().norm() / b.float().norm().clamp_min(1e-12)
    ).item()


def test_swiglu_mul(device: str = "cuda") -> None:
    torch.manual_seed(0)
    g = torch.randn(32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    u = torch.randn(32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    # warm compile so we aren't comparing against a first-run graph artifact
    _ = swiglu_mul(g.detach(), u.detach())
    out = swiglu_mul(g, u)
    ref = F.silu(g) * u
    assert out.shape == ref.shape
    # compiled bf16 silu can differ by a few ulps from eager
    assert _rel(out, ref) < 3e-2, _rel(out, ref)

    (out.float().sum()).backward()
    g2 = g.detach().clone().requires_grad_(True)
    u2 = u.detach().clone().requires_grad_(True)
    (F.silu(g2) * u2).float().sum().backward()
    assert _rel(g.grad, g2.grad) < 5e-2
    assert _rel(u.grad, u2.grad) < 5e-2
    print("swiglu_mul ok")


def test_geglu_mul(device: str = "cuda") -> None:
    torch.manual_seed(0)
    g = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    u = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    _ = geglu_mul(g, u)
    out = geglu_mul(g, u)
    ref = F.gelu(g, approximate="tanh") * u
    assert _rel(out, ref) < 3e-2, _rel(out, ref)
    print("geglu_mul ok")


def test_apply_rotary(device: str = "cuda") -> None:
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 32, 64
    x = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    # classic half-and-half cos/sin
    pos = torch.arange(S, device=device).float()
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))
    freqs = pos[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[None, None, :, :].to(torch.bfloat16)
    sin = emb.sin()[None, None, :, :].to(torch.bfloat16)

    _ = apply_rotary(x, cos, sin)
    out = apply_rotary(x, cos, sin)
    ref = (x * cos.to(x.dtype)) + (rotate_half(x) * sin.to(x.dtype))
    assert _rel(out, ref) < 3e-2, _rel(out, ref)
    print("apply_rotary ok")


def test_rms_norm(device: str = "cuda") -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16)
    w = torch.ones(128, device=device, dtype=torch.float32)
    _ = rms_norm(x, w, eps=1e-6)
    out = rms_norm(x, w, eps=1e-6)
    ref = F.rms_norm(x.float(), (128,), w, eps=1e-6).to(x.dtype)
    assert _rel(out, ref) < 3e-2, _rel(out, ref)

    out_ns = rms_norm(x, None, eps=1e-6)
    xf = x.float()
    ref_ns = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    assert _rel(out_ns, ref_ns) < 3e-2, _rel(out_ns, ref_ns)
    print("rms_norm ok")


def test_logit_softcap(device: str = "cuda") -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
    cap = 30.0
    y = logit_softcap(x, cap)
    ref = cap * torch.tanh(x / cap)
    assert _rel(y, ref) < 1e-5

    y.sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    (cap * torch.tanh(x2 / cap)).sum().backward()
    assert _rel(x.grad, x2.grad) < 1e-5
    x_det = x.detach()
    assert torch.equal(logit_softcap(x_det, None), x_det)
    print("logit_softcap ok")


def _moe_reference(hidden, top_indices, top_weights, gate_up, down):
    return moe_expert_ffn(
        hidden, top_indices, top_weights, gate_up, down, use_grouped_mm=False
    )


def test_moe_grouped_matches_loop(device: str = "cuda") -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("moe skip (no cuda)")
        return
    torch.manual_seed(0)
    T, H, I, E, K = 128, 64, 32, 8, 2
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    # biased routing so some experts empty, some heavy
    logits = torch.randn(T, E, device=device)
    probs = F.softmax(logits, dim=-1)
    top_w, top_i = torch.topk(probs, K, dim=-1)
    top_w = (top_w / top_w.sum(-1, keepdim=True)).to(torch.bfloat16)

    gate_up = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32) * 0.02
    down = torch.randn(E, H, I, device=device, dtype=torch.float32) * 0.02
    gate_up = gate_up.clone().requires_grad_(True)
    down = down.clone().requires_grad_(True)

    # loop reference
    h1 = hidden.detach().clone().requires_grad_(True)
    gu1 = gate_up.detach().clone().requires_grad_(True)
    dn1 = down.detach().clone().requires_grad_(True)
    out1 = _moe_reference(h1, top_i, top_w, gu1, dn1)
    out1.float().pow(2).mean().backward()

    if device == "cpu" or not _has_grouped_mm():
        print("moe loop-only ok (no grouped_mm)")
        return

    h2 = hidden.detach().clone().requires_grad_(True)
    gu2 = gate_up.detach().clone().requires_grad_(True)
    dn2 = down.detach().clone().requires_grad_(True)
    out2 = moe_expert_ffn(h2, top_i, top_w, gu2, dn2, use_grouped_mm=True)
    out2.float().pow(2).mean().backward()

    print(
        f"moe: rel_out={_rel(out2, out1):.2e} "
        f"dh={_rel(h2.grad, h1.grad):.2e} "
        f"dgu={_rel(gu2.grad, gu1.grad):.2e} "
        f"ddn={_rel(dn2.grad, dn1.grad):.2e}"
    )
    assert _rel(out2, out1) < 3e-2
    assert _rel(h2.grad, h1.grad) < 5e-2
    assert _rel(gu2.grad, gu1.grad) < 5e-2
    assert _rel(dn2.grad, dn1.grad) < 5e-2
    print("moe grouped==loop ok")


def test_moe_end_to_end_model() -> None:
    """qwen3_moe still constructs and runs through the kernel path."""
    from pluggy.models.builder import build_model
    from pluggy.objectives.autoregressive import ARObjective

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(
        "qwen3_moe",
        dict(
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            emb_dim=64,
            head_dim=16,
            vocab_size=200,
            moe_ffn_dim=32,
            ffn_dim=128,
            num_experts=4,
            top_k=2,
        ),
    )
    m.init_weights()
    m.to(device)
    x = torch.randint(0, 200, (2, 32), device=device)
    logits = m(x)
    assert logits.shape == (2, 32, 200)
    batch = {"input_ids": x, "labels": torch.randint(0, 200, (2, 32), device=device)}
    loss = ARObjective(fused_linear_ce=False).compute_loss(m, batch)
    loss.backward()
    assert torch.isfinite(loss)
    print("moe model e2e ok")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} grouped_mm={_has_grouped_mm()}")
    test_swiglu_mul(device)
    test_geglu_mul(device)
    test_apply_rotary(device)
    test_rms_norm(device)
    test_logit_softcap(device)
    test_moe_grouped_matches_loop(device)
    test_moe_end_to_end_model()
    print("all kernel checks passed")
