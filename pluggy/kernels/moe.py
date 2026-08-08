"""
dropless MoE expert FFN via sorted grouped GEMM.

replaces the per-expert python loop in `Qwen3MoeExperts`:

    for e in hit_experts:
        tok = hidden[token_idx_e]
        y  = down(silu(gate(tok)) * up(tok)) * routing_weight
        out.index_add_(token_idx_e, y)

with a single sort + two `F.grouped_mm` calls (gate_up, down) + one scatter.
empty experts are expressed as repeated offsets so the grouped kernel still
sees a dense (E,)-sized batch of groups.

weight layout matches HF / our qwen3_moe pack:
    gate_up_proj : (E, 2I, H)   # F.linear(x, W) with W=(2I,H)
    down_proj    : (E, H,  I)   # F.linear(y, W) with W=(H,I)

falls back to the loop path when grouped_mm is unavailable (cpu, old torch)
or when the token count is tiny enough that launch overhead dominates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pluggy.kernels.swiglu import swiglu_mul

# below this many token-expert pairs the loop is usually cheaper than sort+launch
_GROUPED_MM_MIN_PAIRS = 64


def _has_grouped_mm() -> bool:
    return hasattr(F, "grouped_mm") and torch.cuda.is_available()


def _expert_ffn_loop(
    hidden: torch.Tensor,       # (T, H)
    top_indices: torch.Tensor,  # (T, K)
    top_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor, # (E, 2I, H)
    down_proj: torch.Tensor,    # (E, H, I)
) -> torch.Tensor:
    """original correctness path — one GEMM pair per hit expert."""
    T, H = hidden.shape
    E = gate_up_proj.shape[0]
    out = torch.zeros(T, H, device=hidden.device, dtype=hidden.dtype)

    expert_mask = F.one_hot(top_indices, num_classes=E)  # (T, K, E)
    expert_mask = expert_mask.permute(2, 1, 0)           # (E, K, T)
    hit = expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=False).flatten()

    gu = gate_up_proj.to(dtype=hidden.dtype)
    dn = down_proj.to(dtype=hidden.dtype)
    for e in hit.tolist():
        k_pos, token_idx = torch.where(expert_mask[e])
        if token_idx.numel() == 0:
            continue
        tok = hidden[token_idx]
        gate, up = F.linear(tok, gu[e]).chunk(2, dim=-1)
        y = F.linear(swiglu_mul(gate, up), dn[e])
        y = y * top_weights[token_idx, k_pos, None]
        out.index_add_(0, token_idx, y.to(dtype=out.dtype))
    return out


def _expert_ffn_grouped(
    hidden: torch.Tensor,       # (T, H)
    top_indices: torch.Tensor,  # (T, K)
    top_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor, # (E, 2I, H)
    down_proj: torch.Tensor,    # (E, H, I)
) -> torch.Tensor:
    """
    sort tokens by expert, run two grouped GEMMs, scatter-add back.

    each of the T tokens is expanded to K (token, expert) pairs so a token
    routed to several experts appears once per selection — routing weights
    then scale each contribution before the scatter.
    """
    T, H = hidden.shape
    E, two_I, _ = gate_up_proj.shape
    I = two_I // 2
    K = top_indices.shape[1]
    device = hidden.device
    dtype = hidden.dtype

    # (T*K,) flat routing
    token_idx = (
        torch.arange(T, device=device)
        .unsqueeze(1)
        .expand(T, K)
        .reshape(-1)
    )
    expert_idx = top_indices.reshape(-1)
    route_w = top_weights.reshape(-1)

    # sort pairs by expert id so each expert's tokens are contiguous
    order = torch.argsort(expert_idx, stable=True)
    sorted_expert = expert_idx[order]
    sorted_token = token_idx[order]
    sorted_w = route_w[order]
    sorted_h = hidden[sorted_token]  # (T*K, H)

    # offs[e] = end index of expert e in the sorted list (int32 for grouped_mm)
    counts = torch.bincount(sorted_expert, minlength=E)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)

    # cast weights once to activation dtype (bf16 under autocast)
    gu = gate_up_proj.to(dtype=dtype)          # (E, 2I, H)
    dn = down_proj.to(dtype=dtype)             # (E, H, I)
    # grouped_mm does mat_a @ mat_b[g]; F.linear(x, W) = x @ W.T
    # so mat_b for gate_up is (E, H, 2I) = gu.transpose(-2, -1)
    # and for down is (E, I, H) = dn.transpose(-2, -1)
    gate_up = F.grouped_mm(sorted_h, gu.transpose(-2, -1), offs=offs)  # (P, 2I)
    gate, up = gate_up.chunk(2, dim=-1)
    mid = swiglu_mul(gate, up)                                         # (P, I)
    down = F.grouped_mm(mid, dn.transpose(-2, -1), offs=offs)          # (P, H)
    down = down * sorted_w.unsqueeze(1).to(dtype=dtype)

    out = torch.zeros(T, H, device=device, dtype=dtype)
    out.index_add_(0, sorted_token, down)
    return out


def moe_expert_ffn(
    hidden: torch.Tensor,       # (T, H)
    top_indices: torch.Tensor,  # (T, K) int64
    top_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor, # (E, 2I, H)
    down_proj: torch.Tensor,    # (E, H, I)
    *,
    use_grouped_mm: bool | None = None,
) -> torch.Tensor:
    """
    dropless top-k expert FFN.

    use_grouped_mm:
        None  -> auto (cuda + enough tokens)
        True  -> force grouped path (raises if unavailable)
        False -> force loop path
    """
    assert hidden.dim() == 2
    assert top_indices.shape == top_weights.shape
    assert top_indices.shape[0] == hidden.shape[0]
    assert gate_up_proj.dim() == 3 and down_proj.dim() == 3

    T, K = top_indices.shape
    pairs = T * K
    if use_grouped_mm is None:
        use_grouped_mm = (
            _has_grouped_mm()
            and hidden.is_cuda
            and pairs >= _GROUPED_MM_MIN_PAIRS
        )

    if use_grouped_mm:
        if not _has_grouped_mm():
            raise RuntimeError(
                "moe_expert_ffn: use_grouped_mm=True but F.grouped_mm is unavailable"
            )
        return _expert_ffn_grouped(
            hidden, top_indices, top_weights, gate_up_proj, down_proj
        )
    return _expert_ffn_loop(
        hidden, top_indices, top_weights, gate_up_proj, down_proj
    )
