"""
RMSNorm as a free function.

`nn.RMSNorm` is already well-tuned inside compiled blocks; this form is
handy for scale-free norms (gemma4 v_norm) and for call sites outside a
Module. fp32 accumulate, cast back — same recipe as the HF implementations.
`dynamic=True` tolerates varying batch/seq.
"""

import torch


@torch.compile(dynamic=True)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: (..., D)
    weight: (D,) or None for scale-free norm (gemma4 v_norm).
    """
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(orig_dtype)
