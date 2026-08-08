"""
fused RoPE apply: (x * cos) + (rotate_half(x) * sin).

`dynamic=True` so varying seq_len across steps doesn't thrash the compile
cache. cos/sin use the models' existing broadcast layout (B, 1, S, D)
against Q/K of shape (B, H, S, D).
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)
