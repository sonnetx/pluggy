"""
silu(gate) * up  /  gelu-tanh(gate) * up for SwiGLU / GeGLU FFNs.

kept as plain ops (no torch.compile): the MoE path calls this with a
different token count every step, and a compiled dynamic=False kernel
blows the recompile cache. under per-block compile the dense FFN already
fuses this elementwise into the surrounding GEMMs.
"""

import torch
import torch.nn.functional as F


def swiglu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, same shape as inputs."""
    return F.silu(gate) * up


def geglu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gelu-tanh(gate) * up — gemma4's activation."""
    return F.gelu(gate, approximate="tanh") * up
