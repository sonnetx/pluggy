"""
small fused ops used by the trainer / models.

style matches `pluggy/loss/fused_linear_ce.py`: pure torch, `torch.compile`
where it helps, no liger/apex. the MoE path uses `F.grouped_mm` (torch ≥2.9)
with a loop fallback so cpu / older builds still work.

public surface is intentionally tiny — import from here, not from the
submodules, so call sites stay stable if internals move.
"""

from pluggy.kernels.moe import moe_expert_ffn
from pluggy.kernels.rms_norm import rms_norm
from pluggy.kernels.rope import apply_rotary
from pluggy.kernels.softcap import logit_softcap
from pluggy.kernels.swiglu import swiglu_mul

__all__ = [
    "apply_rotary",
    "logit_softcap",
    "moe_expert_ffn",
    "rms_norm",
    "swiglu_mul",
]
