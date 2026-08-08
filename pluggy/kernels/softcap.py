"""
logit softcapping: y = cap * tanh(x / cap).

used by gemma4 on the lm_head output. written as an autograd.Function so the
backward is the analytic d/dx = (1 - tanh^2(x/cap)) = sech^2, without
re-materializing an extra tanh in a way that confuses compile. forward keeps
the tanh value for backward.
"""

import torch


class _LogitSoftcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cap: float) -> torch.Tensor:
        # compute in the input dtype; tanh is fine in bf16 for logits
        t = torch.tanh(x / cap)
        ctx.cap = cap
        ctx.save_for_backward(t)
        return t * cap

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (t,) = ctx.saved_tensors
        # d/dx [cap * tanh(x/cap)] = 1 - tanh^2(x/cap)
        return grad_out * (1.0 - t * t), None


def logit_softcap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    """no-op when cap is None or non-positive."""
    if cap is None or cap <= 0:
        return x
    return _LogitSoftcap.apply(x, float(cap))
