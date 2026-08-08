"""
self-contained olmo3 arch.

deltas vs llama/qwen3 that matter for this repo:

- post-norm residual: x = x + Norm(sublayer(x))  (no pre-norm on the residual
  stream; norms sit on the sublayer *output* before the add)
- qk-norm over the full projected width *before* the head split (qwen3 norms
  per-head after the split)
- hybrid attention: every 4th layer is full causal, the rest are sliding-
  window causal (default window 4096) — matches allenai Olmo-3 layer_types
- untied embeddings, rope base 500_000

no framework imports — torch + stdlib only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def default_layer_types(num_layers: int) -> list[str]:
    """3 sliding + 1 full, repeating — the olmo-3 7B pattern."""
    types: list[str] = []
    for i in range(num_layers):
        types.append("full_attention" if (i + 1) % 4 == 0 else "sliding_attention")
    return types


class Olmo3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 500_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, False)
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]
        return cos, sin

    def forward(
        self, position_ids: torch.Tensor, cacheable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cacheable:
            return self._cos_sin(position_ids)

        key = (position_ids.shape[1], position_ids.device)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cos_sin(position_ids)
            self._cache[key] = cached
        return cached


class Olmo3Attention(nn.Module):
    """
    GQA with full-width q/k RMSNorm (applied to the projected tensor before
    the head reshape) and optional sliding-window causal masking.
    """

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None = None,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0 and num_heads >= num_kv_heads, \
            "num_heads must be a positive multiple of num_kv_heads"

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim

        self.q_proj = nn.Linear(emb_dim, q_out, bias=False)
        self.k_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.v_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, emb_dim, bias=False)

        # full projected width, not per-head — matches HF Olmo3Attention
        self.q_norm = nn.RMSNorm(q_out)
        self.k_norm = nn.RMSNorm(kv_out)

    def split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, n_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)

    def scaled_self_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        S = Q.size(-2)
        # window >= seq (or unset) is plain causal — keep the flash/mem-efficient
        # path and never materialize an S×S mask.
        windowed = (
            self.sliding_window is not None and self.sliding_window < S
        )
        if attention_mask is None and not windowed:
            return F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, enable_gqa=True
            )

        # bool semantics for SDPA: True = "attend to this position"
        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
        if windowed:
            # ban keys more than `sliding_window` tokens behind the query
            idx = torch.arange(S, device=Q.device)
            band = idx[None, :] > idx[:, None] - self.sliding_window
            causal = causal & band

        if attention_mask is None:
            attn_mask = causal[None, None]  # (1, 1, S, S)
        else:
            key_keep = attention_mask[:, None, None, :].bool()
            attn_mask = causal[None, None] & key_keep

        return F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, enable_gqa=True
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # norm the full projected vectors, *then* split into heads
        Q = self.split_heads(self.q_norm(self.q_proj(x)), self.num_heads)
        K = self.split_heads(self.k_norm(self.k_proj(x)), self.num_kv_heads)
        V = self.split_heads(self.v_proj(x), self.num_kv_heads)

        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        attn_output = self.scaled_self_attention(Q, K, V, attention_mask)
        return self.o_proj(self.combine_heads(attn_output))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Olmo3TransformerBlock(nn.Module):
    """
    post-norm block: sublayer first, RMSNorm on its output, then residual add.
    no pre-norm on the residual stream.
    """

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.attn = Olmo3Attention(
            num_heads, emb_dim, num_kv_heads, head_dim, sliding_window
        )
        self.ffn = SwiGLU(emb_dim, ffn_dim)
        self.post_attention_norm = nn.RMSNorm(emb_dim)
        self.post_ffn_norm = nn.RMSNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x = x + Norm(Attn(x))
        x = x + self.post_attention_norm(self.attn(x, cos, sin, attention_mask))
        # x = x + Norm(FFN(x))
        x = x + self.post_ffn_norm(self.ffn(x))
        return x


class Olmo3(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        vocab_size: int,
        ffn_dim: int,
        head_dim: int | None = None,
        rope_base: float = 500_000.0,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
    ):
        super().__init__()
        if head_dim is None:
            head_dim = emb_dim // num_heads

        if layer_types is None:
            layer_types = default_layer_types(num_layers)
        assert len(layer_types) == num_layers, \
            f"layer_types length {len(layer_types)} != num_layers {num_layers}"
        for t in layer_types:
            assert t in ("full_attention", "sliding_attention"), \
                f"unknown layer type {t!r}"

        # if sliding_window is None, every layer is full causal regardless of
        # layer_types — useful for short-seq training without the S×S mask.
        self.layer_types = list(layer_types)
        self.sliding_window = sliding_window

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Olmo3RotaryEmbedding(head_dim, base=rope_base)
        self.blocks = nn.ModuleList(
            Olmo3TransformerBlock(
                num_heads,
                emb_dim,
                num_kv_heads,
                head_dim,
                ffn_dim,
                sliding_window=(
                    sliding_window if lt == "sliding_attention" else None
                ),
            )
            for lt in self.layer_types
        )
        self.norm = nn.RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.RMSNorm) and module.weight is not None:
                nn.init.ones_(module.weight)

        residual_scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            block.ffn.down_proj.weight.data.mul_(residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape

        if attention_mask is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            cacheable = True
        else:
            attention_mask = attention_mask.bool()
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(~attention_mask, 0)
            cacheable = False

        x = self.token_emb(x)
        cos, sin = self.rope(position_ids, cacheable=cacheable)
        hidden_states: list[torch.Tensor] = [x] if return_hidden_states else []

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)

        if return_final_hidden:
            return self.norm(x)

        logits = self.lm_head(self.norm(x))
        if return_hidden_states:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    # compact ~1B-class shape (7B is 32L/4096d/11008ffn — too heavy for a
    # construct-and-count smoke on a single workstation)
    config = {
        "num_layers": 16,
        "num_heads": 16,
        "num_kv_heads": 16,
        "emb_dim": 2048,
        "ffn_dim": 8192,
        "vocab_size": 100_278,
        "sliding_window": 4096,
    }

    model = Olmo3(**config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    print("layer_types:", model.layer_types)
