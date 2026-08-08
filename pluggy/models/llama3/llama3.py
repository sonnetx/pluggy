"""
self-contained llama 3 / 3.2 arch.

structurally qwen3 minus qk-norm, with untied embeddings and the llama3
rope base (500_000). head_dim defaults to emb_dim // num_heads (llama ties
them; qwen3 decouples). no framework imports — torch + stdlib only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pluggy.kernels.rope import apply_rotary
from pluggy.kernels.swiglu import swiglu_mul


class Llama3RotaryEmbedding(nn.Module):
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


class Llama3GroupQueryAttention(nn.Module):
    """GQA without qk-norm — the main structural delta vs qwen3 attention."""

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0 and num_heads >= num_kv_heads, \
            "num_heads must be a positive multiple of num_kv_heads"

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = head_dim if head_dim is not None else emb_dim // num_heads
        self.num_kv_heads = num_kv_heads

        q_out = num_heads * self.head_dim
        kv_out = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(emb_dim, q_out, bias=False)
        self.k_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.v_proj = nn.Linear(emb_dim, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, emb_dim, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        n_heads = x.shape[-1] // self.head_dim
        return x.reshape(*x.shape[:2], n_heads, self.head_dim).transpose(1, 2)

    def split_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
        if attention_mask is None:
            return F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, enable_gqa=True
            )

        S = Q.size(-2)
        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
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
        Q = self.split_heads(self.q_proj(x))
        K = self.split_kv_heads(self.k_proj(x))
        V = self.split_kv_heads(self.v_proj(x))

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
        return self.down_proj(swiglu_mul(self.gate_proj(x), self.up_proj(x)))


class Llama3TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        ffn_dim: int | None = None,
    ):
        super().__init__()
        self.attn = Llama3GroupQueryAttention(num_heads, emb_dim, num_kv_heads, head_dim)
        self.norm1 = nn.RMSNorm(emb_dim)
        self.norm2 = nn.RMSNorm(emb_dim)

        if ffn_dim is None:
            # llama-style: 8/3 expansion, rounded up to multiple of 256
            ffn_dim = 256 * ((int(8 * emb_dim / 3) + 255) // 256)
        self.ffn = SwiGLU(emb_dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Llama3(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        vocab_size: int,
        head_dim: int | None = None,
        ffn_dim: int | None = None,
        rope_base: float = 500_000.0,
    ):
        super().__init__()
        if head_dim is None:
            head_dim = emb_dim // num_heads

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Llama3RotaryEmbedding(head_dim, base=rope_base)
        self.blocks = nn.ModuleList(
            Llama3TransformerBlock(num_heads, emb_dim, num_kv_heads, head_dim, ffn_dim)
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(emb_dim)
        # untied: llama3 keeps a separate lm_head (unlike qwen3)
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
    # llama 3.2 1B-class shape
    config = {
        "num_layers": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
        "emb_dim": 2048,
        "head_dim": 64,
        "ffn_dim": 8192,
        "vocab_size": 128_256,
    }

    model = Llama3(**config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    # ~1.50B with untied emb+lm_head (tied HF 1B checkpoint is ~1.24B)

