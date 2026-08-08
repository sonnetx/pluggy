"""
self-contained gemma 4 dense text decoder.

text-only path of google's gemma-4 (no vision/audio/moe). structural deltas
vs the llama/qwen blocks already in this repo:

- sandwich RMSNorm: pre + post on both attn and ffn
  (x = x + post(sublayer(pre(x))))
- hybrid attention: 5:1 sliding:full by default (last layer always full);
  sliding layers use head_dim, full layers use global_head_dim
- dual RoPE: default theta=10k on sliding; proportional partial-RoPE
  (factor 0.25, theta=1e6) on full layers
- per-head q/k RMSNorm + unscaled v RMSNorm; attention scale is 1.0
  (qk-norm carries the scale — do NOT pass 1/sqrt(d) into SDPA)
- GeGLU with gelu-tanh (not SwiGLU/silu)
- embedding scaled by sqrt(emb_dim); tied lm_head by default
- optional final logit softcapping (tanh)
- optional KV sharing across the last num_kv_shared_layers
- optional per-layer embeddings (PLE) for the E2B/E4B edge variants
- optional attention_k_eq_v (reuse K as V on full layers) and double-wide
  MLP on KV-shared layers

no framework imports — torch + stdlib only.
"""

from __future__ import annotations

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


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def default_layer_types(num_layers: int, pattern: int = 6) -> list[str]:
    """5:1 sliding:full when pattern=6; last layer forced to full_attention."""
    types = [
        "sliding_attention" if (i + 1) % pattern else "full_attention"
        for i in range(num_layers)
    ]
    if types:
        types[-1] = "full_attention"
    return types


def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    # gemma4 v_norm is scale-free RMSNorm
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class Gemma4RotaryEmbedding(nn.Module):
    """
    one rope table. `proportional=True` builds partial-RoPE inv_freq padded
    with zeros so cos/sin stay head_dim-wide (identity on the unrotated dims).
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        partial_rotary_factor: float = 1.0,
        proportional: bool = False,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim

        if proportional and partial_rotary_factor < 1.0:
            rope_angles = int(partial_rotary_factor * head_dim // 2)
            inv_freq_rot = 1.0 / (
                base
                ** (
                    torch.arange(0, 2 * rope_angles, 2).float() / head_dim
                )
            )
            nope = head_dim // 2 - rope_angles
            if nope > 0:
                inv_freq = torch.cat(
                    [inv_freq_rot, torch.zeros(nope, dtype=torch.float32)]
                )
            else:
                inv_freq = inv_freq_rot
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, S) → freqs (B, S, D/2) → emb (B, S, D)
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        # (B, 1, S, D) so it broadcasts over heads
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


class GeGLU(nn.Module):
    """Gated GELU-tanh FFN (gemma4 hidden_activation=gelu_pytorch_tanh)."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x)
        )


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_type: str,
        sliding_window: int | None,
        is_kv_shared_layer: bool,
        store_full_length_kv: bool,
        attention_k_eq_v: bool,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0 and num_heads >= num_kv_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_type = layer_type
        self.sliding_window = sliding_window if layer_type == "sliding_attention" else None
        self.is_kv_shared_layer = is_kv_shared_layer
        self.store_full_length_kv = store_full_length_kv
        # k=v only on full-attention layers (matches HF)
        self.use_k_eq_v = bool(attention_k_eq_v) and layer_type == "full_attention"
        self.rms_norm_eps = rms_norm_eps
        # qk-norm absorbs scale — SDPA must use scale=1.0
        self.scale = 1.0

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim

        self.q_proj = nn.Linear(emb_dim, q_out, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(q_out, emb_dim, bias=False)

        if not is_kv_shared_layer:
            self.k_proj = nn.Linear(emb_dim, kv_out, bias=False)
            self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
            # v_norm is scale-free in HF; we apply rms_norm_no_scale at use-time
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(emb_dim, kv_out, bias=False)
            else:
                self.v_proj = None
        else:
            # weights live on the donor layer; this layer only has q/o
            self.k_proj = None
            self.k_norm = None
            self.v_proj = None

    def _split(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, n_heads, self.head_dim).transpose(1, 2)

    def _combine(self, x: torch.Tensor) -> torch.Tensor:
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
        windowed = self.sliding_window is not None and self.sliding_window < S
        kwargs = dict(enable_gqa=True, scale=self.scale)

        if attention_mask is None and not windowed:
            return F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, **kwargs
            )

        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
        if windowed:
            idx = torch.arange(S, device=Q.device)
            band = idx[None, :] > idx[:, None] - self.sliding_window
            causal = causal & band

        if attention_mask is None:
            attn_mask = causal[None, None]
        else:
            key_keep = attention_mask[:, None, None, :].bool()
            attn_mask = causal[None, None] & key_keep

        return F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        Q = self._split(self.q_proj(x), self.num_heads)
        Q = self.q_norm(Q)
        Q = apply_rotary(Q, cos, sin)

        if self.is_kv_shared_layer:
            K, V = shared_kv[self.layer_type]
            K = K.to(device=Q.device, dtype=Q.dtype)
            V = V.to(device=Q.device, dtype=Q.dtype)
        else:
            K = self._split(self.k_proj(x), self.num_kv_heads)
            if self.use_k_eq_v or self.v_proj is None:
                V = K
            else:
                V = self._split(self.v_proj(x), self.num_kv_heads)

            K = self.k_norm(K)
            K = apply_rotary(K, cos, sin)
            V = rms_norm_no_scale(V, self.rms_norm_eps)

            if self.store_full_length_kv:
                shared_kv[self.layer_type] = (K, V)

        out = self.scaled_self_attention(Q, K, V, attention_mask)
        return self.o_proj(self._combine(out))


class Gemma4TransformerBlock(nn.Module):
    """
    sandwich-norm block + optional PLE residual:
      x = x + post_attn(attn(pre_attn(x)))
      x = x + post_ffn(ffn(pre_ffn(x)))
      [optional PLE residual]
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        layer_type: str,
        sliding_window: int | None,
        is_kv_shared_layer: bool,
        store_full_length_kv: bool,
        attention_k_eq_v: bool,
        ple_dim: int = 0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = Gemma4Attention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_type=layer_type,
            sliding_window=sliding_window,
            is_kv_shared_layer=is_kv_shared_layer,
            store_full_length_kv=store_full_length_kv,
            attention_k_eq_v=attention_k_eq_v,
            rms_norm_eps=rms_norm_eps,
        )
        self.ffn = GeGLU(emb_dim, ffn_dim)
        self.input_norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)
        self.post_attn_norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)
        self.pre_ffn_norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)
        self.post_ffn_norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)

        self.ple_dim = ple_dim
        if ple_dim > 0:
            self.ple_gate = nn.Linear(emb_dim, ple_dim, bias=False)
            self.ple_proj = nn.Linear(ple_dim, emb_dim, bias=False)
            self.ple_norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
        ple_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.attn(self.input_norm(x), cos, sin, attention_mask, shared_kv)
        x = residual + self.post_attn_norm(x)

        residual = x
        x = self.ffn(self.pre_ffn_norm(x))
        x = residual + self.post_ffn_norm(x)

        if self.ple_dim > 0 and ple_input is not None:
            residual = x
            gated = gelu_pytorch_tanh(self.ple_gate(x)) * ple_input
            x = residual + self.ple_norm(self.ple_proj(gated))

        return x


class Gemma4(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        vocab_size: int,
        ffn_dim: int,
        head_dim: int = 256,
        global_head_dim: int = 512,
        num_global_kv_heads: int | None = None,
        sliding_window: int | None = 512,
        layer_types: list[str] | None = None,
        layer_pattern: int = 6,
        rope_theta_sliding: float = 10_000.0,
        rope_theta_full: float = 1_000_000.0,
        partial_rotary_factor: float = 0.25,
        final_logit_softcapping: float | None = 30.0,
        num_kv_shared_layers: int = 0,
        attention_k_eq_v: bool = False,
        use_double_wide_mlp: bool = False,
        hidden_size_per_layer_input: int = 0,
        vocab_size_per_layer_input: int | None = None,
        rms_norm_eps: float = 1e-6,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        if layer_types is None:
            layer_types = default_layer_types(num_layers, pattern=layer_pattern)
        assert len(layer_types) == num_layers, \
            f"layer_types length {len(layer_types)} != num_layers {num_layers}"
        for t in layer_types:
            assert t in ("full_attention", "sliding_attention"), f"bad layer type {t!r}"
        if layer_types and layer_types[-1] != "full_attention":
            layer_types = list(layer_types)
            layer_types[-1] = "full_attention"

        if num_global_kv_heads is None:
            num_global_kv_heads = num_kv_heads
        if vocab_size_per_layer_input is None:
            vocab_size_per_layer_input = vocab_size

        self.layer_types = list(layer_types)
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.ple_dim = int(hidden_size_per_layer_input)
        self.num_kv_shared_layers = num_kv_shared_layers
        self.emb_dim = emb_dim

        # scaled embeddings (gemma multiplies by sqrt(d)); tied lm_head by default
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.embed_scale = emb_dim ** 0.5

        # dual rope tables keyed by layer type
        self.rope_sliding = Gemma4RotaryEmbedding(
            head_dim, base=rope_theta_sliding, proportional=False
        )
        self.rope_full = Gemma4RotaryEmbedding(
            global_head_dim,
            base=rope_theta_full,
            partial_rotary_factor=partial_rotary_factor,
            proportional=True,
        )

        first_shared = num_layers - num_kv_shared_layers
        prev_types = (
            self.layer_types[:first_shared]
            if num_kv_shared_layers > 0
            else list(self.layer_types)
        )
        # a layer can only share K/V if a same-type donor exists in the
        # non-shared prefix (real gemma-4 configs guarantee this; tiny smoke
        # configs with e.g. a single trailing full layer do not).
        donor_idx: dict[str, int] = {}
        for i, lt in enumerate(prev_types):
            donor_idx[lt] = i

        blocks: list[Gemma4TransformerBlock] = []
        for i, lt in enumerate(self.layer_types):
            is_full = lt == "full_attention"
            hd = global_head_dim if is_full else head_dim
            n_kv = num_global_kv_heads if is_full else num_kv_heads
            in_shared_tail = num_kv_shared_layers > 0 and i >= first_shared
            is_shared = in_shared_tail and lt in donor_idx
            store_kv = (not is_shared) and donor_idx.get(lt) == i

            layer_ffn = ffn_dim * (2 if (use_double_wide_mlp and is_shared) else 1)

            blocks.append(
                Gemma4TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    num_kv_heads=n_kv,
                    head_dim=hd,
                    ffn_dim=layer_ffn,
                    layer_type=lt,
                    sliding_window=sliding_window,
                    is_kv_shared_layer=is_shared,
                    store_full_length_kv=store_kv,
                    attention_k_eq_v=attention_k_eq_v,
                    ple_dim=self.ple_dim,
                    rms_norm_eps=rms_norm_eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(emb_dim, eps=rms_norm_eps)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Per-Layer Embeddings (PLE) — E2B/E4B edge variants
        if self.ple_dim > 0:
            self.embed_tokens_per_layer = nn.Embedding(
                vocab_size_per_layer_input, num_layers * self.ple_dim
            )
            self.ple_embed_scale = self.ple_dim ** 0.5
            self.per_layer_model_projection = nn.Linear(
                emb_dim, num_layers * self.ple_dim, bias=False
            )
            self.per_layer_model_projection_scale = emb_dim ** -0.5
            self.per_layer_projection_norm = nn.RMSNorm(self.ple_dim, eps=rms_norm_eps)
            self.per_layer_input_scale = 2.0 ** -0.5

    def _ple_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """(B, S, L, ple_dim) combining token-identity + context projection."""
        B, S = input_ids.shape
        L = len(self.blocks)
        token_ple = (
            self.embed_tokens_per_layer(input_ids) * self.ple_embed_scale
        ).reshape(B, S, L, self.ple_dim)

        ctx = self.per_layer_model_projection(inputs_embeds)
        ctx = ctx * self.per_layer_model_projection_scale
        ctx = ctx.reshape(B, S, L, self.ple_dim)
        ctx = self.per_layer_projection_norm(ctx)
        return (ctx + token_ple) * self.per_layer_input_scale

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
            if block.ple_dim > 0:
                block.ple_proj.weight.data.mul_(residual_scale)

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.final_logit_softcapping
        if cap is None:
            return logits
        return torch.tanh(logits / cap) * cap

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape
        input_ids = x

        if attention_mask is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            cacheable = True
        else:
            attention_mask = attention_mask.bool()
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(~attention_mask, 0)
            cacheable = False

        h = self.token_emb(input_ids) * self.embed_scale

        ple = (
            self._ple_inputs(input_ids, h) if self.ple_dim > 0 else None
        )

        # precompute both rope tables once per forward
        rope_cache = {
            "sliding_attention": self.rope_sliding(position_ids, cacheable=cacheable),
            "full_attention": self.rope_full(position_ids, cacheable=cacheable),
        }

        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        hidden_states: list[torch.Tensor] = [h] if return_hidden_states else []

        for i, block in enumerate(self.blocks):
            lt = self.layer_types[i]
            cos, sin = rope_cache[lt]
            ple_i = ple[:, :, i, :] if ple is not None else None
            h = block(h, cos, sin, attention_mask, shared_kv, ple_i)
            if return_hidden_states:
                hidden_states.append(h)

        if return_final_hidden:
            # fused-CE path: softcap is intentionally skipped (it sits after lm_head)
            return self.norm(h)

        logits = self.lm_head(self.norm(h))
        logits = self._softcap(logits)
        if return_hidden_states:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    # compact shape exercising hybrid attn + dual rope + sandwich norms.
    # full E2B is 35L/1536d/vocab 262k (~5B with PLE) — too heavy for a
    # construct-and-count smoke on a workstation.
    config = {
        "num_layers": 12,
        "num_heads": 8,
        "num_kv_heads": 2,
        "emb_dim": 1024,
        "ffn_dim": 4096,
        "head_dim": 128,
        "global_head_dim": 256,
        "vocab_size": 32_000,
        "sliding_window": 512,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "final_logit_softcapping": 30.0,
    }

    model = Gemma4(**config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    print("layer_types:", model.layer_types)
