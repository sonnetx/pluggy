"""
self-contained qwen3-moe arch.

dense qwen3 attention/rope/norms, with the FFN swapped for a sparse MoE
block (roadmap M4 / phase 6 single-gpu path):

- top-k softmax router (default k=8, matching Qwen3-30B-A3B; configs can
  dial it down), optional norm_topk_prob
- N routed SwiGLU experts, packed as 3D weight tensors — no shared expert
  (qwen3-moe deliberately dropped the qwen2.5 shared expert)
- dropless dispatch: every selected (token, expert) pair runs; loop over
  hit experts for correctness (grouped-GEMM can replace the inner path later)
- load-balancing aux loss (Switch-Transformer style) + router z-loss,
  exposed via `aux_loss()` so the AR objective stays model-agnostic
- decoder_sparse_step / mlp_only_layers keep the HF knobs for dense layers
  mixed into the stack

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


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1_000_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
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


class Qwen3GroupQueryAttention(nn.Module):
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

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

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
        Q = self.q_norm(self.split_heads(self.q_proj(x)))
        K = self.k_norm(self.split_kv_heads(self.k_proj(x)))
        V = self.split_kv_heads(self.v_proj(x))
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)
        return self.o_proj(self.combine_heads(
            self.scaled_self_attention(Q, K, V, attention_mask)
        ))


class SwiGLU(nn.Module):
    """Dense FFN used on non-sparse layers (mlp_only / sparse_step gaps)."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        # zeros → uniform softmax at init (matches HF Qwen3MoeTopKRouter)
        self.weight = nn.Parameter(torch.zeros(num_experts, emb_dim))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: (T, H)
        router_logits = F.linear(hidden_states, self.weight)  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights.to(dtype=hidden_states.dtype)
        return router_logits, top_weights, top_indices


class Qwen3MoeExperts(nn.Module):
    """
    packed expert weights as 3D tensors (E, …). dropless: every selected
    (token, expert) pair contributes. loop over hit experts for correctness;
    a grouped-GEMM path can replace the body later without changing the API.
    """

    def __init__(self, emb_dim: int, moe_ffn_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.emb_dim = emb_dim
        self.moe_ffn_dim = moe_ffn_dim
        # (E, 2I, H) and (E, H, I) — same layout as HF Qwen3MoeExperts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_ffn_dim, emb_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, emb_dim, moe_ffn_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,   # (T, H)
        top_indices: torch.Tensor,     # (T, K)
        top_weights: torch.Tensor,     # (T, K)
    ) -> torch.Tensor:
        T, H = hidden_states.shape
        out = torch.zeros(T, H, device=hidden_states.device, dtype=hidden_states.dtype)

        # expert_mask: (E, K, T)
        expert_mask = F.one_hot(top_indices, num_classes=self.num_experts)  # (T, K, E)
        expert_mask = expert_mask.permute(2, 1, 0)  # (E, K, T)
        # which experts got at least one token
        hit = expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=False).flatten()

        for e in hit.tolist():
            # positions in the top-k axis and token ids routed to expert e
            k_pos, token_idx = torch.where(expert_mask[e])
            if token_idx.numel() == 0:
                continue
            tok = hidden_states[token_idx]  # (N, H)
            gate, up = F.linear(tok, self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e])
            y = y * top_weights[token_idx, k_pos, None]
            out.index_add_(0, token_idx, y.to(dtype=out.dtype))

        return out


class Qwen3MoeSparseMLP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        moe_ffn_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.router = Qwen3MoeTopKRouter(
            emb_dim, num_experts, top_k, norm_topk_prob
        )
        self.experts = Qwen3MoeExperts(emb_dim, moe_ffn_dim, num_experts)
        # filled during forward; the model harvests these for aux loss
        self.last_router_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        flat = x.reshape(B * S, H)
        logits, weights, indices = self.router(flat)
        self.last_router_logits = logits  # (T, E), kept for aux loss
        y = self.experts(flat, indices, weights)
        return y.reshape(B, S, H)


class Qwen3MoeTransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_moe: bool,
        ffn_dim: int | None = None,
        moe_ffn_dim: int | None = None,
        num_experts: int = 128,
        top_k: int = 8,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.gqa = Qwen3GroupQueryAttention(num_heads, emb_dim, num_kv_heads, head_dim)
        self.norm1 = nn.RMSNorm(emb_dim)
        self.norm2 = nn.RMSNorm(emb_dim)
        self.use_moe = use_moe

        if use_moe:
            assert moe_ffn_dim is not None, "moe_ffn_dim required for sparse layers"
            self.ffn = Qwen3MoeSparseMLP(
                emb_dim, moe_ffn_dim, num_experts, top_k, norm_topk_prob
            )
        else:
            if ffn_dim is None:
                ffn_dim = 256 * ((int(8 * emb_dim / 3) + 255) // 256)
            self.ffn = SwiGLU(emb_dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.gqa(self.norm1(x), cos, sin, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


def load_balancing_loss(
    router_logits: list[torch.Tensor],
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """
    Switch-Transformer load-balancing loss over a list of per-layer router
    logits, each (T, E). returns a scalar; multiply by coef outside.
    """
    if not router_logits:
        return torch.tensor(0.0)

    # (L*T, E)
    logits = torch.cat([g.float() for g in router_logits], dim=0)
    probs = F.softmax(logits, dim=-1)                          # (N, E)
    _, top_idx = torch.topk(probs, top_k, dim=-1)              # (N, K)
    mask = F.one_hot(top_idx, num_experts).float()             # (N, K, E)
    # fraction of tokens (across top-k slots) assigned to each expert
    tokens_per_expert = mask.mean(dim=0)                       # (K, E)
    # mean router probability mass per expert
    router_prob = probs.mean(dim=0)                            # (E,)
    # N * sum_i f_i * P_i  (sum over top-k slots then experts)
    loss = (tokens_per_expert * router_prob.unsqueeze(0)).sum() * num_experts
    return loss


def router_z_loss(router_logits: list[torch.Tensor]) -> torch.Tensor:
    """ST-MoE z-loss: mean over tokens of (logsumexp(logits))^2."""
    if not router_logits:
        return torch.tensor(0.0)
    logits = torch.cat([g.float() for g in router_logits], dim=0)
    return torch.square(torch.logsumexp(logits, dim=-1)).mean()


class Qwen3Moe(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        emb_dim: int,
        head_dim: int,
        vocab_size: int,
        moe_ffn_dim: int,
        num_experts: int = 128,
        top_k: int = 8,
        ffn_dim: int | None = None,
        norm_topk_prob: bool = True,
        decoder_sparse_step: int = 1,
        mlp_only_layers: list[int] | None = None,
        router_aux_loss_coef: float = 0.001,
        router_z_loss_coef: float = 0.001,
        rope_base: float = 1_000_000.0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        mlp_only = set(mlp_only_layers or [])
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.rope = Qwen3RotaryEmbedding(head_dim, base=rope_base)

        blocks: list[Qwen3MoeTransformerBlock] = []
        for i in range(num_layers):
            use_moe = (
                i not in mlp_only
                and num_experts > 0
                and (i + 1) % decoder_sparse_step == 0
            )
            blocks.append(
                Qwen3MoeTransformerBlock(
                    num_heads,
                    emb_dim,
                    num_kv_heads,
                    head_dim,
                    use_moe=use_moe,
                    ffn_dim=ffn_dim,
                    moe_ffn_dim=moe_ffn_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    norm_topk_prob=norm_topk_prob,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # filled each forward; consumed by aux_loss()
        self._router_logits: list[torch.Tensor] = []
        # last-forward per-expert token counts (summed over layers), for logging
        self.last_tokens_per_expert: torch.Tensor | None = None

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

        # packed expert weights aren't nn.Linear — init explicitly. routers
        # stay at zero (uniform) so we skip them.
        for block in self.blocks:
            if isinstance(block.ffn, Qwen3MoeSparseMLP):
                nn.init.normal_(block.ffn.experts.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(block.ffn.experts.down_proj, mean=0.0, std=std)
                # leave block.ffn.router.weight at zeros

        residual_scale = (2 * len(self.blocks)) ** -0.5
        for block in self.blocks:
            block.gqa.o_proj.weight.data.mul_(residual_scale)
            if isinstance(block.ffn, Qwen3MoeSparseMLP):
                block.ffn.experts.down_proj.data.mul_(residual_scale)
            else:
                block.ffn.down_proj.weight.data.mul_(residual_scale)

    def _collect_router_logits(self) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        for block in self.blocks:
            ffn = block.ffn
            if isinstance(ffn, Qwen3MoeSparseMLP) and ffn.last_router_logits is not None:
                logits.append(ffn.last_router_logits)
        return logits

    def _update_expert_counts(self, router_logits: list[torch.Tensor]) -> None:
        if not router_logits:
            self.last_tokens_per_expert = None
            return
        counts = torch.zeros(self.num_experts, device=router_logits[0].device)
        for g in router_logits:
            probs = F.softmax(g.float(), dim=-1)
            _, idx = torch.topk(probs, self.top_k, dim=-1)
            counts += F.one_hot(idx, self.num_experts).float().sum(dim=(0, 1))
        self.last_tokens_per_expert = counts.detach()

    def aux_loss(self) -> torch.Tensor | None:
        """
        load-balance + z-loss over the most recent forward's router logits.
        returns None when no sparse layer ran (so ARObjective can no-op).
        """
        logits = self._router_logits
        if not logits:
            return None
        device = logits[0].device
        loss = torch.zeros((), device=device)
        if self.router_aux_loss_coef != 0.0:
            loss = loss + self.router_aux_loss_coef * load_balancing_loss(
                logits, self.num_experts, self.top_k
            )
        if self.router_z_loss_coef != 0.0:
            loss = loss + self.router_z_loss_coef * router_z_loss(logits)
        return loss

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_final_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape

        # drop stale router state so a failed/partial step can't leak into aux
        for block in self.blocks:
            if isinstance(block.ffn, Qwen3MoeSparseMLP):
                block.ffn.last_router_logits = None
        self._router_logits = []

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

        # harvest router logits after the full stack so aux_loss sees every layer
        self._router_logits = self._collect_router_logits()
        self._update_expert_counts(self._router_logits)

        if return_final_hidden:
            return self.norm(x)

        logits = self.lm_head(self.norm(x))
        if return_hidden_states:
            return logits, hidden_states
        return logits


if __name__ == "__main__":
    # tiny shape for construct-and-count; real 30B-A3B is
    # 48L / 2048d / 128 experts / top-8 / moe_ffn=768.
    config = {
        "num_layers": 8,
        "num_heads": 8,
        "num_kv_heads": 2,
        "emb_dim": 512,
        "head_dim": 64,
        "vocab_size": 8_000,
        "moe_ffn_dim": 256,
        "num_experts": 8,
        "top_k": 2,
        "ffn_dim": 1024,
    }
    model = Qwen3Moe(**config)
    print(model)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params:", n)
    print("sparse layers:", sum(1 for b in model.blocks if b.use_moe))
