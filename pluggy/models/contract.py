"""
does a model actually satisfy what the trainer assumes about it?

    uv run -m pluggy.models.contract --module pluggy.models.qwen3.qwen3 \
        --class Qwen3 --config '{"num_layers": 2, ...}'

the interface in AGENTS.md is enforced by nothing: the trainer, the objective
and the parallelism layer reach into a model by attribute name, so a model
that gets it wrong builds fine and then trains wrong. `__main__` scaffolds
only prove the thing constructs. this checks the parts that actually bite:

    1. the attributes the trainer names -- blocks, lm_head, norm, init_weights
    2. forward's three return modes, and their shapes
    3. the padded-mask path (eval/generation) still runs
    4. init sanity: fresh CE ~ ln(vocab). the residual-stream scaling in
       init_weights is invisible at construction and shows up here as CE in
       the hundreds instead of ~12
    5. one real step through ARObjective with fused_linear_ce=True -- the
       actual training path, including backward and a grad on every parameter

it takes a module path rather than a registry key on purpose: this has to run
BEFORE a new architecture is added to MODEL_REGISTRY, since builder.py is on
every training run's import path.

used by pluggy/synth/write_model.py after grok writes an architecture, and
worth running by hand after writing one yourself.
"""

import argparse
import importlib
import json
import math
import sys
import time

import torch
import torch.nn as nn

from pluggy.objectives.autoregressive import ARObjective

# a fresh model's CE is ln(vocab) if the logits start near-uniform. missing
# the 1/sqrt(2*layers) residual scaling puts it in the hundreds, so anything
# short of a wide tolerance here is measuring noise, not correctness
CE_TOLERANCE = 1.5

# seconds to compile ONE block at the tiny size used here. the trainer compiles
# every block by default, so this is the per-block tax paid at every launch,
# multiplied by the layer count and by however much longer the real seq_len is.
# a healthy block is ~2s; a forward with a python loop per timestep unrolls into
# the graph and takes ~30x that even at seq 160.
COMPILE_BUDGET_S = 30


class ContractError(AssertionError):
    pass


def check(model: nn.Module, vocab_size: int, batch: int = 2, seq: int = 160,
          device: str = "cpu", compile_check: bool = True) -> list[str]:
    """returns the list of checks that passed; raises ContractError on the first failure."""
    passed = []

    def require(cond, what, detail=""):
        if not cond:
            raise ContractError(f"{what}{': ' + detail if detail else ''}")
        passed.append(what)

    require(isinstance(getattr(model, "blocks", None), nn.ModuleList),
            "model.blocks is an nn.ModuleList",
            "torch.compile wraps it per-block and fsdp2 shards on it")
    require(isinstance(getattr(model, "lm_head", None), nn.Linear),
            "model.lm_head is an nn.Linear",
            "the fused CE path reads lm_head.weight directly")
    require(isinstance(getattr(model, "norm", None), nn.Module),
            "model.norm exists",
            "trainer._compile calls model.norm.compile()")
    require(callable(getattr(model, "init_weights", None)),
            "model.init_weights() exists")

    model = model.to(device)
    std = 0.02
    model.init_weights(std)
    emb_dim = model.lm_head.in_features

    # the residual-stream scaling is invisible in the initial loss -- the final
    # norm renormalizes whatever variance compounded through the blocks, so CE
    # sits at ln(vocab) with or without it, and only the training curve knows.
    # so check it structurally: after init, the projections that write into the
    # residual stream should carry std/sqrt(2 * num_layers), which is far below
    # the std everything else got.
    scaled = std * (2 * len(model.blocks)) ** -0.5
    stds = [p.detach().std().item() for p in model.parameters() if p.dim() >= 2]
    require(min(stds) < scaled * 2,
            "init_weights scales the residual-stream projections",
            f"the narrowest 2D parameter has std {min(stds):.5f}, but attn o_proj "
            f"and mlp down_proj should be near {scaled:.5f} "
            f"(std / sqrt(2 * {len(model.blocks)} layers)). without it the residual "
            f"variance compounds with depth and training destabilizes")
    ids = torch.randint(0, vocab_size, (batch, seq), device=device)

    logits = model(ids)
    require(tuple(logits.shape) == (batch, seq, vocab_size),
            "forward() returns (batch, seq, vocab) logits",
            f"got {tuple(logits.shape)}")

    hidden = model(ids, return_final_hidden=True)
    require(tuple(hidden.shape) == (batch, seq, emb_dim),
            "forward(return_final_hidden=True) returns post-norm hidden, not logits",
            f"got {tuple(hidden.shape)}, expected {(batch, seq, emb_dim)} -- returning "
            f"logits here defeats the whole point of the fused CE path")

    out = model(ids, return_hidden_states=True)
    require(isinstance(out, tuple) and len(out) == 2,
            "forward(return_hidden_states=True) returns (logits, hidden_states)")

    # padded eval path: an explicit mask must not be ignored or crash
    mask = torch.ones(batch, seq, dtype=torch.bool, device=device)
    mask[:, -3:] = False
    masked = model(ids, attention_mask=mask)
    require(tuple(masked.shape) == (batch, seq, vocab_size),
            "forward(attention_mask=...) runs the padded path")

    # the two properties every autoregressive sequence model must have, and the
    # two that shape checks cannot see. they are mechanism-agnostic: softmax
    # attention, linear attention and a recurrent state all have to satisfy
    # them, and a plausible-looking implementation can miss either.
    with torch.no_grad():
        base = model(ids)
        # (a) causal: a change late in the sequence cannot move earlier outputs
        late = ids.clone()
        late[:, -1] = (late[:, -1] + 1) % vocab_size
        require(torch.allclose(model(late)[:, :-1], base[:, :-1], atol=1e-4),
                "causal: a later token does not change earlier outputs",
                "information is leaking backwards -- a mask is missing, or the "
                "recurrence is reading ahead")
        # (b) the whole context is actually reachable. a state that resets on a
        # chunk boundary, or a window shorter than the sequence, shows up here
        # and nowhere else: shapes, loss and gradients all look perfect
        first = ids.clone()
        first[:, 0] = (first[:, 0] + 1) % vocab_size
        moved = (model(first)[:, -1] - base[:, -1]).abs().max().item()
        require(moved > 1e-5,
                f"the last position depends on the first token ({seq} apart)",
                f"changing token 0 moved the final logits by {moved:.2e}. the "
                f"model cannot see its own context -- a recurrent state being "
                f"re-initialized per chunk does exactly this. (a deliberately "
                f"sliding-window arch with a window under {seq} would also trip "
                f"this; widen the window or raise --seq to test it)")

    # what the loss actually starts at. the objective is the real one, on the
    # real path (fused linear + CE), so this covers lm_head.weight too
    objective = ARObjective(ignore_index=-100, fused_linear_ce=True, ce_chunk_size=1024)
    loss = objective.compute_loss(model, {"input_ids": ids, "labels": ids.clone()})
    require(torch.isfinite(loss), "loss is finite", f"got {loss.item()}")
    expected = math.log(vocab_size)
    require(abs(loss.item() - expected) < CE_TOLERANCE,
            f"fresh CE is ~ln(vocab) ({expected:.2f})",
            f"got {loss.item():.2f}. init_weights is probably missing the "
            f"1/sqrt(2 * num_layers) scaling on the projections that write into "
            f"the residual stream (attn o_proj, mlp down_proj)")

    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    # tied lm_head/token_emb share one tensor, so one name of the pair is fine
    missing = [n for n in missing
               if not (n == "lm_head.weight" and model.lm_head.weight is model.token_emb.weight)]
    require(not missing, "every parameter gets a gradient",
            f"no grad reached {missing[:5]} -- an unused branch, or a tensor "
            f"detached from the graph")

    if compile_check:
        # Trainer._compile wraps every block, so whatever this costs is paid
        # per block at every launch. a python loop over timesteps in the
        # forward is unrolled into the graph -- one copy of the body per token
        # -- and inductor then sits there for minutes with the gpus allocated
        # and idle, which reads exactly like a hang.
        start = time.monotonic()
        model.blocks[0].compile()
        with torch.no_grad():
            model(ids)
        elapsed = time.monotonic() - start
        require(elapsed < COMPILE_BUDGET_S,
                f"one block compiles in under {COMPILE_BUDGET_S}s (took {elapsed:.0f}s)",
                f"at seq {seq} on cpu, and the trainer compiles every block at "
                f"the real seq_len. a loop over timesteps in the forward is the "
                f"usual cause: keep any loop over CHUNKS (O(seq/chunk) iterations "
                f"of matmuls), never one iteration per token")
    return passed


def build(module_path: str, class_name: str, config: dict) -> nn.Module:
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ContractError(f"{module_path} has no class {class_name!r}")
    return cls(**config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True,
                        help="e.g. pluggy.models.qwen3.qwen3")
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--config", required=True,
                        help="json: the __init__ kwargs, at a SMALL size (this runs "
                             "a real forward and backward)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-compile-check", dest="compile_check",
                        action="store_false",
                        help="skip the per-block torch.compile budget (the slowest check)")
    args = parser.parse_args()

    config = json.loads(args.config)
    vocab = config.get("vocab_size")
    if not vocab:
        print("contract: config needs vocab_size")
        return 2
    try:
        model = build(args.module, args.class_name, config)
        passed = check(model, vocab, device=args.device,
                       compile_check=args.compile_check)
    except ContractError as e:
        print(f"CONTRACT FAILED: {e}")
        return 1
    for name in passed:
        print(f"  ok  {name}")
    print("contract: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
