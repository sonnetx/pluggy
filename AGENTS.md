# agents.md

## writing a custom model architecture

**this file is the prompt.** `pluggy/synth/write_model.py` sends this section
verbatim, with the closest reference implementation, when grok writes a new
architecture (`uv run -m pluggy.synth.write_model --name X --description
"..."`, or the /train page's Custom Model box). so a rule that lives only in
someone's head doesn't reach the generator — write it down here, and both
humans and grok get it.

**where it goes:** `pluggy/models/<name>/<name>.py` (own subdirectory per
arch, mirrors `pluggy/models/qwen3/qwen3.py` and the
`pluggy/models/llama3/llama3.py` stub). one file, self-contained — no shared
base-model module to inherit from yet.

**start from `pluggy/models/qwen3/qwen3.py`**, not from scratch. it shows the
exact shape everything else expects: rotary embedding, GQA attention, SwiGLU
block, and the top-level model class, plus a `if __name__ == "__main__":`
scaffold at the bottom that builds the model standalone and prints param
count — copy that scaffold into the new file too, it's the cheapest "does this
even construct" check.

**then go find the nearest existing implementation of whatever is novel about
your architecture, and follow it.** almost nothing is genuinely new here: the
hard parts (numerical care, compile-friendliness, fsdp2 composability) are
already solved somewhere in the tree, and a fresh implementation of a solved
problem is how subtly wrong code gets in. the map:

| if the arch needs | read | and reuse |
| --- | --- | --- |
| rope, GQA, SwiGLU, both SDPA paths | `models/qwen3/qwen3.py` | `kernels.apply_rotary`, `kernels.swiglu_mul` |
| per-layer types (sliding vs full attention, a different head_dim per type, layer patterns) | `models/gemma4/gemma4.py` | its `layer_types` / `layer_pattern` handling |
| sparse routing, top-k experts, per-expert matmul | `models/qwen3_moe/qwen3_moe.py` | `kernels.moe_expert_ffn` (`F.grouped_mm` with a loop fallback) |
| a scan / chunked recurrence that must not materialize the whole thing | `loss/fused_linear_ce.py` | its chunk loop shape: fixed-size chunks, matmuls inside, state carried across |
| norms, activations, logit capping | `kernels/` (`rms_norm`, `softcap`, `swiglu`, `rope`) | import them, don't rewrite them |
| non-uniform init, tied embeddings, residual scaling | any model's `init_weights` | copy it verbatim if the residual structure matches |

`pluggy/kernels/` is the shared surface: importing from it is one line, and
those ops are already compile-clean and fsdp2-safe. adding a kernel is fine;
duplicating one that exists is not.

**interface the rest of the codebase silently assumes** (there's no ABC
enforcing this yet — get it wrong and the failure shows up in the trainer or
objective, not in the model file):

- `self.blocks: nn.ModuleList` of transformer blocks, named exactly
  `blocks`. this is the unit `torch.compile` wraps per-block, and the unit
  future FSDP2 sharding / activation checkpointing / pipeline-parallel stage
  splits will key off. don't rename it, don't nest blocks inside another
  container.
- `self.lm_head` (an `nn.Linear`), named exactly `lm_head`. the default loss
  path (`ARObjective` with `fused_linear_ce=True`, see
  `pluggy/objectives/autoregressive.py:40`) reaches into
  `model.lm_head.weight` directly to run the fused linear+CE kernel — a
  different attribute name breaks that path silently until you turn
  `fused_linear_ce` off and diff logits.
- `forward(x, attention_mask=None, return_hidden_states=False, return_final_hidden=False)`.
  `return_final_hidden=True` must return the post-final-norm hidden state
  *without* applying `lm_head` — that's what lets the fused CE objective
  avoid materializing `(batch, seq, vocab_size)` logits, the single biggest
  activation tensor at large vocab. if tied embeddings, do it the way qwen3
  does: `self.lm_head.weight = self.token_emb.weight`.
- `self.norm` — the final norm module, named exactly `norm`. `Trainer._compile`
  calls `model.norm.compile(...)` right after the blocks, so a model that calls
  it `final_norm` builds fine and then dies the moment compile is on (the
  default).
- `init_weights(std: float = 0.02)` — gpt/llama-style init, plus scaling the
  projections that write into the residual stream (attn out-proj, mlp
  down-proj) by `1/sqrt(2 * num_layers)`. skip this and a fresh model's CE
  starts around ~900 instead of ~ln(vocab_size) — copy `Qwen3.init_weights`
  verbatim if the residual structure matches.

**register it** in `pluggy/models/builder.py`'s `MODEL_REGISTRY` dict —
`"<config-facing name>": <YourClass>`. `build_model` does
`MODEL_REGISTRY[name](**model_config)`, i.e. the `"config"` object in a
training config's `"model"` block is unpacked straight into `__init__` as
kwargs — no schema validation happens before that call, so `__init__`'s
signature *is* the config schema. (`llama3_2` is currently registered as
`None` — a placeholder for the not-yet-written arch. looking it up and
calling it will crash with a `TypeError`; don't copy that pattern once a real
implementation exists.)

**point a config at it** by copying an existing file under `configs/`
(naming convention: `<arch>_<size?>_<dataset>_<parallelism?>_<hardware?>_<variant?>.json`)
and swapping the `"model"` block:

```json
"model": {
  "name": "<your registry key>",
  "config": { /* your __init__ kwargs */ }
}
```

**other things worth knowing before writing the arch:**

- no framework imports — `torch` + stdlib only inside `pluggy/models/`
  (repo-wide rule; no megatron/deepspeed/accelerate/liger, see `README.md`).
- keep `attention_mask` handling working for both the packed/no-mask
  training path (`is_causal=True` fast path through SDPA) and the
  padded-mask eval path — qwen3's `scaled_self_attention` shows both
  branches; dropping the masked branch breaks eval/generation later.
- there's no per-arch test file (`tests/` covers collectives, dataloader,
  checkpointing, data-parallel and fused-CE — nothing model-specific), so the
  contract above is checked by `pluggy/models/contract.py` instead:

  ```bash
  uv run -m pluggy.models.contract --module pluggy.models.<name>.<name> \
      --class <YourClass> --config '{"num_layers": 6, ... small ...}'
  ```

  it builds the model on cpu and asserts the parts that otherwise fail late
  and quietly: the attribute names above, all three `forward` return modes and
  their shapes, the padded-mask path, that fresh CE is ~ln(vocab), that
  `init_weights` really scaled the residual-stream projections, and one full
  step through `ARObjective(fused_linear_ce=True)` with a gradient reaching
  every parameter. run it before registering the arch; the `__main__` scaffold
  only proves the file constructs. after that, real validation is still
  `uv run -m pluggy.train.train --config <your config> --steps 20` (benchmark
  mode, no checkpointing) and watching loss/tps.
- **a loop in `forward` is unrolled into the compiled graph**, one copy of the
  body per iteration, and `Trainer._compile` compiles every block. a loop over
  *chunks* is fine (`seq_len / chunk` iterations of whole-chunk matmuls); a
  loop over *timesteps* is not — one iteration per token turns a block that
  compiles in ~2s into ~50s at seq 160 alone, and at a real seq_len the launch
  sits at 0% gpu with the memory allocated, which looks exactly like a hang.
  if the mechanism is sequential (linear attention, a state-space scan, any
  recurrence), do the intra-chunk work as matmuls and only iterate over chunks.
- **if the layer carries state, that state has to survive the tiling.**
  re-initializing a recurrent state at each chunk boundary is a one-line
  mistake that no shape, loss or gradient check can see: the model still
  trains, it just can't remember anything older than one chunk. the chunking
  is an implementation detail and must not be visible in the output.
- trainer, parallelism, and objective code must stay model-agnostic — don't
  add `if isinstance(model, Qwen3)`-style branches anywhere outside
  `pluggy/models/`. `notes/ROADMAP.md` (part 2, M2) calls this out
  explicitly: llama3 exists to prove no code path assumes qwen3 shapes.
- changing numerics in a way that could move throughput or loss curves
  (attention scaling, rope base, init std) gets a `notes/CHANGES.md` entry —
  that file is this repo's measure-before/after discipline.

## wiring up a new dataset

see the `init-dataset` skill (`.claude/skills/init-dataset/SKILL.md`) —
covers finding the right `text_field`, subset/split, and tokenizer
compatibility for a new HF dataset before writing its `"data"` config block.

## mixing several datasets

`"data"` takes either one dataset (the keys sit at the top level, unchanged)
or a `"sources"` list, each entry with a relative `weight`:

```json
"data": {
  "sources": [
    {"name": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "weight": 50},
    {"name": "OptimalScale/ClimbMix", "weight": 30},
    {"name": "open-web-math/open-web-math", "weight": 20}
  ],
  "mixing": {"stopping_strategy": "all_exhausted"},
  "text_field": "text",
  "... shared keys": "seq_len, tokenizer, eos/pad, batch sizes, seed, num_workers"
}
```

full example: `configs/qwen3_dense_mix_ddp.json`. things worth knowing:

- **weights are token shares, not document shares**, and they're relative
  (30/50/20 means the same as 0.3/0.5/0.2). that only works because each
  source is packed *before* the mix: every packed row is exactly `seq_len`
  real tokens from one source, so drawing rows with probability p draws p of
  the tokens no matter how the sources' document lengths differ. don't move
  the interleave ahead of the packing to save a buffer — it silently turns
  the weights into document counts and lets one sequence straddle two
  corpora.
- a source **inherits** any of `name`/`data_files`/`config`/`split`/
  `text_field`/`weight`/`shuffle_buffer`/`pack_batch` it doesn't set from the
  top level of `"data"`, so shared values are written once. an unknown key in
  a source is an error, not a silent no-op (a typo'd `"weights"` would
  otherwise train on a mixture nobody chose).
- a source is named by **either** `name` (hub dataset) or `data_files` (local
  jsonl shards, e.g. `pluggy/synth` output) — exactly one, and the two mix
  freely, so a synth corpus can be blended into hub data by weight.
- `mixing.stopping_strategy` decides what happens when a source runs dry:
  `all_exhausted` (default) restarts it, so the ratio holds for the whole run
  and small sources repeat; `first_exhausted` (hf's default) ends the stream
  instead; `all_exhausted_without_replacement` drops exhausted sources and
  lets the realized mixture drift. see `STOPPING_STRATEGIES` in
  `pluggy/dataloader/builder.py`.
- each dp rank node-splits every source separately and then draws its own
  source sequence (`mixing_seed`), so the mixture is realized per rank rather
  than replicated across a step.
- `tests/data_mixing.py` (cpu, no network, in ci) is the guard: realized
  token shares vs configured weights, row purity, and resume through a
  `state_dict`. run it after touching anything in the builder's stream path.
