# pluggy

pure pytorch training stack with minimal deps.

no framework imports (megatron, deepspeed, accelerate, liger) — the mesh,
collectives, and parallelism are implemented here, with pytorch's own
implementations as the thing to test against, not the thing to depend on.

## what it can do

- **train autoregressive transformer LLMs**, single GPU up to N-way data
  parallel, from one json config — no code changes to scale a run up or down
- **data parallel training** with bucketed, overlapped gradient all-reduce
  (grad sync runs concurrently with the rest of backward, not after it),
  gradient accumulation via no_sync semantics, and gradient clipping —
  fsdp2/tp/cp/ep are the natural next axes on the same mesh, not built yet
- **bf16 mixed precision**, fp32 master weights and optimizer state — the
  standard recipe, not a naive full-bf16 cast (which silently stalls
  training once updates fall below what bf16's mantissa can resolve)
- **per-block torch.compile**, at the granularity fsdp2 will eventually want:
  one compiled region per transformer block, so warmup happens once and is
  reused across every identical block instead of once per layer
- **streaming, resumable data loading** straight from HF datasets — sequence
  packing (no padding waste), a stateful dataloader that checkpoints its own
  position, and a CUDA prefetcher that overlaps the host→device copy with
  compute
- **memory-efficient loss** — a chunked fused linear+cross-entropy that never
  materializes the (batch, seq, vocab) logits tensor, otherwise the single
  biggest activation at large vocab sizes
- **fused adamw + warmup-stable-decay and warmup-cosine schedules**
- **full checkpoint/resume** — model, optimizer, scheduler, dataloader
  position, and rng state, all restorable via `resume: null | "auto" | <step>`
- **a synthetic data pipeline** (`pluggy/synth`) — an agentic
  generate/judge/refine loop that writes jsonl shards the streaming
  dataloader consumes directly, from seed topics or grounded in uploaded
  customer documents, so the stack covers pretraining as a service end to
  end: upload data, synthesize a corpus, filter it, train on it
- **mid-training** — `model.init_from_hf` loads pretrained hub qwen3
  weights (own safetensors reader, no new deps) so a run continues
  pretraining on a customer mix instead of starting from random init
- **a from-scratch mesh + collectives layer** underneath all of the above —
  a named device mesh over the flat rank space (per-axis process groups,
  coordinates, virtual/flattened axes) and mesh-aware wrappers over every
  collective op (`all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`,
  `all_to_all`, `ring_send_recv`). this is the only place that talks to
  `torch.distributed` directly, so every parallelism strategy above it is
  backend-agnostic

everything is driven by a json config; see `configs/` for single-gpu and
data-parallel examples. optimization history, the ddp investigation, and
the roadmap live under `notes/`.

## running

single gpu and multi gpu are the same code path — world_size is just 1:

```bash
# single gpu
uv run -m pluggy.train.train --config configs/qwen3_dense_climbmix.json

# multi gpu
uv run torchrun --nproc-per-node 8 -m pluggy.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json

# benchmark mode: N steps, no checkpointing, prints tps + peak mem
uv run -m pluggy.train.train --config configs/qwen3_dense_climbmix.json --steps 20

uv run torchrun --nproc-per-node 8 -m pluggy.train.train \
    --config configs/qwen3_dense_climbmix_ddp.json --steps 20

# fsdp2 (per-param sharding: sharded grads + adam state). benchmark-only
# until sharded checkpointing lands -- the trainer refuses a checkpointing
# fsdp2 run rather than writing one rank's shard as if it were the model
uv run torchrun --nproc-per-node 8 -m pluggy.train.train \
    --config configs/qwen3_dense_climbmix_fsdp2.json --steps 20
```

## synthetic data (pretraining as a service)

`pluggy/synth` generates a pretraining corpus from scratch with an agentic
pipeline modeled on Autodata (Chen et al., arXiv:2606.25996): a planner
expands seed domains into a topic taxonomy, generation agents write
documents across a topic/style/variant grid, a judge scores each document
against a fixed rubric, borderline documents get one refinement round with
the judge's feedback, and survivors pass through minhash near-dedup into
sharded jsonl. the output streams into the same dataloader as hub datasets,
so generating a corpus and training on it is two commands:

```bash
# 1. generate the corpus (resumable: state.json + complete-shard fencing).
# default provider is grok: just export XAI_API_KEY, nothing to install
uv run -m pluggy.synth.run --config configs/synth_pretrain.json

# 2. train on it (data.data_files globs the shards; no hub involved)
uv run -m pluggy.train.train --config configs/qwen3_dense_synth.json
```

two providers sit behind the same two-method client interface
(`generate_text` / `generate_json`), selected by `provider` in the config
(or inferred from the model name):

- **grok (default)** — xai's openai-compatible api over stdlib http, zero
  extra deps; needs `XAI_API_KEY`. supports `generation.temperature` for
  sampling diversity on top of the prompt grid
- **anthropic** — needs `uv pip install -e ".[synth]"` +
  `ANTHROPIC_API_KEY`; runs claude-opus-5 with server-side refusal
  fallbacks enabled, so the occasional false-positive safety decline
  retries on a fallback model inside the same request instead of dropping
  the sample

everything is driven by the json config: provider, seed domains, docs per
topic, styles, judge thresholds, refine rounds, dedup jaccard threshold,
shard size. the orchestration is fully testable without network
(`tests/synth.py`) and further providers are a small adapter away.

### bring your own data (grounded synthesis)

alongside topic mode, the pipeline can synthesize from documents you
provide: upload files through the frontend (or drop `{"text": ...}` jsonl
under `data/uploads/<dataset>/`), and a `grounding` block in the config
chunks them and generates rephrasings, textbook-style explanations, q&a
dialogues, and summaries grounded in each chunk. the judge sees the source
chunk too, so hallucinated specifics are scored down as unfaithful rather
than passing as plausible. topic and grounded jobs run through the same
judge/refine/dedup/shard machinery and can run in one config together:

```json
"grounding": {
  "dir": "data/uploads/acme",
  "modes": ["rephrase", "textbook", "qa", "summary"],
  "gens_per_chunk": 2,
  "chunk_words": 600
}
```

## mid-training (continued pretraining)

the service story is rarely from-scratch: start from pretrained weights and
adapt them on a customer mix. `model.init_from_hf` loads a hub qwen3
checkpoint into the model at init (the safetensors reader and the hf->pluggy
name mapping are implemented in `pluggy/models/hf_import.py`, no safetensors
dep), and data mixing supplies the recipe: weight the customer's synthetic
shards against a generic replay corpus so the model learns the domain
without forgetting, then let the wsd decay phase anneal it.

```bash
# 30% customer synthetic data, 70% climbmix replay, from Qwen3-0.6B weights
uv run -m pluggy.train.train --config configs/qwen3_dense_midtrain.json
```

one sharp edge encoded in that config: hub checkpoints require the exact hub
architecture, and qwen3-0.6B's ffn dim is 3072 while pluggy's rounded
default heuristic gives 2816 -- `ffn_dim` must be set explicitly, and the
importer fails loudly (naming the fix) if it isn't.

### frontend

a minimal web ui, stdlib-only http server, no new deps:

```bash
uv run -m pluggy.synth.server    # http://127.0.0.1:8642, run from repo root
```

`/` covers every generation knob (domains, styles, judge thresholds, dedup,
sharding), saves/loads configs under `configs/`, and launches + monitors runs
(live log tail, shard/job progress).

`/train` composes a *training* config -- managed mode picks an arch/size, a
dataset mixture and a gpu count and sizes the schedule from them; expert mode
edits every field of the json directly, and loads any config already in
`configs/`. Launch saves the config and starts the same command the cli would
(`torchrun --nproc_per_node=<mesh> -m pluggy.train.train --config ...`) over
this machine's gpus, streaming loss/tps and the log back into the page;
Benchmark is that with `--steps 20`. the run gets its own process group, so
Stop reaches every rank -- and so a run outlives the server that started it.
config blocks are validated against the constructors they are unpacked into
before anything launches, so a typo'd key is a message in the ui rather than a
`TypeError` a minute into the run.

## tests

no gpus needed for any of these except the last (gloo/cpu, `mp.spawn`):

```bash
uv run tests/collective.py --world-size 4     # 12 collective op tests
uv run tests/dtensor.py --world-size 4        # placement/redistribute table
uv run tests/dataloader_packing.py --check    # packer equality + invariants
uv run tests/checkpointer.py                  # save/load roundtrip + prefetcher exact resume
uv run tests/scheduler.py                     # wsd + cosine shapes, resume parity
uv run tests/synth.py                         # synth pipeline (stubbed llm, no network)
uv run tests/synth_server.py                  # synth frontend api + uploads (localhost, no llm)
uv run tests/hf_import.py                     # safetensors + hf weight mapping (no network)
uv run tests/ui_gutters.py                    # frontend layout, measured in headless chrome
uv run tests/data_parallel.py --world-size 4  # ddp grad parity vs single process
uv run tests/fsdp2.py --world-size 4          # fsdp2 parity + memory invariants
uv run tests/grad_helper.py --world-size 2    # grad clipping vs torch reference
uv run tests/fused_linear_ce.py               # needs cuda
```

## throughput

tokens/sec for qwen3 0.6B dense, seq_len 4096, batch size 2/gpu, bf16:

| hardware | tps |
|----------|-----|
| 1x A40 | ~13.3k |
| 1x A6000 | ~14.4k |
| 1x H100 NVL | ~54.5k |

### ddp

global tps (tokens through the whole job per second), same per-gpu config:

| hardware | tps | scaling eff |
|----------|-----|-------------|
| 8x A40 | ~73.5k | 69% |
| 8x A6000 | ~86.4k | 60% |
| 4x H100 NVL | ~198k | 92% |

on a pcie-only box grad sync is the whole ballgame; `notes/DDP.md` has the
`NCCL_P2P_LEVEL` investigation that took 8x A40 from 17% to 69% scaling.

## installation

if you have uv on your machine, no need to create the conda env

```bash
pip install uv
uv venv
uv pip install -e .
```
### Optional: Weights & Biases logging

Experiment tracking is off by default. To enable it:

```bash
uv pip install -e ".[wandb]"
```

## future plans

support training AR LLMs, DLLMs, continuous diffusion LLMs, etc — focusing
on AR for now.
