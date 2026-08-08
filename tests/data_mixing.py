"""
correctness suite for data mixing (pluggy/dataloader/builder.py).

the claim under test is that a source's weight is its share of TOKENS, not of
documents: that only holds because mixing happens after packing, where every
row is exactly seq_len tokens from one source. so the sources here are given
deliberately mismatched document lengths (~16 tokens vs ~400) -- a doc-level
interleave would put the short source far under its weight, a row-level one
lands on it regardless.

what's checked:
  config     resolve_sources normalizes both config shapes (single dataset,
             mixture), inherits top-level defaults, and rejects the mistakes
             that would silently train on the wrong mixture
  purity     every row is seq_len long and holds tokens from exactly one
             source -- no packed sequence straddles two corpora
  ratio      realized token shares match the configured weights
  resume     a StatefulDataLoader restored mid-stream replays the same rows,
             i.e. the interleave rng really does live in the checkpointed state
             (without that, resume would restart the mixture, not continue it)
  dp         each dp rank draws its own source sequence and still realizes the
             target mixture on its own

the real ListPacker runs, against a fake tokenizer (pre-tokenized docs), so the
path under test is the same .map -> interleave the trainer builds. cpu-only, no
network, no gpu:

    uv run tests/data_mixing.py
"""

from __future__ import annotations

import random

from collections import Counter

import datasets
import torch

from torchdata.stateful_dataloader import StatefulDataLoader

from pluggy.dataloader.builder import (
    Collator,
    mix_streams,
    mixing_seed,
    prepare_source_stream,
    resolve_sources,
)
from pluggy.dataloader.packers import ListPacker


SEQ_LEN = 64
EOS_ID, PAD_ID = 0, 1
SEED = 7
TOL = 0.02  # ~3 sigma on the smallest source's share at the row counts below

# (marker token, doc length range, tokens per shard). the marker is what every
# token of a source's docs is, so a packed row's source is readable straight off
# its contents. lengths differ by ~25x across sources on purpose: that is the
# gap between "30% of documents" and "30% of tokens".
SOURCES = {
    "short": (10, (8, 24), 200_000),
    "long": (11, (200, 600), 400_000),
    "medium": (12, (30, 300), 200_000),
}
WEIGHTS = {"short": 30, "long": 50, "medium": 20}
N_SHARDS = 4


def gen_docs(shards, marker: int, min_len: int, max_len: int, tokens_per_shard: int):
    """
    synthetic pre-tokenized docs, one generator per source. `shards` is a list
    so datasets can hand each shard to a different dataloader worker.

    top-level (not a closure) and driven by plain ints so the whole dataset
    pickles to workers under forkserver -- the same constraint packers.py and
    Collator are written to.
    """
    for shard in shards:
        rng = random.Random(1000 * shard + marker)
        emitted = 0
        while emitted < tokens_per_shard:
            n = rng.randint(min_len, max_len)
            emitted += n
            # "text" here is already a token list; FakeTokenizer passes it
            # through. keeps the test off the network without pretending to
            # tokenize.
            yield {"body": [marker] * n, "shard": shard}


class FakeEncoding:
    """the one attribute ListPacker reads off a tokenizers Encoding."""

    def __init__(self, ids: list[int]):
        self.ids = ids


class FakeTokenizer:
    """pre-tokenized docs in, Encoding-shaped objects out. picklable, unlike a lambda."""

    def encode_batch(self, texts: list[list[int]]) -> list[FakeEncoding]:
        return [FakeEncoding(list(ids)) for ids in texts]


def source_cfgs(dp_size: int) -> list[dict]:
    """the sources as the builder would resolve them from a real config block."""
    return resolve_sources({
        "sources": [{"name": name, "weight": WEIGHTS[name]} for name in SOURCES],
        "text_field": "body",
        # small buffers: the point is to exercise shuffle + node split in the
        # pipeline, not to decorrelate a real corpus
        "shuffle_buffer": 50,
        "pack_batch": 100 * dp_size,
    })


def raw_source(name: str, n_shards: int = N_SHARDS) -> datasets.IterableDataset:
    marker, (min_len, max_len), tokens_per_shard = SOURCES[name]
    return datasets.IterableDataset.from_generator(
        gen_docs,
        gen_kwargs={
            "shards": list(range(n_shards)),
            "marker": marker,
            "min_len": min_len,
            "max_len": max_len,
            "tokens_per_shard": tokens_per_shard,
        },
    )


def build_mixed_stream(dp_rank: int = 0, dp_size: int = 1) -> datasets.IterableDataset:
    """the builder's pipeline end to end, minus load_source."""
    packer = ListPacker(FakeTokenizer(), SEQ_LEN, EOS_ID, PAD_ID)
    sources = source_cfgs(dp_size)
    streams = [
        prepare_source_stream(raw_source(src["name"]), src, packer, SEED + i, dp_rank, dp_size)
        for i, src in enumerate(sources)
    ]
    return mix_streams(
        streams, [src["weight"] for src in sources], seed=mixing_seed(SEED, dp_rank)
    )


def loader(stream: datasets.IterableDataset, num_workers: int = 0) -> StatefulDataLoader:
    return StatefulDataLoader(
        stream,
        batch_size=8,
        num_workers=num_workers,
        collate_fn=Collator(FakeTokenizer(), -100),
    )


def row_sources(batch: torch.Tensor) -> list[int]:
    """
    the marker of each row, asserting the row is pure while it's at it. eos is
    the packer's doc separator, so it's the one non-marker id a row may hold.
    """
    markers = []
    for row in batch.tolist():
        found = set(row) - {EOS_ID}
        assert len(found) == 1, f"row mixes sources {sorted(found)}: packing leaked across a corpus"
        markers.append(found.pop())
    return markers


def collect(dl: StatefulDataLoader, n_batches: int) -> list[list[int]]:
    out = []
    # strict=False: the loader is the longer side (or runs dry, which the
    # length assert below catches with a message that says so)
    for _, batch in zip(range(n_batches), dl, strict=False):
        assert batch["input_ids"].shape[1] == SEQ_LEN, (
            f"row is {batch['input_ids'].shape[1]} tokens, not seq_len {SEQ_LEN}: "
            f"weights stop being token shares"
        )
        assert torch.equal(batch["input_ids"], batch["labels"])
        out.append(row_sources(batch["input_ids"]))
    assert len(out) == n_batches, f"stream ran dry after {len(out)}/{n_batches} batches"
    return out


def shares(rows: list[list[int]]) -> dict[str, float]:
    """realized token share per source. every row is seq_len tokens, so rows == tokens."""
    counts = Counter(m for batch in rows for m in batch)
    total = sum(counts.values())
    return {name: counts[SOURCES[name][0]] / total for name in SOURCES}


def check_resolve_sources() -> None:
    common = {
        "split": "train",
        "text_field": "text",
        "shuffle_buffer": 1000,
        "seq_len": 4096,
        "num_workers": 4,
        "seed": 0,
    }

    # single dataset: the old config shape still resolves to exactly one source
    # owning all the tokens, with no "sources" key in sight
    single = resolve_sources({**common, "name": "OptimalScale/ClimbMix", "config": None})
    assert len(single) == 1
    assert single[0]["name"] == "OptimalScale/ClimbMix"
    assert single[0]["weight"] == 1.0
    assert single[0]["pack_batch"] == 1000  # default, not set in the config

    # local jsonl shards (pluggy/synth output) name a source the same way a hub
    # dataset does, alone or inside a mixture
    local = resolve_sources({**common, "data_files": "data/synth_v0/shard-*.jsonl"})
    assert local[0]["name"] is None
    assert local[0]["data_files"] == "data/synth_v0/shard-*.jsonl"
    assert local[0]["label"] == "data/synth_v0/shard-*.jsonl"
    assert resolve_sources({**common, "sources": [
        {"name": "a", "weight": 1}, {"data_files": "shard-*.jsonl", "weight": 1},
    ]})[1]["data_files"] == "shard-*.jsonl"

    # mixture: relative weights normalize, and unset per-source keys fall back
    # to the top level (text_field/split here) instead of erroring
    mixed = resolve_sources({
        **common,
        "sources": [
            {"name": "a", "weight": 30},
            {"name": "b", "weight": 50, "config": "python", "text_field": "content"},
            {"name": "c", "weight": 20, "split": "validation", "pack_batch": 10},
        ],
    })
    assert [s["weight"] for s in mixed] == [0.3, 0.5, 0.2]
    assert [s["text_field"] for s in mixed] == ["text", "content", "text"]
    assert [s["split"] for s in mixed] == ["train", "train", "validation"]
    assert [s["config"] for s in mixed] == [None, "python", None]
    assert [s["pack_batch"] for s in mixed] == [1000, 1000, 10]
    # 3/5/2 and 0.3/0.5/0.2 are the same mixture
    assert [s["weight"] for s in resolve_sources({**common, "sources": [
        {"name": "a", "weight": 3}, {"name": "b", "weight": 5}, {"name": "c", "weight": 2},
    ]})] == [0.3, 0.5, 0.2]

    # the mistakes worth failing on, all of which would otherwise train on a
    # mixture nobody asked for
    for bad, why in (
        ({**common, "sources": [{"name": "a", "weight": 1}, {"name": "b"}]}, "missing weight"),
        ({**common, "sources": [{"name": "a", "weight": 1}, {"name": "b", "weight": 0}]}, "zero weight"),
        ({**common, "sources": [{"name": "a", "weight": -1}]}, "negative weight"),
        ({**common, "sources": [{"weight": 1}]}, "neither name nor data_files"),
        ({**common, "sources": [{"name": "a", "data_files": "x.jsonl", "weight": 1}]}, "both"),
        ({**common, "sources": []}, "empty sources"),
        # a typo'd key would otherwise be silently ignored -- e.g. "weights"
        # leaves the source unweighted, "field" leaves text_field inherited
        ({**common, "sources": [{"name": "a", "weights": 1}]}, "typo'd key"),
        ({k: v for k, v in common.items() if k != "text_field"} | {"name": "a"}, "no text_field"),
    ):
        try:
            resolve_sources(bad)
        except AssertionError:
            continue
        raise AssertionError(f"resolve_sources accepted a config it shouldn't have: {why}")

    print("  resolve_sources: config shapes, inheritance, and rejections ok")


def check_single_source_unwrapped() -> None:
    """
    one source must come back as the packed stream itself. an interleave
    wrapper around it would be a no-op that still caps num_shards and burns an
    rng, so the single-dataset path has to stay untouched by all of this.
    """
    packer = ListPacker(FakeTokenizer(), SEQ_LEN, EOS_ID, PAD_ID)
    src = source_cfgs(1)[0]
    raw = raw_source(list(SOURCES)[0], n_shards=1)
    stream = prepare_source_stream(raw, src, packer, SEED, 0, 1)
    assert mix_streams([stream], [1.0], seed=0) is stream
    print("  single source: returned unwrapped, no interleave in the path")


def check_ratio_and_purity(num_workers: int) -> None:
    rows = collect(loader(build_mixed_stream(), num_workers=num_workers), n_batches=750)
    realized = shares(rows)
    target = {name: w / sum(WEIGHTS.values()) for name, w in WEIGHTS.items()}
    detail = ", ".join(f"{n} {realized[n]:.3f} (want {target[n]:.2f})" for n in SOURCES)
    for name in SOURCES:
        assert abs(realized[name] - target[name]) < TOL, f"mixture off target: {detail}"
    print(f"  num_workers={num_workers}: {detail}")


def check_resume(num_workers: int) -> None:
    dl = loader(build_mixed_stream(), num_workers=num_workers)
    it = iter(dl)
    for _ in range(20):
        next(it)
    state = dl.state_dict()
    tail = [row_sources(next(it)["input_ids"]) for _ in range(20)]

    resumed = loader(build_mixed_stream(), num_workers=num_workers)
    resumed.load_state_dict(state)
    it2 = iter(resumed)
    tail2 = [row_sources(next(it2)["input_ids"]) for _ in range(20)]
    assert tail == tail2, (
        f"resume diverged at num_workers={num_workers}: the interleave rng isn't in the state dict, "
        f"so the mixture restarts instead of continuing"
    )
    print(f"  num_workers={num_workers}: resumed stream replays the same 20 batches")


def check_dp_ranks() -> None:
    per_rank = [collect(loader(build_mixed_stream(rank, 2)), n_batches=300) for rank in (0, 1)]
    for rank, rows in enumerate(per_rank):
        realized = shares(rows)
        for name in SOURCES:
            target = WEIGHTS[name] / sum(WEIGHTS.values())
            assert abs(realized[name] - target) < 2 * TOL, (
                f"dp rank {rank} mixture off target: {realized}"
            )
    # a shared mixing seed would hand every rank the same source per row index,
    # collapsing a step's mixture to one rank's worth of draws
    assert per_rank[0] != per_rank[1], "dp ranks drew identical source sequences"
    print("  dp=2: both ranks hit the target mixture, with independent draws")


def main() -> None:
    check_resolve_sources()
    check_single_source_unwrapped()
    for num_workers in (0, 2):
        check_ratio_and_purity(num_workers)
        check_resume(num_workers)
    check_dp_ranks()
    print("all data mixing checks passed")


if __name__ == "__main__":
    main()
