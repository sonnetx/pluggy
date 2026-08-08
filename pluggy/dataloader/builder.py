"""
build helper for the dataloader, colocated with DataLoader
note that testing shows numpy is faster than torch or
python arrs, so we should move packing into the collator

but this doesn't seem to be bottleneck for throughput
so leave for now

eos vs pad: eos (doc separator, trained on) and pad (bin filler,
maskable) are looked up separately from config and must be distinct
vocab ids -- see the contract in packers.py

data mixing: `data.sources` is a list of datasets with relative weights
("30% of tokens from this one"). each source is loaded, shuffled, node-split
and PACKED independently, then the packed streams are interleaved with those
weights.

mixing AFTER packing is what makes the weights token ratios rather than
document ratios: every row a packer emits is exactly seq_len real tokens, so
drawing rows from source i with probability p_i puts p_i of the tokens in the
batch stream, whatever the sources' document lengths are. interleaving raw
documents instead would weight by document COUNT -- a 30% weight on a source of
short docs would land far under 30% of tokens -- and would also let two sources
share one packed sequence, so a single attention window straddles corpora.
the cost is one shuffle buffer and one map buffer per source instead of one
total, and one dropped map-batch tail per source (see packers.py).

the invariant "one row == seq_len real tokens" is what ties weights to tokens,
so it holds for the stream packers only. ListPackerBestFit pads its bins, so
under it the weights would be exact in rows but slightly off in real tokens.
"""

import datasets
import torch

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

# packing strategies live in packers.py so they can be swapped/benchmarked
# independently; see tests/dataloader_packing.py. ListPacker is the original
# python-list implementation.
from pluggy.dataloader.packers import ListPacker


# what a source dict may set. everything else under "data" (seq_len, tokenizer,
# batch sizes, num_workers, seed, eos/pad) is shared by every source, and these
# keys may also be set once at the top level as a default the sources inherit.
# a source is named either by hub "name" or by local "data_files".
SOURCE_KEYS = (
    "name", "data_files", "config", "split", "text_field", "weight", "shuffle_buffer", "pack_batch",
)

# what happens when a source runs dry mid-run, i.e. how a mixture behaves once
# it outlives its smallest corpus:
#   all_exhausted   restart exhausted sources so the mixture is held for the
#                   whole run (a small source repeats -- usually the intent of
#                   upweighting one), stopping once every source has been seen
#                   through at least once.
#   first_exhausted stop at the first source to run dry (hf's default). the
#                   mixture stays honest but the loader dies mid-training.
#   all_exhausted_without_replacement
#                   drop exhausted sources instead of restarting: no repeats,
#                   but the realized mixture drifts as sources fall out.
STOPPING_STRATEGIES = ("first_exhausted", "all_exhausted", "all_exhausted_without_replacement")


class Collator:
    """
    stack pre-packed rows into a batch. the Packer (.map step) already emits
    fixed-length seq_len blocks with no padding, so there's nothing to pad,
    truncate, or mask here -- the collate just stacks ids and clones labels.

    every position is a real token, so labels are a straight clone (no
    ignore_index in the pretrain path). label *shifting* stays the objective's
    job. ignore_index is kept on the constructor for SFT prompt-masking later.

    a top-level class (not a closure) so DataLoader workers can pickle it,
    which forkserver/spawn start methods require (py3.14 default on linux).
    """
    def __init__(self, tokenizer: Tokenizer, ignore_index: int):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def normalize_weights(weights: list[float]) -> list[float]:
    """
    relative weights -> fractions summing to 1, so a config can say 30/50/20 or
    0.3/0.5/0.2 and mean the same thing. dividing by the sum (rather than
    demanding a pre-normalized config) also keeps the fractions as close to
    exact as float allows, which interleave_datasets requires of probabilities.
    """
    total = sum(weights)
    assert total > 0, f"data source weights must sum to something positive, got {weights}"
    return [w / total for w in weights]


def resolve_sources(data_cfg: dict) -> list[dict]:
    """
    normalize both config shapes into one list of fully-resolved sources, with
    weights already turned into token fractions.

    single dataset (unchanged, and still the common case) -- one source, weight 1:

        "data": {"name": "OptimalScale/ClimbMix", "text_field": "text", ...}

    mixture -- weights are relative and become TOKEN fractions:

        "data": {"sources": [
            {"name": "...", "weight": 30},
            {"name": "...", "weight": 50, "config": "python", "text_field": "content"},
            {"name": "...", "weight": 20}
        ], "text_field": "text", ...}

    a source inherits any of SOURCE_KEYS it doesn't set from the top level, so
    whatever is the same across sources (usually split and text_field) is
    written once.
    """
    sources = data_cfg.get("sources")
    if sources is None:
        # single-dataset config: the top-level keys ARE the one source.
        sources = [{k: data_cfg[k] for k in SOURCE_KEYS if k in data_cfg}]
    assert sources, "data.sources is empty: give it at least one dataset"

    resolved = []
    for i, src in enumerate(sources):
        unknown = set(src) - set(SOURCE_KEYS)
        assert not unknown, (
            f"data.sources[{i}] has unknown key(s) {sorted(unknown)}; a source may set "
            f"{list(SOURCE_KEYS)}, and everything else under 'data' is shared by all sources"
        )
        # local jsonl shards (e.g. pluggy/synth output) are named by
        # "data_files" instead of a hub "name" -- exactly one of the two
        assert ("name" in src) != ("data_files" in src), (
            f"data.sources[{i}] must have exactly one of 'name' (a hub dataset) or "
            f"'data_files' (local jsonl shards), got {sorted(set(src) & {'name', 'data_files'})}"
        )
        label = src.get("name") or src["data_files"]

        def inherited(key, default=None, _src=src):
            # the top level doubles as the default for every source
            return _src.get(key, data_cfg.get(key, default))

        text_field = inherited("text_field")
        assert text_field is not None, f"data.sources[{i}] ({label}) has no 'text_field'"
        # a mixture with implicit weights is almost always a mistake (the ratio
        # is the point of listing sources), so make it explicit; a lone source
        # needs no weight because it gets all the tokens either way.
        weight = src.get("weight", 1.0 if len(sources) == 1 else None)
        assert weight is not None, (
            f"data.sources[{i}] ({label}) has no 'weight' -- every source in a "
            f"mixture needs one (relative, e.g. 30/50/20)"
        )
        assert weight > 0, f"data.sources[{i}] ({label}) has non-positive weight {weight}"

        resolved.append({
            "name": src.get("name"),
            # glob or list of local jsonl shards, when this isn't a hub dataset
            "data_files": src.get("data_files"),
            "label": label,  # whichever of the two named it, for messages
            # the dataset's subset name (e.g. "en" for c4); None for datasets
            # without subsets
            "config": inherited("config"),
            "split": inherited("split", "train"),
            "text_field": text_field,
            "weight": weight,
            "shuffle_buffer": inherited("shuffle_buffer"),
            "pack_batch": inherited("pack_batch", 1000),
        })

    weights = normalize_weights([s["weight"] for s in resolved])
    for src, weight in zip(resolved, weights, strict=True):
        src["weight"] = weight
    return resolved


def load_source(src) -> datasets.IterableDataset:
    """
    the one i/o touch, kept alone so the rest of the pipeline is testable offline.

    streaming on by default. an explicit split is required so we get an
    IterableDataset back, not an IterableDatasetDict (no .num_shards, and
    split_dataset_by_node rejects it).
    TODO: a dataset registry once per-dataset split/field quirks pile up.
    """
    # local jsonl shards (e.g. pluggy/synth output) stream through the same
    # path as hub datasets: "data_files" (glob or list) instead of "name"
    if src["data_files"] is not None:
        return datasets.load_dataset(
            "json",
            data_files=src["data_files"],
            split=src["split"],
            streaming=True,
        )
    # when the dataset is in json format on huggingface, this no longer
    # works and we'll need to change the format
    return datasets.load_dataset(
        src["name"],
        src["config"],
        split=src["split"],
        streaming=True,
    )


def prepare_source_stream(
    dataset: datasets.IterableDataset, src, packer, seed: int, dp_rank: int, dp_size: int
) -> datasets.IterableDataset:
    """
    one loaded source -> a stream of packed seq_len rows: rename the text
    column -> shuffle -> split by node -> pack.

    each source is node-split BEFORE the mix, so sharding stays exactly the
    per-dataset thing it was for a single dataset and every rank divides up a
    source's shards rather than the mixed stream's.
    """
    # normalize whatever the source calls its text column to "text" so the
    # packer stays dataset-agnostic -- and, in a mixture, so every source's
    # stream carries the same schema into the interleave. rename_column, NOT a
    # lambda .map: the dataset is pickled to DataLoader workers under
    # forkserver/spawn (py3.14 default on linux) and lambdas don't pickle --
    # same reason Packer/Collator are top-level classes.
    if src["text_field"] != "text":
        dataset = dataset.rename_column(src["text_field"], "text")

    # the packer emits FEWER rows than it consumes docs, so every original
    # column has to be dropped or .map raises on the length mismatch. that
    # normally comes free from dataset.column_names, but it is None whenever a
    # streaming source's features aren't resolved (json-backed sources, and
    # anything built from a generator), and remove_columns=None silently keeps
    # every column. peek one example for the schema in that case -- here, while
    # it costs one example, rather than after .shuffle() where it would cost a
    # whole buffer fill.
    column_names = dataset.column_names or list(next(iter(dataset)))

    dataset = dataset.shuffle(buffer_size=src["shuffle_buffer"], seed=seed)
    # split by the data-parallel mesh coordinate: ranks in the same TP/CP
    # group share a dp_rank and so deterministically read the same shards.
    dataset = split_dataset_by_node(dataset, rank=dp_rank, world_size=dp_size)

    # rows out of here are exactly {"input_ids": [seq_len ints]} -- the same
    # schema for every source, which is what lets them be interleaved.
    return dataset.map(
        packer, batched=True,
        batch_size=src["pack_batch"],
        remove_columns=column_names,
    )


def build_source_stream(src, packer, seed: int, dp_rank: int, dp_size: int) -> datasets.IterableDataset:
    """load one source off the hub and turn it into a stream of packed rows."""
    dataset = load_source(src)
    if dp_rank == 0:
        print(f"{src['label']}: {dataset.num_shards} shards, {src['weight']:.1%} of tokens")
    return prepare_source_stream(dataset, src, packer, seed, dp_rank, dp_size)


def mixing_seed(seed: int, dp_rank: int) -> int:
    """
    the seed for the source draws, one per dp rank. with a single shared seed
    every rank would pick the SAME source for the same row index, so a step's
    realized mixture would come from (micro_batch x accum) draws instead of
    dp_size times that -- same mean, needlessly noisier per step. the offset is
    far enough from the per-source shuffle seeds (seed + source index) not to
    collide, and deterministic, so a resume lands on the same stream.
    """
    return seed + 10_000 + dp_rank


def mix_streams(
    streams: list[datasets.IterableDataset],
    probabilities: list[float],
    seed: int,
    stopping_strategy: str = "all_exhausted",
) -> datasets.IterableDataset:
    """
    interleave packed streams: draw the next row from source i with probability
    probabilities[i]. one row == seq_len tokens, so those probabilities are the
    token mixture (see the module docstring).

    the draws come from a seeded rng that lives in the dataset's state_dict, so
    the realized mixture is reproducible and survives checkpoint/resume along
    with the rest of the loader state.
    """
    assert len(streams) == len(probabilities), (
        f"{len(streams)} streams but {len(probabilities)} probabilities"
    )
    assert stopping_strategy in STOPPING_STRATEGIES, (
        f"unknown stopping_strategy {stopping_strategy!r}, expected one of {list(STOPPING_STRATEGIES)}"
    )
    if len(streams) == 1:
        # no interleave wrapper at all for a single source: it would only add an
        # rng and a cycling iterable that always picks source 0, and it caps
        # num_shards. keeps the single-dataset path identical to before.
        return streams[0]
    assert abs(sum(probabilities) - 1.0) < 1e-6, (
        f"probabilities must sum to 1, got {sum(probabilities)}"
    )
    return datasets.interleave_datasets(
        streams,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )


def build_dataloader(data_cfg, tokenizer: Tokenizer, ignore_id: int, dp_rank: int, dp_size: int):
    sources = resolve_sources(data_cfg)

    # eos delimits documents in the packed stream (real, trained-on signal);
    # pad only fills bestfit bins and must be a DIFFERENT vocab id so labels
    # can mask filler later without masking genuine doc boundaries (see
    # packers.py). token_to_id returns None for tokens missing from the
    # vocab -- fail here, loudly, not as a TypeError inside a worker.
    eos_id = tokenizer.token_to_id(data_cfg["eos_token"])
    pad_id = tokenizer.token_to_id(data_cfg["pad_token"])
    assert eos_id is not None, f"eos_token {data_cfg['eos_token']!r} not in tokenizer vocab"
    assert pad_id is not None, f"pad_token {data_cfg['pad_token']!r} not in tokenizer vocab"

    # stateless, so every source can .map the same instance
    packer = ListPacker(tokenizer, data_cfg["seq_len"], eos_id, pad_id)

    seed = data_cfg["seed"]
    streams = [
        # per-source seed offset so no two sources walk their shard order in
        # lockstep; source 0 keeps the bare config seed, which is what a
        # single-dataset config used to pass, so its data order is unchanged.
        build_source_stream(src, packer, seed + i, dp_rank, dp_size)
        for i, src in enumerate(sources)
    ]

    mixing_cfg = data_cfg.get("mixing", {})
    dataset = mix_streams(
        streams,
        [src["weight"] for src in sources],
        seed=mixing_seed(seed, dp_rank),
        stopping_strategy=mixing_cfg.get("stopping_strategy", "all_exhausted"),
    )

    num_workers = data_cfg["num_workers"]
    if dp_rank == 0 and dataset.num_shards < num_workers:
        # hf hands each worker a subset of the stream's shards and STOPS the
        # workers it has none for, so those processes sit idle. two things
        # shrink the count: a mixture's shard count is the MINIMUM over its
        # sources (one lightly-sharded source throttles the whole mix), and
        # datasets>=5 .shuffle() collapses it to 1 outright unless called with
        # max_buffer_input_shards=1.
        print(
            f"warning: {num_workers} dataloader workers but the stream has "
            f"{dataset.num_shards} shard(s); the rest will idle"
        )

    # the loader only ever sees the MICRO batch: one iteration == one
    # forward/backward, which is also what torch's own DataLoader(batch_size=)
    # means. how many microbatches make up an optimizer step (grad
    # accumulation) is entirely the trainer's business -- it derives that from
    # global_batch_size and validates divisibility against dp_size, so nothing
    # here needs to know the global batch exists.
    #return torch.utils.data.DataLoader(
    return StatefulDataLoader(
        dataset,
        batch_size=data_cfg["micro_batch_size"],
        num_workers=num_workers,
        collate_fn=Collator(tokenizer, ignore_id),
        persistent_workers=True,
        pin_memory=True
    )
