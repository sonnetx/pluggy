"""
placement types for dtensor: how one logical tensor relates to the local
tensor each rank holds, per mesh axis.

- Shard(dim): the logical tensor is split along tensor dim `dim`; this rank
  holds the coordinate-th contiguous slice. even splits only (the
  collectives assert divisibility; padding is a documented later problem,
  see FSDP2_SCOPE.md stage 1).
- Replicate(): every rank holds the full logical tensor.
- Partial(op): every rank holds a partial value; the logical tensor is the
  `op`-reduction across the axis (the state a matmul against sharded inputs
  leaves you in, before any comm).

frozen dataclasses on purpose: they are value types that end up as keys and
in checkpoint metadata, so they need eq/hash/repr, not behavior. the
behavior (the redistribute table) lives on DTensor, which is what talks to
the collectives.
"""

from dataclasses import dataclass

from pluggy.core.collective import ReduceOpName


@dataclass(frozen=True)
class Placement:
    pass


@dataclass(frozen=True)
class Shard(Placement):
    dim: int


@dataclass(frozen=True)
class Replicate(Placement):
    pass


@dataclass(frozen=True)
class Partial(Placement):
    op: ReduceOpName = "sum"
