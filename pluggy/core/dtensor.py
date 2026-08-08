"""
dtensor: a logical tensor + how it is laid out over the mesh.

this is a BOOKKEEPING WRAPPER WITH EXPLICIT REDISTRIBUTE, not a
__torch_dispatch__ tensor subclass with sharding propagation for every aten
op (roadmap 2.1's key decision). pytorch's DTensor does full op dispatch;
that's a compiler-sized project, and none of fsdp2/tp/cp/checkpointing here
needs it -- they all know exactly which collective they want. consequence:
you never compute THROUGH a DTensor. you ask for the local tensor, do plain
torch math, and use redistribute/full_tensor when the layout has to change.

the companion decision (FSDP2_SCOPE.md stage 1): the optimizer never sees a
DTensor. params stay plain tensors at runtime -- fused adamw, foreach calls,
and clip_grad_norm keep working untouched -- and DTensor is how sharded
state is described, moved, and (phase 3) written to checkpoint metadata.

the redistribute table, per mesh axis:

    | from -> to            | collective                    |
    |-----------------------|-------------------------------|
    | Shard(d) -> Replicate | all_gather(gather_dim=d)      |
    | Replicate -> Shard(d) | local slice (no comm)         |
    | Partial -> Replicate  | all_reduce(op)                |
    | Partial -> Shard(d)   | reduce_scatter(scatter_dim=d) |
    | Shard(i) -> Shard(j)  | all_to_all                    |

everything else (Replicate/Shard -> Partial) has no consumer and asserts.

v0 limits, asserted loudly (same policy as the collectives): even shards
only, and no two mesh axes may shard the same tensor dim (strided/nested
sharding is a tp x fsdp2 problem for later).

run tests with: uv run tests/dtensor.py  (gloo/cpu, no gpus needed)
"""

import torch

from pluggy.core.collective import MeshLike, all_gather, all_reduce, reduce_scatter, all_to_all
from pluggy.core.placement import Partial, Placement, Replicate, Shard


class DTensor:
    """
    (local_tensor, mesh, placements): `placements` maps mesh axis name ->
    Placement; axes not named are Replicate. Shard.dim indexes the LOGICAL
    (global) tensor's dims, which equal the local tensor's dims -- sharding
    changes sizes, never rank.
    """

    def __init__(self, local: torch.Tensor, mesh: MeshLike, placements: dict[str, Placement]):
        # not validated against the mesh here beyond shape bookkeeping: the
        # constructor trusts its callers (from_local / from_full /
        # redistribute), which do the asserting
        self.local = local
        self.mesh = mesh
        # normalized: explicit Replicate entries are dropped, so two ways of
        # writing the same layout compare equal and checkpoint metadata is
        # canonical
        self.placements = {
            axis: p for axis, p in placements.items() if not isinstance(p, Replicate)
        }

    @classmethod
    def from_local(cls, local: torch.Tensor, mesh: MeshLike, placements: dict[str, Placement]) -> "DTensor":
        """
        wrap the shard/partial/replica this rank already holds. trusts that
        `local` really is the coordinate-th slice (or partial value) --
        there is nothing to check it against without comm.
        """
        cls._validate(local.ndim, mesh, placements)
        return cls(local, mesh, placements)

    @classmethod
    def from_full(cls, full: torch.Tensor, mesh: MeshLike, placements: dict[str, Placement]) -> "DTensor":
        """
        construct from the full logical tensor (identical on every rank, the
        caller's contract) by slicing out this rank's shards locally -- the
        Replicate -> Shard(d) row of the table, applied per axis, no comm.
        this is fsdp2's v0 init path: init full weights with a fixed seed,
        slice.

        Partial makes no sense here (the full tensor IS the logical value;
        calling one rank's copy a partial term would change it) and asserts.
        """
        cls._validate(full.ndim, mesh, placements)
        local = full
        for axis, placement in placements.items():
            if isinstance(placement, Replicate):
                continue
            assert isinstance(placement, Shard), f"from_full only takes Shard/Replicate, got {axis}: {placement}"
            local = _slice(local, mesh, axis, placement.dim)
        # contiguous + detached from `full`: a shard that aliases the full
        # tensor keeps the whole storage alive, which is the opposite of the
        # memory win sharding exists for
        return cls(local.contiguous().clone(), mesh, placements)

    @staticmethod
    def _validate(ndim: int, mesh: MeshLike, placements: dict[str, Placement]) -> None:
        sharded_dims = []
        for axis, placement in placements.items():
            # mesh.size raises KeyError on an unknown axis, which is the
            # loud failure we want; calling it here surfaces it at
            # construction instead of first redistribute
            mesh.size(axis)
            if isinstance(placement, Shard):
                assert 0 <= placement.dim < ndim, f"Shard({placement.dim}) out of range for ndim={ndim}"
                sharded_dims.append(placement.dim)
        assert len(sharded_dims) == len(set(sharded_dims)), (
            f"two mesh axes shard the same tensor dim ({sharded_dims}); "
            "nested/strided sharding is out of scope (v0)"
        )

    def to_local(self) -> torch.Tensor:
        return self.local

    @property
    def global_shape(self) -> tuple[int, ...]:
        shape = list(self.local.shape)
        for axis, placement in self.placements.items():
            if isinstance(placement, Shard):
                shape[placement.dim] *= self.mesh.size(axis)
        return tuple(shape)

    def placement(self, axis: str) -> Placement:
        return self.placements.get(axis, Replicate())

    def redistribute(self, axis: str, to: Placement) -> "DTensor":
        """
        change ONE axis's placement via the table; returns a new DTensor
        (self is never mutated -- its (local, placements) invariant must
        survive, so paths whose collective works in place clone first).
        multi-axis changes compose: call once per axis.
        """
        src = self.placement(axis)
        if src == to:
            return self

        match (src, to):
            case (Shard(dim=d), Replicate()):
                new_local = all_gather(self.local, self.mesh, axis, gather_dim=d)
            case (Replicate(), Shard(dim=d)):
                new_local = _slice(self.local, self.mesh, axis, d).contiguous().clone()
            case (Partial(op=op), Replicate()):
                # all_reduce is in-place on its input (and the gloo "avg"
                # path pre-divides it), so reduce a clone: self must keep
                # holding a valid partial term
                new_local = all_reduce(self.local.clone(), self.mesh, axis, op)
            case (Partial(op=op), Shard(dim=d)):
                new_local = reduce_scatter(self.local, self.mesh, axis, op, scatter_dim=d)
            case (Shard(dim=i), Shard(dim=j)):
                # each rank owns (i-shard mine, all of j); it must end with
                # (all of i, j-shard mine). send the j-chunk k of my i-shard
                # to coordinate k; the received pieces, ordered by sender =
                # i-shard owner, concatenate back along i
                new_local = all_to_all(self.local, self.mesh, axis, split_dim=j, concat_dim=i)
            case _:
                raise AssertionError(
                    f"unsupported redistribute {src} -> {to} on axis {axis!r} "
                    "(nothing produces Replicate/Shard -> Partial)"
                )

        new_placements = dict(self.placements)
        new_placements[axis] = to
        return DTensor(new_local, self.mesh, new_placements)

    def full_tensor(self) -> torch.Tensor:
        """
        materialize the full logical tensor on every rank: Partial axes
        reduce first (cheaper -- the operand is still shard-sized), then
        Shard axes gather. returns a plain tensor; when no comm was needed
        it aliases the local one.
        """
        out = self
        for axis, placement in list(out.placements.items()):
            if isinstance(placement, Partial):
                out = out.redistribute(axis, Replicate())
        for axis, placement in list(out.placements.items()):
            if isinstance(placement, Shard):
                out = out.redistribute(axis, Replicate())
        return out.local

    def __repr__(self) -> str:
        placements = ", ".join(f"{a}: {p}" for a, p in self.placements.items()) or "Replicate()"
        return f"DTensor(local_shape={tuple(self.local.shape)}, global_shape={self.global_shape}, {placements})"


def _slice(tensor: torch.Tensor, mesh: MeshLike, axis: str, dim: int) -> torch.Tensor:
    """this rank's Shard(dim) slice along `axis`: chunk dim evenly, take the coordinate-th."""
    size = mesh.size(axis)
    assert tensor.shape[dim] % size == 0, (
        f"shape[{dim}]={tensor.shape[dim]} not divisible by mesh axis {axis!r} size {size}; "
        "even shards only (v0), padding is a documented later problem"
    )
    return tensor.chunk(size, dim=dim)[mesh.coordinate(axis)]
