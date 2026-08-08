"""
tests for pluggy/core/{placement,dtensor}.py (roadmap 2.1 / FSDP2_SCOPE
stage 1).

every cell of the redistribute table is checked against full_tensor()
ground truth that each rank recomputes locally (deterministic arange
tensors, no comm needed for the comparison):

    Shard(d) -> Replicate    all_gather, bit-exact roundtrip
    Replicate -> Shard(d)    local slice
    Partial -> Replicate     all_reduce (sum and avg, incl. the gloo
                             avg emulation not corrupting the source)
    Partial -> Shard(d)      reduce_scatter, plus the fsdp identity:
                             partial->shard then shard->replicate ==
                             partial->replicate
    Shard(i) -> Shard(j)     all_to_all

plus the loud-failure paths: non-divisible shapes, unsupported
transitions, unknown mesh axes, two axes sharding one tensor dim.

gloo/cpu via mp.spawn, no gpus needed:
    uv run tests/dtensor.py --world-size 4
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pluggy.core.dtensor import DTensor
from pluggy.core.mesh import Mesh
from pluggy.core.placement import Partial, Replicate, Shard


def _full(mesh: Mesh, dim: str) -> torch.Tensor:
    # deterministic, rank-independent, divisible along both dims by the
    # axis size (and by 2*size, so Shard(i)->Shard(j) stays even too)
    size = mesh.size(dim)
    return torch.arange(size * 4 * size * 6, dtype=torch.float32).reshape(size * 4, size * 6)


def test_shard_roundtrip(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    for d in (0, 1):
        dt = DTensor.from_full(full, mesh, {dim: Shard(d)})
        expected_local = full.chunk(mesh.size(dim), dim=d)[mesh.coordinate(dim)]
        assert torch.equal(dt.to_local(), expected_local), f"from_full slice wrong (dim {d})"
        assert dt.global_shape == tuple(full.shape)

        gathered = dt.redistribute(dim, Replicate())
        assert torch.equal(gathered.to_local(), full), f"shard->replicate not bit-exact (dim {d})"
        assert torch.equal(dt.full_tensor(), full)


def test_replicate_to_shard(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    dt = DTensor.from_local(full, mesh, {})
    sharded = dt.redistribute(dim, Shard(1))
    expected = full.chunk(mesh.size(dim), dim=1)[mesh.coordinate(dim)]
    assert torch.equal(sharded.to_local(), expected)
    # the slice must not alias the replica's storage
    assert sharded.to_local().data_ptr() != dt.to_local().data_ptr() or mesh.size(dim) == 1


def test_partial_to_replicate(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    size = mesh.size(dim)
    coord = mesh.coordinate(dim)
    # rank r holds full * (r+1); the logical (summed) tensor is known locally
    term = full * (coord + 1)
    weight_sum = size * (size + 1) / 2

    for op, expected in (("sum", full * weight_sum), ("avg", full * weight_sum / size)):
        dt = DTensor.from_local(term.clone(), mesh, {dim: Partial(op)})
        reduced = dt.redistribute(dim, Replicate())
        torch.testing.assert_close(reduced.to_local(), expected)
        # redistribute must not mutate the source: all_reduce works in
        # place, and the gloo "avg" path pre-divides its input -- both must
        # hit a clone, not dt.local
        assert torch.equal(dt.to_local(), term), f"partial->replicate ({op}) corrupted the source dtensor"
        torch.testing.assert_close(dt.full_tensor(), expected)


def test_partial_to_shard(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    size = mesh.size(dim)
    term = full * (mesh.coordinate(dim) + 1)
    logical = full * (size * (size + 1) / 2)

    dt = DTensor.from_local(term, mesh, {dim: Partial("sum")})
    sharded = dt.redistribute(dim, Shard(0))
    expected = logical.chunk(size, dim=0)[mesh.coordinate(dim)]
    torch.testing.assert_close(sharded.to_local(), expected)

    # the identity fsdp is built on, at the dtensor level:
    # partial -> shard -> replicate == partial -> replicate
    via_scatter = sharded.redistribute(dim, Replicate()).to_local()
    via_reduce = dt.redistribute(dim, Replicate()).to_local()
    torch.testing.assert_close(via_scatter, via_reduce)


def test_shard_to_shard(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    size = mesh.size(dim)
    coord = mesh.coordinate(dim)

    for i, j in ((0, 1), (1, 0)):
        dt = DTensor.from_full(full, mesh, {dim: Shard(i)})
        moved = dt.redistribute(dim, Shard(j))
        expected = full.chunk(size, dim=j)[coord]
        assert torch.equal(moved.to_local(), expected), f"shard({i})->shard({j}) wrong"
        assert torch.equal(moved.full_tensor(), full)


def test_multi_axis(mesh: Mesh, dim: str) -> None:
    world = dist.get_world_size()
    if world % 2 != 0:
        return
    # every rank builds it together, in the same order (new_group is
    # collective); axes: a=2, b=world/2
    mesh2 = Mesh({"a": 2, "b": world // 2})
    b_size = mesh2.size("b")
    full = torch.arange(4 * 2 * 3 * b_size, dtype=torch.float32).reshape(4 * 2, 3 * b_size)

    # sharded over "a", partial over "b": the layout fsdp2 x anything lands in
    shard = DTensor.from_full(full, mesh2, {"a": Shard(0)}).to_local()
    term = shard * (mesh2.coordinate("b") + 1)
    dt = DTensor.from_local(term, mesh2, {"a": Shard(0), "b": Partial("sum")})

    assert dt.global_shape == tuple(full.shape)
    expected = full * (b_size * (b_size + 1) / 2)
    torch.testing.assert_close(dt.full_tensor(), expected)


def test_loud_failures(mesh: Mesh, dim: str) -> None:
    full = _full(mesh, dim)
    size = mesh.size(dim)

    def raises(fn, exc=AssertionError) -> bool:
        try:
            fn()
        except exc:
            return True
        return False

    if size > 1:
        ragged = torch.zeros(size * 4 + 1, 3)
        assert raises(lambda: DTensor.from_full(ragged, mesh, {dim: Shard(0)})), "non-divisible must assert"

    dt = DTensor.from_local(full, mesh, {})
    assert raises(lambda: dt.redistribute(dim, Partial("sum"))), "replicate->partial must assert"
    sharded = DTensor.from_full(full, mesh, {dim: Shard(0)})
    assert raises(lambda: sharded.redistribute(dim, Partial("sum"))), "shard->partial must assert"

    assert raises(lambda: DTensor.from_local(full, mesh, {"no_such_axis": Shard(0)}), KeyError), \
        "unknown mesh axis must fail loudly"
    assert raises(lambda: DTensor.from_local(full, mesh, {dim: Shard(2)})), "out-of-range shard dim must assert"
    assert raises(lambda: DTensor.from_full(full, mesh, {dim: Partial("sum")})), "from_full with Partial must assert"

    # same-placement redistribute is a no-op, not a collective
    assert sharded.redistribute(dim, Shard(0)) is sharded


TESTS = [
    ("test_shard_roundtrip", test_shard_roundtrip),
    ("test_replicate_to_shard", test_replicate_to_shard),
    ("test_partial_to_replicate", test_partial_to_replicate),
    ("test_partial_to_shard", test_partial_to_shard),
    ("test_shard_to_shard", test_shard_to_shard),
    ("test_multi_axis", test_multi_axis),
    ("test_loud_failures", test_loud_failures),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29534")
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    mesh, dim = Mesh({"dp": world_size}), "dp"
    failed = []
    for name, test in TESTS:
        try:
            test(mesh, dim)
            err = None
        except AssertionError as e:
            err = f"FAIL: {e}"
        except Exception as e:
            err = f"ERROR: {type(e).__name__}: {e}"
        dist.barrier()
        if rank == 0:
            print(f"[{name}] {err or 'PASS'}")
        if err:
            failed.append(name)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.world_size,), nprocs=args.world_size)
