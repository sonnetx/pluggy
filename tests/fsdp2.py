"""
correctness tests for pluggy/parallelism/fsdp2.py (roadmap 2.2 /
FSDP2_SCOPE stage 2).

the invariant is the same one tests/data_parallel.py pins for ddp: N ranks
x bs=1 must produce the same result as 1 process x bs=N -- fsdp2 computes
the identical thing to ddp, just sharded. the batch is deterministic, so
every rank recomputes the single-process reference locally and compares
its own SHARD of the reference grads/params; no gathers are needed for
the comparison.

the model is a tiny real Qwen3: the tied embedding (one param, two grad
contributions, must land in ONE unit) and the rmsnorm/rope structure are
exactly what breaks naive implementations.

beyond parity, the invariants that are invisible from the outside (an
fsdp that never frees memory still trains perfectly):
- per-rank param/grad/adam-state numel shrinks by 1/dp_shard
- block units' full storages are actually freed after forward; the root
  unit stays gathered (that is the ARObjective fix under test in
  test_fused_ce_objective)
- dp=1 is a total no-op

gloo/cpu via mp.spawn, no gpus needed:
    uv run tests/fsdp2.py --world-size 4
"""

import argparse
import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from pluggy.core.grad_helper import clip_grad_norm
from pluggy.core.mesh import Mesh
from pluggy.models.qwen3.qwen3 import Qwen3
from pluggy.objectives.autoregressive import ARObjective
from pluggy.parallelism.fsdp2 import FSDP2

CFG = {
    "num_layers": 2,
    "num_heads": 2,
    "num_kv_heads": 1,
    "emb_dim": 32,
    "head_dim": 16,
    "vocab_size": 64,
}
SEQ = 16


def build_model(seed: int) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(**CFG)
    model.init_weights()
    return model


def global_batch(n_seqs: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, CFG["vocab_size"], (n_seqs, SEQ), generator=g)


def loss_fn(model: Qwen3, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG["vocab_size"]), ids[:, 1:].reshape(-1)
    )


def my_shard(full: torch.Tensor, mesh: Mesh, dim: str) -> torch.Tensor:
    return full.chunk(mesh.size(dim), dim=0)[mesh.coordinate(dim)]


def grad_mismatches(model, reference, mesh, dim) -> list[str]:
    """
    returns error strings instead of raising: tests with collectives still
    ahead must NOT bail early on a comparison failure, or the failing rank
    stops issuing collectives while the passing ranks keep going, and the
    whole suite deadlocks instead of failing. collect, finish the
    collective schedule, then raise.
    """
    errors = []
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        try:
            torch.testing.assert_close(
                p.grad, my_shard(p_ref.grad, mesh, dim), rtol=1e-4, atol=1e-6
            )
        except AssertionError as e:
            errors.append(f"param {name}: {e}")
    return errors


def assert_grads_match_reference(model, reference, mesh, dim) -> None:
    errors = grad_mismatches(model, reference, mesh, dim)
    assert not errors, "\n".join(errors)


def test_shard_shapes(mesh, dim):
    size = mesh.size(dim)
    model = build_model(seed=dist.get_rank())
    full_numel = sum(p.numel() for p in build_model(seed=0).parameters())
    FSDP2(model, mesh, dim)

    # the tie must survive sharding: one param object, two module slots
    assert model.lm_head.weight is model.token_emb.weight, "tied weight broken by sharding"
    sharded_numel = sum(p.numel() for p in model.parameters())
    assert sharded_numel * size == full_numel, (
        f"per-rank param numel {sharded_numel} != total/{size}"
    )
    # replication check (ddp's test_replicate equivalent): every rank's
    # shard must be the coord-th slice of seed-0 weights
    reference = build_model(seed=0)
    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            p, my_shard(p_ref, mesh, dim), rtol=0, atol=0,
            msg=f"param {name} shard is not the coord-th slice of rank 0's weights",
        )


def test_grad_parity(mesh, dim):
    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)

    reference = build_model(seed=0)
    batch = global_batch(mesh.size(dim))
    loss_fn(reference, batch).backward()

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    errors = []
    for cycle in range(2):  # two full cycles: catches rearm/state-clearing bugs
        loss_fn(model, local).backward()
        fsdp.sync()
        errors += [f"cycle {cycle}: {e}" for e in grad_mismatches(model, reference, mesh, dim)]
        for p in model.parameters():
            p.grad = None  # what optimizer.zero_grad(set_to_none=True) does
    assert not errors, "\n".join(errors)


def test_fused_ce_objective(mesh, dim):
    """
    the FSDP2_SCOPE stage 0 blocker, exercised directly: ARObjective's
    fused path consumes model.lm_head.weight AFTER forward returns. the
    root unit staying gathered through the loss is what makes this work;
    a reshard-everything-after-forward design fails here.
    """
    objective = ARObjective(ignore_index=-100, fused_linear_ce=True, ce_chunk_size=8)

    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)

    reference = build_model(seed=0)
    batch = global_batch(mesh.size(dim))
    objective.compute_loss(reference, {"input_ids": batch, "labels": batch}).backward()

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    loss = objective.compute_loss(model, {"input_ids": local, "labels": local})
    loss.backward()
    fsdp.sync()
    assert_grads_match_reference(model, reference, mesh, dim)


def test_grad_accumulation(mesh, dim, microbatches: int):
    """
    N ranks x K microbatches of bs=1 == 1 process x bs=(N*K). unlike ddp
    there is no no_sync mode: every microbatch reduce-scatters into the
    sharded fp32 accumulator (never a full grad held across microbatches),
    and the 1/K loss scaling composes the same way.
    """
    size = mesh.size(dim)
    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)

    reference = build_model(seed=0)
    batch = global_batch(size * microbatches)
    loss_fn(reference, batch).backward()

    for k in range(microbatches):
        local = batch[k * size + mesh.coordinate(dim)].unsqueeze(0)
        (loss_fn(model, local) / microbatches).backward()
    fsdp.sync()
    assert_grads_match_reference(model, reference, mesh, dim)


def _state_numel(optimizer) -> int:
    return sum(
        v.numel()
        for st in optimizer.state.values()
        for v in st.values()
        if torch.is_tensor(v) and v.dim() > 0
    )


def test_multi_step(mesh, dim):
    """
    5 optimizer steps: catches optimizer-state sharding bugs a single
    backward can't see. sgd+momentum for the parity half -- its update is
    linear in the grad, so the ~1e-6 reduction-order noise between fsdp
    and the reference stays ~1e-6 in the params. adam's m/sqrt(v)
    normalization amplifies that noise into sign flips wherever a grad
    entry is near zero (measured: step-0 divergence at 4 ranks), so adam
    is verified by what actually matters and is stable: its state
    allocates at shard shapes, which is the memory win this phase exists
    for. both optimizers are built AFTER wrapping (trainer order).
    """
    size = mesh.size(dim)
    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    reference = build_model(seed=0)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-2, momentum=0.9)
    batch = global_batch(size)
    local = batch[mesh.coordinate(dim)].unsqueeze(0)

    errors = []
    for step in range(5):
        loss_fn(reference, batch).backward()
        ref_optimizer.step()
        ref_optimizer.zero_grad()

        loss_fn(model, local).backward()
        fsdp.sync()
        optimizer.step()
        optimizer.zero_grad()

        # collect, don't raise: bailing out of the loop on one rank while
        # the others keep launching collectives deadlocks the suite
        for (name, p), (_, p_ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            try:
                torch.testing.assert_close(
                    p, my_shard(p_ref, mesh, dim), rtol=1e-4, atol=1e-6
                )
            except AssertionError as e:
                errors.append(f"step {step}, param {name}: {e}")

    # momentum buffers are per-param state and must be shard-shaped, and so
    # must adam's exp_avg/exp_avg_sq after one adamw step
    assert _state_numel(optimizer) * size == _state_numel(ref_optimizer), (
        "sgd momentum state is not sharded -- the optimizer saw unsharded params"
    )
    adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn(model, local).backward()
    fsdp.sync()
    adam.step()
    adam.zero_grad()
    sharded_numel = sum(p.numel() for p in model.parameters())
    assert _state_numel(adam) == 2 * sharded_numel, (
        f"adam state numel {_state_numel(adam)} != 2x sharded param numel {2 * sharded_numel}"
    )

    assert not errors, "\n".join(errors)


def test_storage_freed(mesh, dim):
    """
    fsdp that never frees still trains perfectly and just doesn't do its
    job: assert the block units' full storages are 0 after forward while
    the root unit (tied embedding + head + norm) is still gathered, then
    that EVERYTHING is freed after backward + sync.
    """
    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)
    local = global_batch(mesh.size(dim))[mesh.coordinate(dim)].unsqueeze(0)

    loss = loss_fn(model, local)
    errors = []  # collect, don't raise: backward's collectives still ahead
    for unit in fsdp._units:
        for ps in unit.params:
            allocated = ps.full.untyped_storage().size()
            if unit.reshard_after_forward and allocated != 0:
                errors.append(f"unit {unit.name}: full storage not freed after forward")
            if not unit.reshard_after_forward and allocated != ps.full_nbytes:
                errors.append(
                    f"unit {unit.name}: root must stay gathered until backward "
                    "(the objective reads lm_head.weight after forward)"
                )

    loss.backward()
    fsdp.sync()
    for unit in fsdp._units:
        for ps in unit.params:
            if ps.full.untyped_storage().size() != 0:
                errors.append(f"unit {unit.name}: full storage not freed after backward")
            if ps.param.data_ptr() != ps.sharded.data_ptr():
                errors.append(f"unit {unit.name}: param not pointing at the sharded master at step time")
    assert not errors, "\n".join(sorted(set(errors)))


def test_clip_grad_norm(mesh, dim):
    """the sharded-norm path, live: shard_dims=(dim,) must reproduce the
    single-process norm and the post-clip grads."""
    model = build_model(seed=dist.get_rank())
    fsdp = FSDP2(model, mesh, dim)

    reference = build_model(seed=0)
    batch = global_batch(mesh.size(dim))
    loss_fn(reference, batch).backward()
    ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1e-3)

    local = batch[mesh.coordinate(dim)].unsqueeze(0)
    loss_fn(model, local).backward()
    fsdp.sync()
    norm = clip_grad_norm(model.parameters(), mesh, max_norm=1e-3, shard_dims=(dim,))

    torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=1e-7)
    assert_grads_match_reference(model, reference, mesh, dim)


def test_noop_at_size_1(mesh, dim):
    # a mesh where the shard axis has size 1: the ctor must be a total
    # no-op -- params untouched, no hooks, forward works
    world = dist.get_world_size()
    mesh1 = Mesh({"other": world, "shard": 1})
    model = build_model(seed=0)
    reference = build_model(seed=0)
    FSDP2(model, mesh1, "shard")

    for (name, p), (_, p_ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert p.shape == p_ref.shape, f"param {name} touched at dp=1"
    batch = global_batch(1)
    torch.testing.assert_close(loss_fn(model, batch), loss_fn(reference, batch))


TESTS = [
    ("test_shard_shapes", test_shard_shapes),
    ("test_grad_parity", test_grad_parity),
    ("test_fused_ce_objective", test_fused_ce_objective),
    ("test_accum_2", partial(test_grad_accumulation, microbatches=2)),
    ("test_accum_3", partial(test_grad_accumulation, microbatches=3)),
    ("test_multi_step", test_multi_step),
    ("test_storage_freed", test_storage_freed),
    ("test_clip_grad_norm", test_clip_grad_norm),
    ("test_noop_at_size_1", test_noop_at_size_1),
]


def _worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29535")
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
