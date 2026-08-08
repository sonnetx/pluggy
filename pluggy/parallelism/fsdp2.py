"""
fsdp2: shard every parameter along the mesh's dp axis, gather it back just
in time for compute, and reduce-scatter grads so each rank only ever
optimizes (and holds adam state for) its own shard. per-parameter Shard(0)
over plain tensors -- the DTensor decision from core/dtensor.py: placements
are bookkeeping, the optimizer and autograd only ever see plain tensors.

the unit is the reshard granularity (roadmap principle 4): one unit per
transformer block, plus a ROOT unit holding everything outside the blocks
(token_emb + lm_head + final norm for qwen3). the tied embedding forces
embed and head into one unit -- one weight, two consumers; sharding them in
different units would double-gather one storage and corrupt the grad flow.
the root unit is also the fix for the ARObjective blocker (FSDP2_SCOPE
stage 0): the objective consumes `model.lm_head.weight` AFTER the model's
forward has returned (fused linear+ce), so the root unit does NOT reshard
after forward -- it stays gathered through the loss and backward and
reshards after its grads are reduced, which costs nothing since its params
are needed first thing in backward anyway.

parameter lifecycle, per unit, per step:

    unshard (pre-forward hook): resize the preallocated full tensor's
        storage back up, all_gather the shard into it, point p.data at it.
    reshard (post-forward hook, block units only): resize the full
        storage to 0. p.data keeps pointing at the (now empty) full tensor
        on purpose: anything that touches the param before the pre-backward
        re-gather crashes loudly instead of silently computing on a shard.
    re-unshard (full_backward_pre hook): same gather, same storage, same
        tensor object -- the values autograd saved for backward were views
        of this storage, and the master shards haven't changed within the
        step, so the re-gather reproduces them exactly.
    post-backward (per-param post_accumulate_grad hook): reduce_scatter
        the full grad ("avg" over dp) into the sharded fp32 grad
        accumulator, drop the full grad, and once every param in the unit
        has fired, reshard with p.data pointed back at the SHARDED master
        -- which is what the optimizer must see.
    sync() (trainer, between backward and clip/step): publishes each
        param's accumulated sharded grad as p.grad. after this, p.data and
        p.grad are both shard-shaped: fused adamw allocates its state
        against shard shapes, which is the actual memory win, and
        clip_grad_norm gets `shard_dims=(dim,)`.

grad accumulation: grads are reduce-scattered EVERY microbatch and
accumulated in the sharded fp32 buffer (the FSDP2_SCOPE stage 2 decision:
one extra collective per microbatch, but never a full unsharded grad held
across microbatches -- holding it would forfeit the memory sharding exists
for). consequence: there is no no_sync mode. `requires_sync` exists so the
trainer loop keeps one idiom with DDP, and is deliberately ignored --
"avg" per microbatch summed over microbatches equals avg of the
accumulated grads, and the trainer's 1/K loss scaling composes the same
way it does under DDP.

v0 scope (correctness first, per roadmap principle 2):
- collectives are SYNCHRONOUS: no comm stream, no prefetch of unit i+1,
  no overlap. gloo/cpu parity comes first; the stream/overlap work is the
  next stage and changes none of this file's semantics.
- gathers move fp32 (the master dtype). the bf16-cast-on-gather halving of
  comm is a later, measured change; autocast keeps handling activation
  dtype exactly as it does under DDP.
- even shards only: every param's dim 0 must divide by the axis size,
  asserted loudly at wrap time (same policy as the collectives).

dp=1 is a total no-op, like DDP. one FSDP2 instance per model: hooks are
registered once and never removed. checkpointing a sharded model is phase
3; Trainer.checkpoint must not be pointed at an fsdp model until then.
"""

import torch
import torch._dynamo
import torch.nn as nn

from pluggy.core.collective import MeshLike, all_gather, broadcast, reduce_scatter


class _ParamState:
    """one sharded parameter: the master shard, and the reusable full tensor."""

    def __init__(self, param: nn.Parameter, mesh: MeshLike, dim: str):
        size = mesh.size(dim)
        assert param.shape[0] % size == 0, (
            f"param dim 0 ({param.shape[0]}) not divisible by mesh axis "
            f"{dim!r} size {size}; even shards only (v0)"
        )
        self.param = param
        # contiguous clone: the shard must own its storage, not keep the
        # full init tensor alive as a view base
        self.sharded = param.data.chunk(size, dim=0)[mesh.coordinate(dim)].clone()
        # preallocated once, storage resized 0 <-> full for the rest of the
        # run: stable tensor identity is what lets backward's saved
        # references (and, later, compile guards) survive resharding
        self.full = torch.empty_like(param.data)
        self.full_nbytes = self.full.untyped_storage().size()
        self.full.untyped_storage().resize_(0)
        self.grad_shard: torch.Tensor | None = None
        param.data = self.sharded

    def unshard(self, mesh: MeshLike, dim: str) -> None:
        if self.full.untyped_storage().size() != self.full_nbytes:
            self.full.untyped_storage().resize_(self.full_nbytes)
        all_gather(self.sharded, mesh, dim, gather_dim=0, output_tensor=self.full)
        self.param.data = self.full

    def free_full(self) -> None:
        self.full.untyped_storage().resize_(0)


class _Unit:
    def __init__(self, name: str, params: list[_ParamState], reshard_after_forward: bool):
        self.name = name
        self.params = params
        self.reshard_after_forward = reshard_after_forward
        self.n_grads = sum(1 for ps in params if ps.param.requires_grad)
        self.pending = self.n_grads


class FSDP2:
    def __init__(self, model: nn.Module, mesh: MeshLike, dim: str = "dp"):
        self.mesh = mesh
        self.dim = dim
        self.group_size = mesh.size(dim)
        # api parity with DDP only; see the module docstring -- fsdp2
        # reduce-scatters every microbatch by design, so this is ignored
        self.requires_sync = True

        if self.group_size == 1:
            return

        assert hasattr(model, "blocks"), (
            "fsdp2 units are per transformer block; the model contract "
            "(roadmap M1) requires model.blocks"
        )

        # ranks must agree bit-for-bit on the full weights before slicing,
        # even if per-rank rng drifted (same belt-and-suspenders as DDP)
        for tensor in [*model.parameters(), *model.buffers()]:
            broadcast(tensor.detach(), mesh, dim, src=0)

        # units: one per block + the root unit with everything else. params
        # collected via .parameters(), which dedupes the tied weight, and
        # id() bookkeeping splits block params from root params.
        block_param_ids = {
            id(p) for block in model.blocks for p in block.parameters()
        }
        self._units: list[_Unit] = []
        for i, block in enumerate(model.blocks):
            unit = _Unit(
                f"block{i}",
                [_ParamState(p, mesh, dim) for p in block.parameters()],
                reshard_after_forward=True,
            )
            self._units.append(unit)
            self._install_hooks(block, unit)

        root_params = [
            _ParamState(p, mesh, dim)
            for p in model.parameters()
            if id(p) not in block_param_ids
        ]
        # reshard_after_forward=False: the objective consumes lm_head.weight
        # after forward returns (fused linear+ce), and backward needs these
        # params first anyway -- see the module docstring
        self._root = _Unit("root", root_params, reshard_after_forward=False)
        self._units.append(self._root)
        self._install_hooks(model, self._root)

    def _install_hooks(self, module: nn.Module, unit: _Unit) -> None:
        module.register_forward_pre_hook(
            lambda m, args, unit=unit: self._unshard(unit)
        )
        if unit.reshard_after_forward:
            module.register_forward_hook(
                lambda m, args, out, unit=unit: self._reshard_after_forward(unit)
            )
            module.register_full_backward_pre_hook(
                lambda m, grad_out, unit=unit: self._unshard(unit)
            )
        for ps in unit.params:
            if ps.param.requires_grad:
                ps.param.register_post_accumulate_grad_hook(
                    lambda p, ps=ps, unit=unit: self._on_grad_ready(ps, unit)
                )

    # every hook body below is hidden from dynamo. nn.Module.compile() compiles
    # _call_impl, which RUNS THE HOOKS, so without this the unshard lands inside
    # the traced graph -- and `p.data = ...` lowers to a set_ on a graph input
    # that also gets its metadata mutated, which AOTAutograd rejects outright:
    #
    #   AssertionError: Encountered a set_ on a graph input, but the input has
    #   other mutations that we cannot keep in the graph. mutates_metadata=True
    #
    # torch._dynamo.disable makes dynamo graph-break at the hook and run it
    # eagerly, so the storage juggling stays where it belongs (between graphs)
    # and the block body still compiles as one region. this is the same policy
    # upstream applies to its own fsdp2 (torch._dynamo.config.skip_fsdp_hooks,
    # default True -- which only covers torch.distributed's hooks, not ours).
    #
    # the resharded param never appears in a graph: at trace time p.data is the
    # gathered full tensor, and it is full for every compiled forward, so shapes
    # and guards stay stable across the resize_(0) / re-gather cycle.
    @torch._dynamo.disable
    def _unshard(self, unit: _Unit) -> None:
        for ps in unit.params:
            ps.unshard(self.mesh, self.dim)

    @torch._dynamo.disable
    def _reshard_after_forward(self, unit: _Unit) -> None:
        # p.data stays pointed at the (empty) full tensor: a use before the
        # pre-backward re-gather must crash, not silently read a shard
        for ps in unit.params:
            ps.free_full()

    @torch._dynamo.disable
    def _on_grad_ready(self, ps: _ParamState, unit: _Unit) -> None:
        # AccumulateGrad fired: ps.param.grad holds the full grad for this
        # microbatch (both contributions already merged for the tied
        # weight). reduce-scatter it into the sharded accumulator and drop
        # the full grad immediately -- it is the largest per-param
        # allocation in backward.
        grad = ps.param.grad
        rs = reduce_scatter(grad, self.mesh, self.dim, "avg", scatter_dim=0)
        if ps.grad_shard is None:
            ps.grad_shard = rs
        else:
            ps.grad_shard.add_(rs)
        ps.param.grad = None

        unit.pending -= 1
        if unit.pending == 0:
            unit.pending = unit.n_grads
            # backward is done with this unit: free the full params and
            # point p.data at the sharded master, which is what the
            # optimizer (and clip_grad_norm) must see
            for p_state in unit.params:
                p_state.param.data = p_state.sharded
                p_state.free_full()

    def sync(self) -> None:
        """
        call between backward and clip/optimizer.step, like DDP.sync.
        publishes the accumulated sharded grads as p.grad. v0 collectives
        are synchronous so there is nothing to wait on -- this is where the
        works get waited once the comm stream lands.
        """
        if self.group_size == 1:
            return
        for unit in self._units:
            assert unit.pending == unit.n_grads, (
                f"unit {unit.name}: only {unit.n_grads - unit.pending}/"
                f"{unit.n_grads} grads arrived -- a param got no grad this "
                "backward (frozen or unused param)"
            )
            for ps in unit.params:
                assert ps.full.untyped_storage().size() == 0, (
                    f"unit {unit.name}: full param storage still allocated at sync"
                )
                if ps.param.requires_grad:
                    assert ps.grad_shard is not None, (
                        f"unit {unit.name}: no reduced grad to publish"
                    )
                    ps.param.grad = ps.grad_shard
                    ps.grad_shard = None
