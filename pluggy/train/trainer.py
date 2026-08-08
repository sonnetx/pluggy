"""
training orchestrator

lifecycle is split into explicit phases so that moving to
distributed/parallelism later means filling in _parallelize and
_init_weights rather than reordering the constructor:

    build -> parallelize -> init/materialize -> optimizer -> data

for a single gpu run _parallelize is a no-op and world_size == 1.

TODO:
remove the decorators for profiling... just keep that now for
ease of use then move to different profiling methods
"""

import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchdata
from tokenizers import Tokenizer

from pluggy.checkpoint.checkpointer import Checkpointer
from pluggy.core.collective import all_reduce, barrier
from pluggy.core.grad_helper import clip_grad_norm
from pluggy.core.mesh import Mesh
from pluggy.dataloader.builder import build_dataloader
from pluggy.dataloader.prefetcher import CUDAPrefetcher
from pluggy.models.builder import build_model
from pluggy.objectives.builder import build_objective
from pluggy.optimizer.builder import build_optimizer, build_scheduler
from pluggy.parallelism.data_parallel import DDP
from pluggy.utils import record_time, debug_time, get_project_dir


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# allow tf32 for the fp32 matmuls that autocast leaves alone (rope, some
# reductions) and for cudnn. bf16 autocast already covers the big matmuls, so
# this is a small but free win and is orthogonal to distributed.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

PROJECT_DIR = get_project_dir()

def load_json(train_config_path: str) -> dict:
    with open(train_config_path, 'r') as file:
        return json.load(file)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    def __init__(
        self,
        train_config_path: str,
        rank: int,
        local_rank: int,
        world_size: int,
        benchmark: bool = False,
    ):
        # we might want to add some asserts that
        # makes sure that the model and tokenizer
        # vocab sizes are the same
        self.config = load_json(train_config_path)
        self.rank = rank
        self.world_size = world_size
        # benchmark mode (train.py --steps) is a fixed-step tps measurement,
        # not an experiment: no checkpoints, and no wandb run either
        self.benchmark = benchmark
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)  # have the process own this specific GPU
        self.mesh = Mesh(self.config["mesh"])
        self.dp_size = self.mesh.size("dp")

        self.ignore_index = self.config["objective"]["config"]["ignore_index"]
        self.tokenizer = Tokenizer.from_pretrained(self.config["data"]["tokenizer"])

        # same seed on EVERY rank for build/init, so replicas are born
        # identical (DDP's broadcast from rank 0 becomes belt-and-suspenders,
        # and future sharded init slices the same full tensor). before this,
        # nothing seeded torch at all, so no run was reproducible.
        self.seed = self.config.get("seed", 0)
        torch.manual_seed(self.seed)

        self.model = self._build_model()
        self._init_weights(self.model)
        self.model = self._parallelize(self.model)
        self._compile(self.model)

        # after init, fork the streams per rank (roadmap 0.3): keyed on the dp
        # coord, not the global rank, so future tp/cp peers of a dp shard
        # would keep identical draws while dp shards differ. the +1 keeps
        # rank 0's training stream from replaying the init stream. inert
        # today (nothing samples per step), load-bearing the moment dropout
        # or masked diffusion lands.
        torch.manual_seed(self.seed + 1 + self.mesh.coordinate("dp"))

        self.objective = self._build_objective()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        self.num_train_steps = self.config["optimizer"]["scheduler"]["total_steps"]
        self.max_norm = self.config["optimizer"].get("max_norm")

        self.dataloader = self._build_dataloader()
        self.global_batch_size = self.config["data"]["global_batch_size"]
        self.micro_batch_size = self.config["data"]["micro_batch_size"]

        # one backward consumes micro_batch_size seqs on each of the dp ranks
        # simultaneously. identical on every rank (config + mesh only, no
        # rank-local state), which is what keeps the collective count per step
        # in lockstep across ranks.
        seqs_per_backward = self.micro_batch_size * self.dp_size
        assert self.global_batch_size % seqs_per_backward == 0, (
            f"global_batch_size {self.global_batch_size} must be divisible by "
            f"micro_batch_size {self.micro_batch_size} x dp {self.dp_size} "
            f"= {seqs_per_backward}"
        )
        self.microbatches_per_step = self.global_batch_size // seqs_per_backward

        
        # two different names on purpose. the checkpoint dir is keyed on the
        # config name alone so a relaunch lands in the SAME directory and
        # resume="auto" can find it -- a timestamp here would hand every
        # launch a fresh empty dir and silently restart from step 0 (and,
        # under ddp, give each rank its own datetime.now() and its own dir).
        # the timestamp belongs on the wandb display name, where per-launch
        # is exactly what you want.
        time_launched = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.config['run_name']}_{time_launched}"
        self.checkpointer_path = f"{PROJECT_DIR}/checkpoints/{self.config['run_name']}"
        self.checkpointer = Checkpointer(self.checkpointer_path)
        self.resume = self.config["checkpointing"]["resume"]
        self.save_steps = self.config["checkpointing"]["save_steps"]
        # a "continue" run that grows total_steps past the checkpointed
        # schedule wants the freshly-built scheduler's warmup/decay, not the
        # old one -- loading the old scheduler state would clobber
        # total_steps/warmup_ratio/decay_ratio right back to the prior run's
        # values (LRScheduler.load_state_dict is a blind __dict__.update)
        self.resume_scheduler = self.config["checkpointing"].get("resume_scheduler", True)
        # set by _resume when it finds a checkpoint: the wandb run this
        # experiment was already logging to, so we reattach instead of
        # splitting one loss curve across a new run per restart
        self.resume_wandb_id: str | None = None
        self.start_step = self._resume()
        self._setup_wandb()

        # the prefetcher owns the loader iterator (it needs the loader itself
        # to snapshot state for exact-resume checkpointing); has to come after
        # _resume so iteration starts from the restored position. prefetches
        # batch N+1's host->device copy on a side stream while step N
        # computes; see pluggy/dataloader/prefetcher.py.
        self.prefetcher = CUDAPrefetcher(self.dataloader, self.device)


    def _setup_wandb(self) -> None:
        """
        rank 0 only: every rank calling init would open world_size duplicate
        runs racing over the same name. runs after _resume so we can hand
        wandb the run id we recovered from the checkpoint.
        """
        self.wandb_run = None
        wandb_cfg = self.config.get("wandb", {})
        if not wandb_cfg.get("enabled", False) or self.rank != 0 or self.benchmark:
            return
        assert HAS_WANDB, (
            "wandb.enabled is true in the config but wandb isn't installed: "
            "uv sync --extra wandb"
        )

        # the full config doubles as the run's searchable hyperparameters, so
        # two runs can be diffed in the ui instead of by digging up the json
        self.wandb_run = wandb.init(
            project=wandb_cfg.get("project", "pluggy"),
            entity=wandb_cfg.get("entity"),
            name=self.run_name,
            id=self.resume_wandb_id,
            resume="must" if self.resume_wandb_id else None,
            config=self.config,
        )

    def _log_metrics(self, metrics: dict, step: int, tokens_per_step: int, time: float) -> None:
        # `step` is the 1-based count of completed steps, matching what
        # checkpoint() writes. wandb DROPS any log at a step <= the last one
        # recorded, so a resumed run whose numbering disagrees with the
        # pre-crash run loses steps silently -- keep the two conventions equal.
        if self.wandb_run is None:
            return
        # .item() syncs on the device, but the print below already does via
        # __format__ on a cuda tensor, so this costs nothing extra
        self.wandb_run.log(
            {
                "train/loss": metrics["loss"].item(),
                "train/grad_norm": metrics["grad_norm"].item(),
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/tokens": step * tokens_per_step,
                "perf/tps": tokens_per_step / time,
                "perf/step_time": time,
                "perf/peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            step=step,
        )

    def _build_model(self) -> nn.Module:
        # for single gpu right now, when we want to do
        # sharding, we need this to be an empty init
        # where we actually don't init any weights yet
        model_cfg = self.config["model"]
        return build_model(model_cfg["name"], model_cfg["config"])

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """
        ddp for now: replicate along the dp axis, bucketed grad sync
        overlapped with backward. tensor/expert/context parallel + fsdp2
        slot in here later, driven by the same mesh.

        runs before _compile on purpose: the grad hooks sit on params at the
        AccumulateGrad boundary, outside the compiled regions. at dp=1 the
        DDP ctor is a complete no-op (no hooks, no groups touched).
        """
        ddp_cfg = self.config.get("ddp", {})
        self.ddp = DDP(
            model,
            self.mesh,
            dim="dp",
            bucket_mb=ddp_cfg.get("bucket_mb", 25),
        )
        return model

    def _compile(self, model: nn.Module) -> None:
        """
        compile per transformer block rather than the whole model.

        this is the same granularity FSDP2 wants: each block becomes one
        compiled region, so the graph is captured once and reused for all
        identical blocks (fast warmup, no giant whole-model graph), and it
        composes with per-block FSDP wrapping/activation-checkpointing later.
        gated by config so eager stays available for debugging.
        """
        if not self.config.get("compile", True):
            return
        # "default" is the sweet spot here; "max-autotune-no-cudagraphs" buys a
        # few % more at the cost of a much longer warmup (which multiplies once
        # every rank compiles), so it's opt-in via config.
        mode = self.config.get("compile_mode", "default")
        for block in model.blocks:
            block.compile(mode=mode)
        # the final norm otherwise runs eager with a bf16 input vs fp32 weight
        # dtype mismatch (no fused kernel); compiling it is free and keeps the
        # block-level granularity FSDP2 wants.
        model.norm.compile(mode=mode)

    def _init_weights(self, model: nn.Module) -> None:
        # single gpu again, when we init for sharding
        # we won't have any weights
        # model.to_empty(device=self.device)
        # maybe we should make an ABC and require models
        # to have init_weight as a function
        model.to(self.device)
        model.init_weights()
        # norm weights stay fp32 ON PURPOSE. they used to be cast to bf16 here
        # to kill the "bf16 input vs fp32 weight" mismatch warning, but that
        # made them bf16 *parameters*, so AdamW updated them in bf16: bf16
        # spacing at 1.0 is 2^-8 ~ 0.0039 while a step's update is ~lr = 3e-4,
        # so every update rounded back to the same value and the gains sat
        # frozen at their init for the whole run -- silently, since the loss
        # curve only degrades slightly. the mismatch is handled the right way
        # instead: _compile covers the blocks and the final norm, and compile
        # generates a fused kernel that takes the fp32 weight directly.

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optim_cfg = self.config["optimizer"]
        return build_optimizer(model, optim_cfg["name"], optim_cfg["config"])

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        # currently we set last_epoch to -1 by default for a fresh training
        # run, we need to change this when we do resumption
        scheduler_cfg = self.config["optimizer"]["scheduler"]
        return build_scheduler(optimizer, scheduler_cfg)

    def _build_objective(self):
        obj_cfg = self.config["objective"]
        return build_objective(obj_cfg["name"], obj_cfg["config"])
    
    @debug_time
    def _build_dataloader(self) -> torchdata.stateful_dataloader.StatefulDataLoader:
        return build_dataloader(
            self.config["data"], self.tokenizer, self.ignore_index, 
            self.mesh.coordinate("dp"), self.mesh.size("dp")
        )
    
    # @debug_time
    def get_batch_prefetcher(self) -> dict[str, torch.Tensor]:
        # the prefetcher already queued this batch's copy on a side stream last
        # step; this waits for it and kicks off the next one. pin_memory=True on
        # the DataLoader is what makes the async copy actually overlap.
        return next(self.prefetcher)


    @debug_time
    def checkpoint(self, step: int) -> None:
        # model/optimizer/scheduler are replicas under ddp: rank 0 writes the
        # single copy (sharded state is roadmap phase 3). dataloader position
        # is per dp shard, so every dp rank writes its own file.
        if self.rank == 0:
            self.checkpointer.save_model(self.model, step)
            self.checkpointer.save_optimizer(self.optimizer, step)
            self.checkpointer.save_scheduler(self.scheduler, step)
            # written AFTER completing `step`, so _resume returns step + 1.
            # the config snapshot is provenance for init_from and future
            # drift warnings.
            self.checkpointer.save_trainer(
                {
                    "step": step,
                    "config": self.config,
                    # so a resume reattaches to this run instead of opening a
                    # new one and cutting the loss curve in half
                    "wandb_id": self.wandb_run.id if self.wandb_run else None,
                },
                step,
            )
        # rng is per PROCESS: every rank writes its own, so resume restores
        # each rank's stream instead of stamping rank 0's everywhere. this is
        # what keeps resume exact once anything samples per step (dropout,
        # masked diffusion).
        self.checkpointer.save_rng(
            {
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
            },
            step,
            self.rank,
        )
        # the prefetcher's snapshot, NOT self.dataloader.state_dict(): the
        # live state is one prefetched-but-unconsumed batch ahead, and saving
        # it would make resume skip that batch
        self.checkpointer.save_dataloader_state(
            self.prefetcher.loader_state_dict(), step, self.mesh.coordinate("dp")
        )
        # fence: no rank moves on (or exits) while writers are mid-file
        barrier()
        if self.rank == 0:
            self.checkpointer.mark_complete(step)
            
    def _resume(self) -> int:
        if self.resume is None:
            return 0
        if self.resume == "auto":
            step = self.checkpointer.latest()
            if step is None:
                return 0
        elif isinstance(self.resume, int):
            step = self.checkpointer.valid_step(self.resume)
            assert step is not None, "non valid step to resume from"
        else:
            raise ValueError("invalid value for resume, must be either None, 'auto', or valid integer training step")


        state = self.checkpointer.load_trainer(step)
        # a different dp size would silently misread the per-rank dataloader
        # files (subset of shards, wrong global batch) -- refuse instead
        assert state["config"]["mesh"] == self.config["mesh"], (
            f"resume across a mesh change is unsupported: checkpoint has "
            f"{state['config']['mesh']}, config has {self.config['mesh']}"
        )
        self.checkpointer.load_model(self.model, step)
        self.checkpointer.load_optimizer(self.optimizer, step)
        if self.resume_scheduler:
            self.checkpointer.load_scheduler(self.scheduler, step)
        else:
            # keep this run's freshly-built scheduler (its total_steps /
            # warmup_ratio / decay_ratio come from the current config, not
            # the checkpointed run's) but still pick up the curve at the
            # step the prior run left off, matching the numbering
            # load_scheduler would otherwise have restored
            self.scheduler.last_epoch = step
        self.checkpointer.load_dataloader(self.dataloader, step, self.mesh.coordinate("dp"))
        rng = self.checkpointer.load_rng(step, self.rank)
        if rng is None:
            # checkpoints from before per-rank rng files carried rank 0's
            # states in trainer.pt; restoring them everywhere matches the
            # old behavior and stays harmless while nothing samples per step
            rng = {"cpu_rng": state["cpu_rng"], "cuda_rng": state["cuda_rng"]}
        torch.set_rng_state(rng["cpu_rng"])
        torch.cuda.set_rng_state(rng["cuda_rng"], self.device)
        # .get, not [...]: checkpoints written before wandb landed have no
        # such key, and they should still resume
        self.resume_wandb_id = state.get("wandb_id")
        return state["step"]

    def micro_train_step(self) -> torch.Tensor:
        batch = self.get_batch_prefetcher()
        # cast to bf16 so we can take advantage of sdpa
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.objective.compute_loss(self.model, batch) / self.microbatches_per_step

        loss.backward()
        return loss.detach()

    @record_time
    def train_step(self) -> dict[str, torch.Tensor]:
        loss = torch.zeros((), device=self.device)
        self.ddp.requires_sync = False
        for _ in range(self.microbatches_per_step - 1):
            loss += self.micro_train_step()

        # the last microbatch is the one whose hooks launch the bucket
        # all-reduces, over the fully accumulated grads
        self.ddp.requires_sync = True
        loss += self.micro_train_step()
        self.ddp.sync()
        grad_norm = clip_grad_norm(
            self.model.parameters(), self.mesh, self.max_norm, shard_dims=()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        if self.dp_size > 1:
            all_reduce(loss, self.mesh, "dp", "avg")

        return {"loss": loss, "grad_norm": grad_norm}

    @debug_time
    def train_n_step_test(self, n_steps: int) -> None:
        # global_batch_size already spans every dp rank (and, once roadmap 1.3
        # lands, every accumulated microbatch), so tokens/step is just it x seq
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        for step in range(n_steps):
            metrics, time = self.train_step()
            if self.rank == 0:
                print(f"{step=} || loss={metrics['loss']:.4f} || gnorm={metrics['grad_norm']:.3f} "
                    f"|| lr={self.scheduler.get_last_lr()[0]:.2e} || tps={tokens_per_step / time:.0f}")

        if self.rank == 0:
            print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


    def train(self) -> None:
        """
        we currently operate in a step regime, not epoch
        we could switch, but maybe no need
        """
        tokens_per_step = self.config["data"]["seq_len"] * self.global_batch_size
        try:
            for step in range(self.start_step, self.num_train_steps):
                metrics, time = self.train_step()
                if ((step + 1) % self.save_steps == 0 and step != 0) or (step + 1) == self.num_train_steps:
                    self.checkpoint(step + 1)
                if self.rank == 0:
                    self._log_metrics(metrics, step + 1, tokens_per_step, time)
                    print(f"{step=} || loss={metrics['loss']:.4f} || gnorm={metrics['grad_norm']:.3f} "
                        f"|| lr={self.scheduler.get_last_lr()[0]:.2e} || tps={tokens_per_step / time:.0f}")

            if self.rank == 0:
                print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        finally:
            # in a finally so a crash marks the run crashed, instead of
            # leaving it "running" in the ui until wandb times it out
            if self.wandb_run is not None:
                self.wandb_run.finish()

# no __main__ here on purpose: a Trainer can't be built without a process group
# (Mesh asserts on it), so launching this file directly always crashed. the one
# entrypoint is pluggy/train/train.py, which does the rendezvous first --
# including for the single-gpu benchmark path (--steps).

