"""
shape checks for pluggy/optimizer/scheduler.py.

both schedulers share the warmup convention ((step + 1) / warmup_steps, so
the last warmup step lands exactly on base_lr); wsd then holds flat and
decays linearly to 0, cosine follows a half-cosine down to
base_lr * min_lr_ratio and holds the floor past total_steps.

also covers the resume path: a scheduler restored via state_dict (or with
last_epoch set manually, the resume_scheduler=false path) must produce the
same lr sequence as one that never stopped.

cpu-only, no gpu needed:
    uv run tests/scheduler.py
"""

import math

import torch

from pluggy.optimizer.builder import build_scheduler
from pluggy.optimizer.scheduler import CosineAnnealingScheduler, WarmupStableDecaySchedulder

TOTAL, WARMUP_RATIO = 100, 0.1
WARMUP = round(TOTAL * WARMUP_RATIO)
BASE_LR = 1e-3


def _build(scheduler_cls, **kwargs):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    return scheduler_cls(optimizer, **kwargs)


def _lr_curve(scheduler, n: int) -> list[float]:
    # get_last_lr()[0] at last_epoch = i is the lr step i trains with; the
    # trainer calls scheduler.step() once per train step, same as here
    lrs = []
    for _ in range(n):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


def check_wsd() -> None:
    decay_ratio = 0.2
    decay_start = round(TOTAL * (1 - decay_ratio))
    lrs = _lr_curve(
        _build(WarmupStableDecaySchedulder, total_steps=TOTAL, warmup_ratio=WARMUP_RATIO, decay_ratio=decay_ratio),
        TOTAL + 5,
    )

    assert math.isclose(lrs[0], BASE_LR / WARMUP), f"first warmup step: {lrs[0]}"
    assert math.isclose(lrs[WARMUP - 1], BASE_LR), "last warmup step must hit base_lr"
    assert all(lrs[i] < lrs[i + 1] for i in range(WARMUP - 1)), "warmup must increase"
    assert all(math.isclose(lr, BASE_LR) for lr in lrs[WARMUP:decay_start]), "stable phase must hold base_lr"
    assert all(lrs[i] > lrs[i + 1] for i in range(decay_start, TOTAL - 1)), "decay must decrease"
    assert all(lr == 0.0 for lr in lrs[TOTAL:]), "past total_steps wsd clamps to 0"


def check_cosine() -> None:
    min_ratio = 0.1
    lrs = _lr_curve(
        _build(CosineAnnealingScheduler, total_steps=TOTAL, warmup_ratio=WARMUP_RATIO, min_lr_ratio=min_ratio),
        TOTAL + 20,
    )

    assert math.isclose(lrs[0], BASE_LR / WARMUP), f"first warmup step: {lrs[0]}"
    assert math.isclose(lrs[WARMUP - 1], BASE_LR), "last warmup step must hit base_lr"
    assert math.isclose(lrs[WARMUP], BASE_LR), "cosine must start at exactly base_lr"
    assert all(lrs[i] > lrs[i + 1] for i in range(WARMUP, TOTAL - 1)), "cosine must decrease"
    # half way down the cosine: scale = min + (1 - min) / 2
    mid = WARMUP + (TOTAL - WARMUP) // 2
    assert math.isclose(lrs[mid], BASE_LR * (min_ratio + (1 - min_ratio) / 2)), f"midpoint: {lrs[mid]}"
    # past total_steps the floor holds -- the clamp keeps a "continue" run
    # from riding the cosine back up
    assert all(math.isclose(lr, BASE_LR * min_ratio) for lr in lrs[TOTAL:]), "must hold the floor past total_steps"


def check_builder() -> None:
    # the config-dict path the trainer takes; cosine used to raise here
    for cfg in (
        {"type": "wsd", "total_steps": TOTAL, "warmup_ratio": 0.1, "decay_ratio": 0.2},
        {"type": "cosine", "total_steps": TOTAL, "warmup_ratio": 0.1, "min_lr_ratio": 0.05},
    ):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
        scheduler = build_scheduler(optimizer, cfg)
        assert scheduler.get_last_lr()[0] > 0


def check_resume() -> None:
    for cls, kwargs in (
        (WarmupStableDecaySchedulder, {"total_steps": TOTAL, "warmup_ratio": 0.1, "decay_ratio": 0.2}),
        (CosineAnnealingScheduler, {"total_steps": TOTAL, "warmup_ratio": 0.1, "min_lr_ratio": 0.1}),
    ):
        full = _lr_curve(_build(cls, **kwargs), TOTAL)

        # state_dict roundtrip (resume_scheduler=true): run 37 steps, save,
        # restore into a fresh scheduler, and the tail must match exactly
        stop = 37
        first = _build(cls, **kwargs)
        _lr_curve(first, stop)
        resumed = _build(cls, **kwargs)
        resumed.load_state_dict(first.state_dict())
        assert _lr_curve(resumed, TOTAL - stop) == full[stop:], f"{cls.__name__} state_dict resume diverged"

        # manual last_epoch (the resume_scheduler=false path in the trainer).
        # setting last_epoch does not recompute the cached lr -- in the real
        # trainer the first resumed step trains on the lr restored by
        # load_optimizer -- so the curve is only rejoined from the next
        # step() onward, and that is what must match
        fresh = _build(cls, **kwargs)
        fresh.last_epoch = stop
        fresh.step()
        assert _lr_curve(fresh, TOTAL - stop - 1) == full[stop + 1:], f"{cls.__name__} last_epoch resume diverged"


def main() -> None:
    check_wsd()
    check_cosine()
    check_builder()
    check_resume()
    print("all scheduler checks passed")


if __name__ == "__main__":
    main()
