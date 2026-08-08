"""
host->device prefetching for the training loop.

the DataLoader (with num_workers > 0) already prefetches collated CPU batches
into a queue, so next(iter) usually pops a ready batch. what's still serial is
the host->device copy: by default it runs on the compute stream and blocks the
next step. CUDAPrefetcher issues that copy on a side stream so it overlaps the
previous step's kernels, and only makes the compute stream wait right before it
needs the batch.

checkpointing interaction: the prefetcher is always one batch ahead of the
caller, so the loader's live state_dict() counts a batch the trainer never
consumed -- naively saving it makes resume skip that batch. the prefetcher
therefore snapshots the loader's state right before each pull, and
loader_state_dict() hands back the snapshot matching what the caller has
actually consumed. Trainer.checkpoint saves that, not the live loader state.
"""

import torch


class CUDAPrefetcher:
    """
    overlaps the host->device copy of batch N+1 with the compute of batch N.

    the copy runs on a dedicated CUDA stream so it doesn't serialize behind the
    default (compute) stream. right before we hand a batch to the caller we make
    the compute stream wait on the copy stream, and record_stream the tensors so
    the caching allocator doesn't recycle their memory while the async copy is
    still in flight.

    requires the source tensors to be in pinned memory (pin_memory=True on the
    DataLoader) -- otherwise .to(non_blocking=True) silently falls back to a
    blocking copy and nothing overlaps.

    a cpu device skips the stream machinery entirely (plain synchronous
    next + .to); that path exists so the pull-ahead/state-snapshot logic is
    testable without a gpu, not as a training configuration.
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.next_batch: dict[str, torch.Tensor] | None = None
        self._loader_state: dict | None = None
        self._preload()

    def loader_state_dict(self) -> dict:
        """
        the loader state as of the last batch the CALLER consumed, i.e. not
        counting the batch sitting prefetched in next_batch. saving this makes
        resume re-yield the prefetched-but-unconsumed batch instead of
        skipping it.
        """
        return self._loader_state

    def _preload(self) -> None:
        # snapshot BEFORE pulling: this state says "everything handed to the
        # caller so far is consumed, the batch we're about to pull is not"
        self._loader_state = self.loader.state_dict()
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = {k: v.to(self.device) for k, v in batch.items()}
            return
        # queue the copy on the side stream; returns immediately on the host.
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        if self.stream is not None:
            # ensure the copy issued on self.stream is complete before compute uses it.
            torch.cuda.current_stream().wait_stream(self.stream)
            # these tensors were produced on self.stream; tell the allocator the
            # compute stream now owns them so their memory isn't freed too early.
            for v in batch.values():
                v.record_stream(torch.cuda.current_stream())
        # kick off the copy for the *following* batch, then return the current one.
        self._preload()
        return batch
