"""
sharded jsonl output. each row is {"text": ..., "domain": ..., "topic": ...,
"style": ..., "score": ...} -- the dataloader only needs "text"; the rest is
provenance the packer's remove_columns drops for free.

shards are written to a tmp name and renamed on completion (same torn-file
fence as the checkpointer), so a killed run never leaves a half shard that a
resume would double-count. resume continues shard numbering after the last
complete shard.
"""

import json

from pathlib import Path


class ShardWriter:
    def __init__(self, out_dir: Path, shard_docs: int = 1000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self.buffer = []
        self.total = 0
        existing = sorted(self.out_dir.glob("shard-*.jsonl"))
        self.shard_idx = len(existing)

    def add(self, row: dict):
        self.buffer.append(row)
        self.total += 1
        if len(self.buffer) >= self.shard_docs:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        path = self.out_dir / f"shard-{self.shard_idx:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for row in self.buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.rename(path)
        self.shard_idx += 1
        self.buffer = []

    def close(self, manifest_extra: dict | None = None):
        self._flush()
        manifest = {
            "num_shards": self.shard_idx,
            "docs_written_this_run": self.total,
            **(manifest_extra or {}),
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
