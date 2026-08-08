"""
synth pipeline tests -- no network, no gpus. a stub client stands in for
SynthClient (same generate_text / generate_json interface), so this covers
the orchestration: taxonomy plumbing, judge/refine routing, dedup, sharded
writes, and resume via state.json.

uv run tests/synth.py
"""

import json
import shutil
import tempfile

from pathlib import Path

from pluggy.synth.dedup import Deduper
from pluggy.synth.pipeline import run_pipeline
from pluggy.synth.writer import ShardWriter


class StubClient:
    """deterministic fake llm keyed off prompt shape."""

    def __init__(self, judge_score=8, refuse_topics=()):
        self.judge_score = judge_score
        self.refuse_topics = refuse_topics
        self.calls = {"taxonomy": 0, "generate": 0, "judge": 0, "refine": 0}

    def generate_json(self, prompt, schema, system=None, max_tokens=4096):
        if "Propose" in prompt and "subtopics" in prompt:
            self.calls["taxonomy"] += 1
            return {"subtopics": [f"subtopic {i}" for i in range(3)]}
        if "Score this document" in prompt:
            self.calls["judge"] += 1
            return {"score": self.judge_score, "weaknesses": "none"}
        raise AssertionError(f"unexpected json prompt: {prompt[:60]}")

    def generate_text(self, prompt, system=None, max_tokens=4096):
        if "Rewrite the document" in prompt:
            self.calls["refine"] += 1
            return "refined " + "content " * 40
        self.calls["generate"] += 1
        for t in self.refuse_topics:
            if t in prompt:
                return None
        # unique per prompt so dedup keeps everything
        return f"doc[{hash(prompt) % 10**8}] " + "filler words " * 40


def base_cfg(out_dir):
    return {
        "seed": 0,
        "seed_domains": ["math", "history"],
        "taxonomy": {"subtopics_per_domain": 3},
        "generation": {"docs_per_topic": 2, "styles": ["textbook", "qa"],
                       "max_tokens": 512, "concurrency": 4},
        "quality": {"enabled": True, "min_score": 7, "refine_min": 5,
                    "max_refine_rounds": 1},
        "dedup": {"ngram": 5, "jaccard_threshold": 0.8},
        "output": {"dir": str(out_dir), "shard_docs": 4},
    }


def test_dedup():
    d = Deduper(ngram=3, threshold=0.8)
    a = "the quick brown fox jumps over the lazy dog again and again"
    assert not d.is_duplicate(a)
    assert d.is_duplicate(a)                       # exact
    assert d.is_duplicate(a + " indeed")           # near
    assert not d.is_duplicate("completely different text about training language models at scale")
    print("dedup: ok")


def test_writer(tmp):
    out = tmp / "writer"
    w = ShardWriter(out, shard_docs=2)
    for i in range(5):
        w.add({"text": f"doc {i}"})
    w.close()
    shards = sorted(out.glob("shard-*.jsonl"))
    assert len(shards) == 3, shards                # 2 + 2 + 1
    assert not list(out.glob("*.tmp"))
    rows = [json.loads(line) for s in shards for line in open(s)]
    assert [r["text"] for r in rows] == [f"doc {i}" for i in range(5)]
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["num_shards"] == 3
    # resume continues numbering
    w2 = ShardWriter(out, shard_docs=2)
    assert w2.shard_idx == 3
    print("writer: ok")


def test_pipeline_end_to_end(tmp):
    out = tmp / "run1"
    client = StubClient(judge_score=8)
    stats = run_pipeline(base_cfg(out), client=client)
    # 2 domains x 3 subtopics x 2 docs = 12 jobs, all pass at score 8
    assert stats["kept"] == 12, stats
    assert client.calls["taxonomy"] == 2
    assert client.calls["generate"] == 12
    assert client.calls["judge"] == 12
    assert client.calls["refine"] == 0
    rows = [json.loads(line) for s in sorted(out.glob("shard-*.jsonl"))
            for line in open(s)]
    assert len(rows) == 12
    assert all(r["text"] and r["score"] == 8 for r in rows)
    state = json.load(open(out / "state.json"))
    assert len(state["done"]) == 12
    # resume: nothing left to do, no new llm calls
    client2 = StubClient()
    stats2 = run_pipeline(base_cfg(out), client=client2)
    assert stats2["kept"] == 0
    assert client2.calls["generate"] == 0
    assert client2.calls["taxonomy"] == 0          # taxonomy pinned in state
    print("pipeline end-to-end + resume: ok")


def test_refine_band(tmp):
    # score 6 sits in [refine_min, min_score): every doc takes one refine
    # round, re-judges at 6 again, and drops (max_refine_rounds=1)
    out = tmp / "run2"
    client = StubClient(judge_score=6)
    stats = run_pipeline(base_cfg(out), client=client)
    assert stats["kept"] == 0 and stats["dropped"] == 12, stats
    assert client.calls["refine"] == 12
    print("refine band: ok")


def test_refusal_skips(tmp):
    out = tmp / "run3"
    client = StubClient(refuse_topics=("subtopic 1",))
    stats = run_pipeline(base_cfg(out), client=client)
    # 2 domains x 1 refused subtopic x 2 docs = 4 dropped
    assert stats["dropped"] == 4 and stats["kept"] == 8, stats
    print("refusal skip: ok")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="pluggy_synth_test_"))
    try:
        test_dedup()
        test_writer(tmp)
        test_pipeline_end_to_end(tmp)
        test_refine_band(tmp)
        test_refusal_skips(tmp)
        print("all synth tests passed")
    finally:
        shutil.rmtree(tmp)
