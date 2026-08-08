"""
orchestrates the full loop: taxonomy -> (generate -> judge -> refine) per
doc, fanned out over a thread pool -> dedup -> sharded jsonl.

concurrency model: llm calls (the slow part) run on worker threads, one
worker per in-flight doc; dedup and the shard writer live on the main
thread only, so neither needs locks.

resume: state.json records the taxonomy and the set of completed job keys,
and is committed only when the docs backing it are on disk (right after a
shard flush), mirroring the checkpointer's fencing. the deduper is rebuilt
from existing shards on startup so a resumed run dedups against prior output.
"""

import json
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pluggy.synth.dedup import Deduper
from pluggy.synth.generate import STYLES, generate_doc
from pluggy.synth.judge import evaluate
from pluggy.synth.taxonomy import build_taxonomy
from pluggy.synth.writer import ShardWriter


def _load_state(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"topics": None, "done": []}


def _save_state(path: Path, state: dict):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(path)


def build_client(cfg: dict):
    """
    pick the provider from cfg["provider"], falling back to inference from
    the model name (grok-* -> xai, everything else -> anthropic). both
    clients expose the same generate_text / generate_json interface.
    """
    model = cfg.get("model", "grok-4")
    provider = cfg.get("provider") or ("grok" if model.startswith("grok") else "anthropic")
    if provider == "grok":
        from pluggy.synth.grok import GrokClient
        return GrokClient(model, temperature=cfg.get("generation", {}).get("temperature"))
    if provider == "anthropic":
        from pluggy.synth.llm import SynthClient
        return SynthClient(model, fallbacks=cfg.get("fallbacks", True))
    raise ValueError(f"unknown provider {provider!r} (expected 'grok' or 'anthropic')")


def _job(client, topic, style, variant, gen_cfg, quality_cfg):
    """generate -> judge -> refine for one doc. returns (key, doc, score)."""
    key = f"{topic['id']}:{style}:{variant}"
    max_tokens = gen_cfg.get("max_tokens", 4096)
    doc = generate_doc(client, topic, style, variant, max_tokens)
    if doc is None:
        return key, None, None
    if quality_cfg.get("enabled", True):
        doc, score = evaluate(client, doc, quality_cfg, max_tokens)
        return key, doc, score
    return key, doc, None


def run_pipeline(cfg: dict, client=None):
    if client is None:
        client = build_client(cfg)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)

    # taxonomy is generated once and pinned in state so topic ids are stable
    if state["topics"] is None:
        tax_cfg = cfg.get("taxonomy", {})
        state["topics"] = build_taxonomy(
            client, cfg["seed_domains"],
            tax_cfg.get("subtopics_per_domain", 10),
        )
        _save_state(state_path, state)
    topics = state["topics"]
    done = set(state["done"])

    gen_cfg = cfg.get("generation", {})
    quality_cfg = cfg.get("quality", {})
    styles = gen_cfg.get("styles", list(STYLES))
    docs_per_topic = gen_cfg.get("docs_per_topic", 2)

    jobs = [
        (topic, styles[k % len(styles)], k)
        for topic in topics
        for k in range(docs_per_topic)
        if f"{topic['id']}:{styles[k % len(styles)]}:{k}" not in done
    ]
    # shuffle so partial runs cover topics evenly instead of front-loading
    random.Random(cfg.get("seed", 0)).shuffle(jobs)
    print(f"synth: {len(topics)} topics, {len(jobs)} docs to generate "
          f"({len(done)} already done)")

    dedup_cfg = cfg.get("dedup", {})
    deduper = Deduper(ngram=dedup_cfg.get("ngram", 13),
                      threshold=dedup_cfg.get("jaccard_threshold", 0.8))
    for shard in sorted(out_dir.glob("shard-*.jsonl")):
        with open(shard) as f:
            for line in f:
                deduper.is_duplicate(json.loads(line)["text"])

    writer = ShardWriter(out_dir, cfg["output"].get("shard_docs", 1000))
    kept = dropped = dupes = 0
    pending = []                      # done keys not yet fenced by a shard flush
    fenced_shard = writer.shard_idx

    def commit_if_flushed(force=False):
        nonlocal fenced_shard, pending
        if force or writer.shard_idx > fenced_shard:
            state["done"] = sorted(done | set(pending))
            done.update(pending)
            pending = []
            fenced_shard = writer.shard_idx
            _save_state(state_path, state)

    with ThreadPoolExecutor(max_workers=gen_cfg.get("concurrency", 8)) as pool:
        futures = [
            pool.submit(_job, client, topic, style, k, gen_cfg, quality_cfg)
            for topic, style, k in jobs
        ]
        jobs_by_key = {f"{t['id']}:{s}:{k}": (t, s) for t, s, k in jobs}
        for fut in as_completed(futures):
            key, doc, score = fut.result()
            topic, style = jobs_by_key[key]
            if doc is None:
                dropped += 1
            elif deduper.is_duplicate(doc):
                dupes += 1
            else:
                writer.add({"text": doc, "domain": topic["domain"],
                            "topic": topic["topic"], "style": style,
                            "score": score})
                kept += 1
            pending.append(key)
            commit_if_flushed()
            n = kept + dropped + dupes
            if n % 25 == 0:
                print(f"synth: {n}/{len(jobs)} jobs, "
                      f"{kept} kept / {dropped} dropped / {dupes} dupes")

    writer.close(manifest_extra={
        "model": cfg.get("model", "grok-4"),
        "kept": kept, "dropped": dropped, "dupes": dupes,
    })
    commit_if_flushed(force=True)
    print(f"synth: done. {kept} kept, {dropped} dropped, {dupes} dupes "
          f"-> {writer.shard_idx} shards in {out_dir}")
    return {"kept": kept, "dropped": dropped, "dupes": dupes}
