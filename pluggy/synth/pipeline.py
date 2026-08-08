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
from pluggy.synth.grounded import GROUNDED_MODES, generate_grounded, load_chunks
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


def _job(client, job, gen_cfg, quality_cfg):
    """
    generate -> judge -> refine for one doc, topic or grounded kind.
    grounded jobs hand the source chunk to the judge for faithfulness.
    returns (job, doc, score).
    """
    max_tokens = gen_cfg.get("max_tokens", 4096)
    if job["kind"] == "topic":
        doc = generate_doc(client, job["topic"], job["style"], job["variant"], max_tokens)
        source = None
    else:
        doc = generate_grounded(client, job["chunk"], job["mode"], job["variant"], max_tokens)
        source = job["chunk"]
    if doc is None:
        return job, None, None
    if quality_cfg.get("enabled", True):
        doc, score = evaluate(client, doc, quality_cfg, max_tokens, source=source)
        return job, doc, score
    return job, doc, None


def _row(job, doc, score, grounding_cfg):
    """provenance columns differ by kind; the dataloader only reads 'text'."""
    if job["kind"] == "topic":
        return {"text": doc, "domain": job["topic"]["domain"],
                "topic": job["topic"]["topic"], "style": job["style"],
                "score": score}
    return {"text": doc, "source": grounding_cfg["dir"],
            "chunk_id": job["chunk_id"], "mode": job["mode"], "score": score}


def run_pipeline(cfg: dict, client=None):
    if client is None:
        client = build_client(cfg)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)

    grounding_cfg = cfg.get("grounding")
    assert cfg.get("seed_domains") or grounding_cfg, (
        "config produces no work: set seed_domains (topic mode), grounding "
        "(uploaded-data mode), or both"
    )

    # taxonomy is generated once and pinned in state so topic ids are stable
    if state["topics"] is None:
        domains = cfg.get("seed_domains") or []
        tax_cfg = cfg.get("taxonomy", {})
        state["topics"] = build_taxonomy(
            client, domains, tax_cfg.get("subtopics_per_domain", 10),
        ) if domains else []
        _save_state(state_path, state)
    topics = state["topics"]
    done = set(state["done"])

    gen_cfg = cfg.get("generation", {})
    quality_cfg = cfg.get("quality", {})
    styles = gen_cfg.get("styles", list(STYLES))
    docs_per_topic = gen_cfg.get("docs_per_topic", 2)

    jobs = []
    for topic in topics:
        for k in range(docs_per_topic):
            style = styles[k % len(styles)]
            key = f"{topic['id']}:{style}:{k}"
            if key not in done:
                jobs.append({"kind": "topic", "key": key, "topic": topic,
                             "style": style, "variant": k})

    num_chunks = 0
    if grounding_cfg:
        chunks = load_chunks(grounding_cfg["dir"],
                             grounding_cfg.get("chunk_words", 600))
        num_chunks = len(chunks)
        # grounded job keys are chunk indices, so the chunking must not move
        # under a resume -- refuse instead of silently regrounding on the
        # wrong chunks. new/changed uploads want a fresh output dir.
        if state.get("num_chunks") is None:
            state["num_chunks"] = num_chunks
            _save_state(state_path, state)
        assert state["num_chunks"] == num_chunks, (
            f"resume with changed uploads is unsupported: state has "
            f"{state['num_chunks']} chunks, source now yields {num_chunks}. "
            f"use a fresh output dir for the new data"
        )
        modes = grounding_cfg.get("modes") or list(GROUNDED_MODES)
        gens_per_chunk = grounding_cfg.get("gens_per_chunk", 1)
        for ci, chunk in enumerate(chunks):
            for k in range(gens_per_chunk):
                mode = modes[k % len(modes)]
                key = f"g{ci}:{mode}:{k}"
                if key not in done:
                    jobs.append({"kind": "grounded", "key": key, "chunk": chunk,
                                 "chunk_id": ci, "mode": mode, "variant": k})

    # shuffle so partial runs cover topics/chunks evenly instead of front-loading
    random.Random(cfg.get("seed", 0)).shuffle(jobs)
    print(f"synth: {len(topics)} topics + {num_chunks} grounded chunks, "
          f"{len(jobs)} docs to generate ({len(done)} already done)")

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
            pool.submit(_job, client, job, gen_cfg, quality_cfg)
            for job in jobs
        ]
        for fut in as_completed(futures):
            job, doc, score = fut.result()
            if doc is None:
                dropped += 1
            elif deduper.is_duplicate(doc):
                dupes += 1
            else:
                writer.add(_row(job, doc, score, grounding_cfg))
                kept += 1
            pending.append(job["key"])
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
