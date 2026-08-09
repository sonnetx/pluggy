"""
synth frontend api tests -- localhost only, no llm, no training run actually
launched. spins the stdlib server up on an ephemeral port in a thread and
exercises the config save/load/validate endpoints plus status/meta, for both
the generation page (/api/*) and the training page (/api/train/*).

the train half is where validation earns its keep: those configs are unpacked
straight into constructors, so a typo'd key would otherwise surface as a
TypeError a minute into a launch.

uv run tests/synth_server.py
"""

import json
import os
import tempfile
import threading
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer
from pathlib import Path

from pluggy.synth import server as srv


def request(base, path, body=None):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def upload(base, dataset, filename, data: bytes):
    req = urllib.request.Request(
        f"{base}/api/upload?dataset={dataset}&filename={filename}",
        data=data, method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def good_config():
    return {
        "model": "claude-opus-5",
        "seed_domains": ["math"],
        "taxonomy": {"subtopics_per_domain": 2},
        "generation": {"docs_per_topic": 1, "styles": ["textbook"],
                       "max_tokens": 512, "concurrency": 2},
        "quality": {"enabled": True, "min_score": 7, "refine_min": 5,
                    "max_refine_rounds": 1},
        "dedup": {"ngram": 13, "jaccard_threshold": 0.8},
        "output": {"dir": "data/tiny", "shard_docs": 10},
    }


def train_config():
    """what the /train page builds in expert mode, at the smallest size."""
    return {
        "model": {"name": "qwen3_dense", "config": {
            "num_layers": 12, "num_heads": 12, "num_kv_heads": 4, "emb_dim": 768,
            "head_dim": 64, "vocab_size": 151936, "ffn_dim": 2048}},
        "mesh": {"dp": 1},
        "parallelism": "ddp",
        "ddp": {"bucket_mb": 25},
        "compile": True,
        "optimizer": {
            "name": "adamw",
            "config": {"lr": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "fused": True},
            "max_norm": 1.0,
            "scheduler": {"type": "wsd", "total_steps": 50, "warmup_ratio": 0.01,
                          "decay_ratio": 0.1}},
        "objective": {"name": "AR", "config": {
            "ignore_index": -100, "fused_linear_ce": True, "ce_chunk_size": 4096}},
        "data": {"name": "OptimalScale/ClimbMix", "split": "train",
                 "tokenizer": "Qwen/Qwen3-0.6B", "eos_token": "<|endoftext|>",
                 "pad_token": "<|fim_pad|>", "text_field": "text", "shuffle_buffer": 100,
                 "seed": 42, "num_workers": 2, "global_batch_size": 8,
                 "micro_batch_size": 2, "seq_len": 1024, "pack_batch": 100},
        "checkpointing": {"save_steps": 10000, "resume": None},
        "wandb": {"enabled": False, "project": "pluggy"},
        "seed": 0,
        "run_name": "test_run",
    }


def train_tests(base):
    # the train page serves, and meta reports gpus without importing torch
    with urllib.request.urlopen(base + "/train") as resp:
        assert resp.status == 200 and b"Launch run" in resp.read()
    code, meta = request(base, "/api/train/meta")
    assert code == 200 and meta["configs"] == [] and meta["gpu_count"] == len(meta["gpus"])

    # a good config saves, lists separately from the synth configs, loads back
    code, out = request(base, "/api/train/config", {"name": "tiny_train",
                                                    "config": train_config()})
    assert code == 200, out
    # the two config lists are disjoint: same directory, told apart by shape
    code, meta = request(base, "/api/train/meta")
    assert meta["configs"] == ["tiny_train"], meta
    code, meta = request(base, "/api/meta")
    assert "tiny_train" not in meta["configs"], "a train config listed as a synth config"
    code, loaded = request(base, "/api/train/config/tiny_train")
    assert code == 200 and loaded == train_config()

    # the checks worth having: each of these dies deep inside a launch otherwise
    def broken(mutate, why):
        cfg = train_config()
        mutate(cfg)
        code, out = request(base, "/api/train/config", {"name": "bad", "config": cfg})
        assert code == 400, f"should have been rejected: {why}"
        return out["error"]

    broken(lambda c: c.pop("mesh"), "no mesh block")
    broken(lambda c: c["model"].update(name="nope"), "unregistered model")
    broken(lambda c: c["model"]["config"].pop("num_layers"), "missing constructor kwarg")
    broken(lambda c: c["model"]["config"].update(num_layer=12), "typo'd constructor kwarg")
    broken(lambda c: c["data"].pop("pad_token"), "no pad token")
    broken(lambda c: c["data"].update(global_batch_size=7), "batch not divisible by mbs x dp")
    broken(lambda c: c["optimizer"]["scheduler"].update(type="cosine"),
           "cosine scheduler takes no decay_ratio")
    broken(lambda c: c["optimizer"]["scheduler"].update(total_steps=0), "no steps")
    # wandb.init is rank-0-only work, so a missing wandb used to wedge the run
    # rather than fail it; the config must not get as far as a launch
    if not srv._has_wandb():
        err = broken(lambda c: c["wandb"].update(enabled=True), "wandb not installed")
        assert "extra wandb" in err, err

    # the two run slots are independent, and neither starts anything here
    code, status = request(base, "/api/train/status")
    assert code == 200 and status["running"] is False and status["metrics"] == []
    code, _ = request(base, "/api/train/stop", {})
    assert code == 409, "stopping with no run is a 409"
    code, _ = request(base, "/api/train/run", {"name": "missing"})
    assert code == 404
    code, out = request(base, "/api/train/run", {"name": "tiny_train", "steps": 0})
    assert code == 400 and "steps" in out["error"], out
    # fsdp2 would have the checkpointer write one rank's shard as the model
    fsdp = {**train_config(), "parallelism": "fsdp2"}
    code, out = request(base, "/api/train/config", {"name": "fsdp", "config": fsdp})
    assert code == 200, out
    code, out = request(base, "/api/train/run", {"name": "fsdp"})
    assert code == 400 and "benchmark" in out["error"], out

    # writing a custom architecture: guards only, no api call and nothing
    # spawned (the real thing needs XAI_API_KEY and takes minutes)
    code, out = request(base, "/api/model/write", {"name": "bad-name",
                                                   "description": "x" * 40})
    assert code == 400 and "module name" in out["error"], out
    code, out = request(base, "/api/model/write", {"name": "ok_name", "description": "too short"})
    assert code == 400 and "describe" in out["error"], out
    if "XAI_API_KEY" not in os.environ:
        code, out = request(base, "/api/model/write",
                            {"name": "ok_name", "description": "a dense model " * 5})
        assert code == 409 and "XAI_API_KEY" in out["error"], out
    code, status = request(base, "/api/model/status")
    assert code == 200 and status["running"] is False and status["result"] is None

    # the registry keys the page offers are read out of builder.py's source,
    # so the page load never pays for a torch import
    assert "qwen3_dense" in srv._registered_models()

    # metrics are scraped out of the run log the trainer writes
    log = Path("train_run_tiny_train.log")
    log.write_text(
        "step=0 || loss=11.2340 || gnorm=1.234 || lr=3.00e-04 || tps=54321\n"
        "warning: something unrelated\n"
        "step=1 || loss=10.0000 || gnorm=0.500 || lr=2.99e-04 || tps=55000\n")
    points = srv._train_metrics(log)
    assert [p["step"] for p in points] == [0, 1], points
    assert points[0]["loss"] == 11.234 and points[1]["tps"] == 55000, points
    print("all train server tests passed")


def main():
    tmp = tempfile.mkdtemp(prefix="pluggy_synth_server_test_")
    os.chdir(tmp)  # server reads/writes configs/ relative to cwd

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    # index page serves
    with urllib.request.urlopen(base + "/") as resp:
        assert resp.status == 200
        assert b"pluggy" in resp.read()

    # meta lists styles, no configs yet
    code, meta = request(base, "/api/meta")
    assert code == 200 and "textbook" in meta["styles"] and meta["configs"] == []

    # validation rejects bad configs
    for bad, why in [
        ({**good_config(), "seed_domains": []}, "empty domains"),
        ({**good_config(), "generation": {"styles": ["nope"]}}, "unknown style"),
        ({**good_config(), "quality": {"enabled": True, "min_score": 4,
                                       "refine_min": 6}}, "inverted thresholds"),
    ]:
        code, out = request(base, "/api/config", {"name": "bad", "config": bad})
        assert code == 400 and "error" in out, why
    code, out = request(base, "/api/config", {"name": "../evil", "config": good_config()})
    assert code == 400, "path traversal in name must be rejected"

    # save -> listed in meta -> loads back identical
    code, out = request(base, "/api/config", {"name": "tiny", "config": good_config()})
    assert code == 200, out
    code, meta = request(base, "/api/meta")
    assert meta["configs"] == ["tiny"]
    code, loaded = request(base, "/api/config/tiny")
    assert code == 200 and loaded == good_config()

    # status is idle, stop with no run is a 409, run with missing config 404
    code, status = request(base, "/api/status")
    assert code == 200 and status["running"] is False
    code, _ = request(base, "/api/stop", {})
    assert code == 409
    code, _ = request(base, "/api/run", {"name": "missing"})
    assert code == 404

    # uploads: every format normalizes to {"text": ...} jsonl
    code, out = upload(base, "acme", "notes.txt", b"plain text doc about widgets")
    assert code == 200 and out["docs"] == 1, out
    rows = [json.dumps({"text": f"row {i}"}) for i in range(3)]
    code, out = upload(base, "acme", "rows.jsonl", "\n".join(rows).encode())
    assert code == 200 and out["docs"] == 3, out
    code, out = upload(base, "acme", "list.json",
                       json.dumps(["a doc", {"text": "another doc"}]).encode())
    assert code == 200 and out["docs"] == 2, out
    code, out = upload(base, "acme", "img.png", b"\x89PNG binary")
    assert code == 400, "binary must be rejected"
    code, out = upload(base, "../evil", "x.txt", b"hi")
    assert code == 400, "path traversal in dataset must be rejected"
    code, meta = request(base, "/api/meta")
    ds = next(u for u in meta["uploads"] if u["name"] == "acme")
    assert ds["docs"] == 6 and ds["words"] > 0, ds
    assert "rephrase" in meta["grounded_modes"]

    # grounding-only config validates; broken grounding configs don't
    grounded = {**good_config(), "grounding": {"dir": "data/uploads/acme"}}
    del grounded["seed_domains"]
    code, out = request(base, "/api/config", {"name": "grounded", "config": grounded})
    assert code == 200, out
    code, out = request(base, "/api/config", {
        "name": "bad", "config": {**grounded, "grounding": {"dir": "d", "modes": ["nope"]}}})
    assert code == 400
    no_work = {k: v for k, v in good_config().items() if k != "seed_domains"}
    code, out = request(base, "/api/config", {"name": "bad", "config": no_work})
    assert code == 400, "config with neither domains nor grounding must be rejected"

    train_tests(base)

    httpd.shutdown()
    print("all synth server tests passed")


if __name__ == "__main__":
    main()
