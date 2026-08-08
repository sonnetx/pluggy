"""
synth frontend api tests -- localhost only, no llm, no run launched. spins
the stdlib server up on an ephemeral port in a thread and exercises the
config save/load/validate endpoints plus status/meta.

uv run tests/synth_server.py
"""

import json
import os
import tempfile
import threading
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

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

    httpd.shutdown()
    print("all synth server tests passed")


if __name__ == "__main__":
    main()
