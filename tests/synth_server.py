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

    httpd.shutdown()
    print("all synth server tests passed")


if __name__ == "__main__":
    main()
