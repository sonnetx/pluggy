"""
minimal frontend for pluggy: a stdlib-only http server (no new deps) serving
two pages off a shared stylesheet --

    /        synthetic corpus config editor, drives generation runs
    /train   training run composer, drives training runs

    uv run -m pluggy.synth.server            # http://127.0.0.1:8642
    uv run -m pluggy.synth.server --port 9000

run it from the repo root: config files go to ./configs and output dirs are
resolved relative to the cwd, same as the cli. runs launch as a subprocess
of this server (`python -m pluggy.synth.run`), so the server's environment
needs the provider credentials to actually generate: XAI_API_KEY for grok
(no extra install), or the [synth] extra + ANTHROPIC_API_KEY for anthropic.

training runs launch the same way, as `torch.distributed.run` (torchrun) over
the local gpus -- same command the cli would run, so a run started here and a
run started from a shell are the same thing. the child gets its own process
group (so Stop reaches every rank, not just the supervisor), which also means
a run OUTLIVES this server: kill the server mid-run and the ranks keep going,
but the next server has no handle on them and reports idle.

api (all json unless noted):
    GET  /api/meta            styles, grounded modes, saved configs, uploads
    GET  /api/config/<name>   load a saved config
    POST /api/config          {"name": ..., "config": {...}} -> save
    POST /api/upload?dataset=<ds>&filename=<fn>   raw file body -> normalized
                              jsonl under data/uploads/<ds>/ (.txt/.md whole
                              file, .jsonl rows with "text", .json list)
    POST /api/run             {"name": ...} -> start a run
    POST /api/stop            terminate the running subprocess
    GET  /api/status          running?, exit code, log tail, progress

    GET  /api/train/meta          saved train configs, local gpus
    GET  /api/train/config/<name> load a saved train config
    POST /api/train/config        {"name": ..., "config": {...}} -> save
    POST /api/train/run           {"name": ..., "steps": null|N} -> torchrun
    POST /api/train/stop          signal the whole torchrun process group
    GET  /api/train/status        running?, exit code, log tail, step metrics
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pluggy.synth.generate import STYLES
from pluggy.synth.grounded import GROUNDED_MODES

STATIC = Path(__file__).parent / "static"
CONFIG_DIR = Path("configs")
UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_BYTES = 256 * 2**20
LOG_TAIL_CHARS = 4000

# explicit allowlist rather than mapping urls onto the filesystem: the whole
# static dir is three files, and this way there is no traversal to get wrong.
PAGES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/train": "train.html",
    "/train.html": "train.html",
    "/theme.css": "theme.css",
    "/ui.js": "ui.js",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}

_run_lock = threading.Lock()
_run = {"proc": None, "config": None, "log": None}

# training runs get their own slot: a generation run and a training run have
# nothing to do with each other and the two pages poll separately, so sharing
# one slot would have either page's "stop" kill the other page's run.
_train_lock = threading.Lock()
_train = {"proc": None, "config": None, "log": None, "steps": None}

# what the trainer prints per step (pluggy/train/trainer.py):
#   step=0 || loss=11.2340 || gnorm=1.234 || lr=3.00e-04 || tps=54321
STEP_RE = re.compile(
    r"step=(\d+) \|\| loss=(\S+) \|\| gnorm=(\S+) \|\| lr=(\S+) \|\| tps=(\S+)"
)
MAX_METRIC_POINTS = 400
MAX_LOG_SCAN_BYTES = 4 * 2**20


def _is_synth_config(path: Path) -> bool:
    try:
        with open(path) as f:
            return "seed_domains" in json.load(f)
    except (json.JSONDecodeError, OSError):
        return False


def _is_train_config(path: Path) -> bool:
    try:
        with open(path) as f:
            cfg = json.load(f)
        return isinstance(cfg, dict) and "model" in cfg and "optimizer" in cfg
    except (json.JSONDecodeError, OSError):
        return False


def _gpus() -> list[str]:
    """local gpu names via nvidia-smi -- no torch import in the server."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for line in out.stdout.splitlines():
        m = re.match(r"GPU \d+: (.+?) \(UUID", line)
        if m:
            names.append(m.group(1))
    return names


def _free_port() -> int:
    """let the kernel pick the rendezvous port, so back-to-back launches
    don't collide with the previous run's 29500 still in TIME_WAIT."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _normalize_upload(filename: str, body: bytes) -> list[str] | str:
    """file bytes -> list of doc texts, or an error string."""
    suffix = Path(filename).suffix.lower()
    try:
        raw = body.decode("utf-8")
    except UnicodeDecodeError:
        return f"{filename}: not utf-8 text (pdf and binary formats aren't supported yet)"
    if suffix in (".txt", ".md"):
        docs = [raw]
    elif suffix == ".jsonl":
        try:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            docs = [r if isinstance(r, str) else r["text"] for r in rows]
        except (json.JSONDecodeError, KeyError, TypeError):
            return f"{filename}: jsonl rows must be strings or objects with a 'text' field"
    elif suffix == ".json":
        try:
            rows = json.loads(raw)
            assert isinstance(rows, list)
            docs = [r if isinstance(r, str) else r["text"] for r in rows]
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            return f"{filename}: json must be a list of strings or of objects with a 'text' field"
    else:
        return f"{filename}: unsupported extension {suffix} (use .txt, .md, .jsonl, or .json)"
    docs = [d.strip() for d in docs if isinstance(d, str) and d.strip()]
    if not docs:
        return f"{filename}: no non-empty documents found"
    return docs


def _list_uploads() -> list[dict]:
    out = []
    if not UPLOAD_DIR.exists():
        return out
    for ds in sorted(p for p in UPLOAD_DIR.iterdir() if p.is_dir()):
        docs = words = 0
        for fp in ds.glob("*.jsonl"):
            with open(fp) as f:
                for line in f:
                    docs += 1
                    words += len(json.loads(line)["text"].split())
        out.append({"name": ds.name, "docs": docs, "words": words})
    return out


def _validate(cfg: dict) -> str | None:
    """returns an error string or None. keep in sync with pipeline.py."""
    if cfg.get("provider") not in (None, "grok", "anthropic"):
        return "provider must be 'grok' or 'anthropic'"
    temp = cfg.get("generation", {}).get("temperature")
    if temp is not None and not (isinstance(temp, (int, float)) and 0 <= temp <= 2):
        return "generation.temperature must be a number in [0, 2]"
    domains = cfg.get("seed_domains")
    grounding = cfg.get("grounding")
    if not domains and not grounding:
        return "set seed_domains (topic mode), grounding (uploaded data), or both"
    if domains is not None and (
            not isinstance(domains, list) or
            not all(isinstance(d, str) and d.strip() for d in domains)):
        return "seed_domains must be a list of strings"
    if grounding is not None:
        if not isinstance(grounding.get("dir"), str):
            return "grounding.dir is required (e.g. data/uploads/<dataset>)"
        unknown = [m for m in grounding.get("modes", []) if m not in GROUNDED_MODES]
        if unknown:
            return f"unknown grounded modes: {unknown}"
        if grounding.get("chunk_words", 600) < 50:
            return "grounding.chunk_words must be at least 50"
    if not isinstance(cfg.get("output", {}).get("dir"), str):
        return "output.dir is required"
    styles = cfg.get("generation", {}).get("styles", [])
    unknown = [s for s in styles if s not in STYLES]
    if unknown:
        return f"unknown styles: {unknown}"
    q = cfg.get("quality", {})
    if q.get("enabled", True) and not (
            1 <= q.get("refine_min", 5) <= q.get("min_score", 7) <= 10):
        return "quality thresholds must satisfy 1 <= refine_min <= min_score <= 10"
    return None


def _progress(config_path: Path) -> dict:
    """best-effort progress from the run's output dir (state + shards)."""
    try:
        with open(config_path) as f:
            out_dir = Path(json.load(f)["output"]["dir"])
    except (OSError, json.JSONDecodeError, KeyError):
        return {}
    prog = {"shards": len(list(out_dir.glob("shard-*.jsonl")))}
    state_path = out_dir / "state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            prog["topics"] = len(state["topics"] or [])
            prog["jobs_done"] = len(state["done"])
        except (json.JSONDecodeError, KeyError):
            pass
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prog["manifest"] = json.load(f)
        except json.JSONDecodeError:
            pass
    return prog


def _kwarg_error(what: str, fn, cfg: dict) -> str | None:
    """
    check a config block against the callable it is unpacked into.

    the trainer unpacks these dicts straight into constructors
    (`build_model` does `MODEL_REGISTRY[name](**config)`), so a typo'd or
    missing key is a TypeError ~40s into a launch, after the tokenizer
    download and model build. the signature IS the schema -- check it here
    instead, while the user is still looking at the form.
    """
    import inspect

    params = inspect.signature(fn).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        unknown = sorted(set(cfg) - set(params))
        if unknown:
            return f"{what}: unknown key(s) {unknown}"
    # "optimizer" is the scheduler ctor's first arg, supplied by the trainer
    # rather than by the config -- never a missing key
    missing = sorted(
        name for name, p in params.items()
        if name not in ("self", "optimizer") and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in cfg
    )
    if missing:
        return f"{what}: missing required key(s) {missing}"
    return None


def _validate_train(cfg: dict) -> str | None:
    """
    returns an error string or None. structural only -- anything that needs
    the tokenizer vocab or the actual dataset (eos/pad ids, text_field) is
    left to the trainer, which reports it in the run log.

    torch is imported lazily: it costs seconds and a chunk of memory, and the
    generation half of this server never needs it.
    """
    for key in ("model", "mesh", "optimizer", "objective", "data", "run_name"):
        if key not in cfg:
            return f"config is missing the '{key}' block"
    for key in ("model", "mesh", "optimizer", "objective", "data"):
        if not isinstance(cfg[key], dict):
            return f"config block '{key}' must be an object"

    mesh = cfg["mesh"]
    if not mesh or not all(isinstance(v, int) and v >= 1 for v in mesh.values()):
        return "mesh sizes must be integers >= 1"

    parallelism = cfg.get("parallelism", "ddp")
    if parallelism not in ("ddp", "fsdp2"):
        return f"unknown parallelism {parallelism!r}: expected 'ddp' or 'fsdp2'"

    data = cfg["data"]
    for key in ("tokenizer", "eos_token", "pad_token", "seq_len",
                "global_batch_size", "micro_batch_size"):
        if key not in data:
            return f"data.{key} is required"
    for key in ("seq_len", "global_batch_size", "micro_batch_size"):
        if not isinstance(data[key], int) or data[key] < 1:
            return f"data.{key} must be a positive integer"
    if "sources" not in data and "name" not in data and "data_files" not in data:
        return "data needs a 'name' (hub dataset), 'data_files' (local shards), or 'sources'"
    dp = mesh.get("dp", 1)
    seqs = data["micro_batch_size"] * dp
    if data["global_batch_size"] % seqs:
        return (f"global_batch_size {data['global_batch_size']} must be divisible by "
                f"micro_batch_size {data['micro_batch_size']} x dp {dp} = {seqs}")

    from pluggy.models.builder import MODEL_REGISTRY
    from pluggy.objectives.builder import OBJECTIVE_REGISTRY
    from pluggy.optimizer.builder import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

    model_name = cfg["model"].get("name")
    if model_name not in MODEL_REGISTRY or MODEL_REGISTRY[model_name] is None:
        return (f"unknown model {model_name!r}: registered models are "
                f"{sorted(k for k, v in MODEL_REGISTRY.items() if v is not None)}")
    optim_name = cfg["optimizer"].get("name")
    if optim_name not in OPTIMIZER_REGISTRY:
        return f"unknown optimizer {optim_name!r}: {sorted(OPTIMIZER_REGISTRY)}"
    obj_name = cfg["objective"].get("name")
    if obj_name not in OBJECTIVE_REGISTRY:
        return f"unknown objective {obj_name!r}: {sorted(OBJECTIVE_REGISTRY)}"
    sched = dict(cfg["optimizer"].get("scheduler", {}))
    sched_type = sched.pop("type", None)
    if sched_type not in SCHEDULER_REGISTRY:
        return f"unknown scheduler {sched_type!r}: {sorted(SCHEDULER_REGISTRY)}"
    if not isinstance(sched.get("total_steps"), int) or sched["total_steps"] < 1:
        return "optimizer.scheduler.total_steps must be a positive integer"

    return (_kwarg_error(f"model.config ({model_name})",
                         MODEL_REGISTRY[model_name], cfg["model"].get("config", {}))
            or _kwarg_error(f"objective.config ({obj_name})",
                            OBJECTIVE_REGISTRY[obj_name], cfg["objective"].get("config", {}))
            or _kwarg_error(f"optimizer.scheduler ({sched_type})",
                            SCHEDULER_REGISTRY[sched_type], sched))


def _train_metrics(log: Path) -> list[dict]:
    """per-step metrics scraped from the run log, downsampled for plotting."""
    try:
        text = log.read_text(errors="replace")[-MAX_LOG_SCAN_BYTES:]
    except OSError:
        return []
    points = [
        {"step": int(m[1]), "loss": float(m[2]), "gnorm": float(m[3]),
         "lr": float(m[4]), "tps": float(m[5])}
        for m in STEP_RE.finditer(text)
    ]
    if len(points) <= MAX_METRIC_POINTS:
        return points
    # keep the last point: it is the one the status line reads
    stride = len(points) // MAX_METRIC_POINTS + 1
    return points[::stride] + [points[-1]]


class Handler(BaseHTTPRequestHandler):
    # quiet request logging; the terminal is for pipeline output
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, msg: str, code: int = 400):
        self._json({"error": msg}, code)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    @staticmethod
    def _safe_name(name: str) -> str | None:
        name = name.removesuffix(".json")
        return name if re.fullmatch(r"[A-Za-z0-9_\-]+", name) else None

    def do_GET(self):
        if self.path in PAGES:
            name = PAGES[self.path]
            self._send(200, (STATIC / name).read_bytes(),
                       CONTENT_TYPES[Path(name).suffix])
        elif self.path == "/api/meta":
            configs = sorted(
                p.stem for p in CONFIG_DIR.glob("*.json") if _is_synth_config(p)
            ) if CONFIG_DIR.exists() else []
            self._json({"styles": STYLES, "grounded_modes": GROUNDED_MODES,
                        "configs": configs, "uploads": _list_uploads()})
        elif self.path.startswith("/api/config/"):
            name = self._safe_name(self.path.removeprefix("/api/config/"))
            path = CONFIG_DIR / f"{name}.json" if name else None
            if path is None or not path.exists():
                return self._error("config not found", 404)
            with open(path) as f:
                self._json(json.load(f))
        elif self.path == "/api/status":
            with _run_lock:
                proc, log, cfg = _run["proc"], _run["log"], _run["config"]
            status = {"running": False, "exit_code": None, "log_tail": "",
                      "config": str(cfg) if cfg else None, "progress": {}}
            if proc is not None:
                code = proc.poll()
                status["running"] = code is None
                status["exit_code"] = code
            if log is not None and log.exists():
                status["log_tail"] = log.read_text()[-LOG_TAIL_CHARS:]
            if cfg is not None:
                status["progress"] = _progress(cfg)
            self._json(status)
        elif self.path == "/api/train/meta":
            # deliberately does NOT touch pluggy.models: importing torch costs
            # seconds, and this runs on every page load. the registry check
            # happens in _validate_train, on save/launch.
            configs = sorted(
                p.stem for p in CONFIG_DIR.glob("*.json") if _is_train_config(p)
            ) if CONFIG_DIR.exists() else []
            gpus = _gpus()
            self._json({"configs": configs, "gpus": gpus, "gpu_count": len(gpus)})
        elif self.path.startswith("/api/train/config/"):
            name = self._safe_name(self.path.removeprefix("/api/train/config/"))
            path = CONFIG_DIR / f"{name}.json" if name else None
            if path is None or not path.exists():
                return self._error("config not found", 404)
            with open(path) as f:
                self._json(json.load(f))
        elif self.path == "/api/train/status":
            with _train_lock:
                proc, log, cfg = _train["proc"], _train["log"], _train["config"]
                steps = _train["steps"]
            status = {"running": False, "exit_code": None, "log_tail": "",
                      "config": str(cfg) if cfg else None, "steps": steps,
                      "total_steps": None, "metrics": []}
            if cfg is not None and cfg.exists():
                try:
                    with open(cfg) as f:
                        status["total_steps"] = (
                            steps or json.load(f)["optimizer"]["scheduler"]["total_steps"])
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            if proc is not None:
                code = proc.poll()
                status["running"] = code is None
                status["exit_code"] = code
            if log is not None and log.exists():
                status["log_tail"] = log.read_text(errors="replace")[-LOG_TAIL_CHARS:]
                status["metrics"] = _train_metrics(log)
            self._json(status)
        else:
            self._error("not found", 404)

    def do_POST(self):
        # upload takes a raw file body, not json -- handle before _body()
        if self.path.startswith("/api/upload"):
            return self._upload()
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._error("invalid json body")

        if self.path == "/api/config":
            name = self._safe_name(body.get("name", ""))
            if name is None:
                return self._error("name must be [A-Za-z0-9_-]+")
            cfg = body.get("config")
            err = _validate(cfg) if isinstance(cfg, dict) else "config must be an object"
            if err:
                return self._error(err)
            CONFIG_DIR.mkdir(exist_ok=True)
            path = CONFIG_DIR / f"{name}.json"
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._json({"saved": str(path)})

        elif self.path == "/api/run":
            name = self._safe_name(body.get("name", ""))
            path = CONFIG_DIR / f"{name}.json" if name else None
            if path is None or not path.exists():
                return self._error("config not found; save it first", 404)
            with _run_lock:
                if _run["proc"] is not None and _run["proc"].poll() is None:
                    return self._error("a run is already in progress", 409)
                log = Path(f"synth_run_{name}.log")
                logf = open(log, "w")
                _run["proc"] = subprocess.Popen(
                    [sys.executable, "-m", "pluggy.synth.run", "--config", str(path)],
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                logf.close()  # child holds its own handle
                _run["config"], _run["log"] = path, log
            self._json({"started": str(path), "log": str(log)})

        elif self.path == "/api/stop":
            with _run_lock:
                proc = _run["proc"]
                if proc is None or proc.poll() is not None:
                    return self._error("no run in progress", 409)
                proc.terminate()
            self._json({"stopped": True})

        elif self.path == "/api/train/config":
            name = self._safe_name(body.get("name", ""))
            if name is None:
                return self._error("name must be [A-Za-z0-9_-]+")
            cfg = body.get("config")
            err = _validate_train(cfg) if isinstance(cfg, dict) else "config must be an object"
            if err:
                return self._error(err)
            CONFIG_DIR.mkdir(exist_ok=True)
            path = CONFIG_DIR / f"{name}.json"
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._json({"saved": str(path)})

        elif self.path == "/api/train/run":
            return self._train_run(body)

        elif self.path == "/api/train/stop":
            with _train_lock:
                proc = _train["proc"]
                if proc is None or proc.poll() is not None:
                    return self._error("no training run in progress", 409)
                # torchrun is a supervisor with one child per gpu; terminating
                # just the supervisor can orphan workers holding vram, so
                # signal the whole group (Popen set start_new_session, so the
                # group is exactly this run and not the server itself)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            self._json({"stopped": True})

        else:
            self._error("not found", 404)

    def _train_run(self, body: dict):
        name = self._safe_name(body.get("name", ""))
        path = CONFIG_DIR / f"{name}.json" if name else None
        if path is None or not path.exists():
            return self._error("config not found; save it first", 404)
        with open(path) as f:
            cfg = json.load(f)
        err = _validate_train(cfg)
        if err:
            return self._error(err)

        steps = body.get("steps")
        if steps is not None and (not isinstance(steps, int) or steps < 1):
            return self._error("steps must be a positive integer or null")
        if cfg.get("parallelism", "ddp") == "fsdp2" and steps is None:
            return self._error(
                "fsdp2 runs are benchmark-only until sharded checkpointing lands "
                "(the checkpointer would write one rank's shard as the model) -- "
                "use Benchmark, or switch parallelism to ddp"
            )

        # one process per mesh slot: Mesh asserts world_size == prod(mesh)
        nproc = 1
        for size in cfg["mesh"].values():
            nproc *= size
        gpus = _gpus()
        if not gpus:
            return self._error("no cuda gpus visible to the server (nvidia-smi found none)", 409)
        if nproc > len(gpus):
            return self._error(
                f"config wants {nproc} ranks but this machine has {len(gpus)} gpu(s): "
                f"lower mesh.dp", 409)

        with _train_lock:
            if _train["proc"] is not None and _train["proc"].poll() is None:
                return self._error("a training run is already in progress", 409)
            log = Path(f"train_run_{name}.log")
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   f"--nproc_per_node={nproc}", f"--master_port={_free_port()}",
                   "-m", "pluggy.train.train", "--config", str(path)]
            if steps is not None:
                cmd += ["--steps", str(steps)]
            logf = open(log, "w")
            logf.write(" ".join(cmd) + "\n\n")
            logf.flush()
            _train["proc"] = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                # unbuffered, or the step lines sit in the child's stdio buffer
                # and the ui shows nothing for minutes
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                # own process group, so stop can signal every rank at once
                start_new_session=True,
            )
            logf.close()  # child holds its own handle
            _train["config"], _train["log"], _train["steps"] = path, log, steps
        self._json({"started": str(path), "log": str(log), "nproc": nproc,
                    "steps": steps, "cmd": " ".join(cmd)})

    def _upload(self):
        query = parse_qs(urlparse(self.path).query)
        dataset = self._safe_name(query.get("dataset", [""])[0])
        if dataset is None:
            return self._error("dataset must be [A-Za-z0-9_-]+")
        filename = Path(query.get("filename", [""])[0]).name  # strip any path
        if not filename:
            return self._error("filename query param is required")
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return self._error("empty upload")
        if length > MAX_UPLOAD_BYTES:
            return self._error(f"upload over {MAX_UPLOAD_BYTES >> 20} MiB", 413)
        docs = _normalize_upload(filename, self.rfile.read(length))
        if isinstance(docs, str):
            return self._error(docs)
        out_dir = UPLOAD_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(filename).stem) or "upload"
        path = out_dir / f"{stem}.jsonl"
        with open(path, "w") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        self._json({"saved": str(path), "docs": len(docs),
                    "words": sum(len(d.split()) for d in docs),
                    "grounding_dir": str(out_dir)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--host", default="127.0.0.1")  # local tool, keep it local
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"synth frontend: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
