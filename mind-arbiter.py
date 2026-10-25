#!/usr/bin/env python3
"""Serve a choice of local models from one endpoint, swapping them only while idle.

`mind-router.py` picks *which host* should answer, because every host serves the same
model. This picks *which model* a single host is currently holding, which is a
different problem with a harder constraint: on a unified-memory box the models do not
fit together. Measured on the gfx1151 box (~122 GiB usable pool):

    MiniCPM-V 4.7 Q4_K_M       ~35 GB resident at 131072 ctx
    Qwen3.8-Flash-Next Q2      137 GiB file, SSD-streamed, cache-hungry
    DeepSeek V4.1 Flash Q2     340 GiB file, ~101 GiB planned by auto cache sizing

So MiniCPM and V4.1 can never be co-resident, and an expert-cache sweep on this box
was killed by its own watchdog at 95.67 and 99.67 GiB planned. The useful service is
therefore not "run them all" but "let a client ask for any of them, hold as many as
genuinely fit, and swap the rest at a safe moment".

Three rules follow, and they are the whole design:

  * **Swap only at zero in-flight requests.** Killing a backend mid-generation loses
    a request and its prompt cache. A model being evicted is drained first, and a
    drain that does not finish in time abandons the swap rather than the request.
  * **Keep per-model prompt caches across a swap.** Both engines here can persist KV
    to disk -- ds4-server does it on a policy with `--kv-disk-dir`, llama-server
    needs `--slot-save-path` plus explicit `/slots/{id}?action=save|restore` calls,
    which this process makes around the swap. Without that, a switch back re-prefills
    from scratch, and prefix reuse is worth 19 ms against 1785 ms on ROCm.
  * **Advertise every model even while it is unloaded,** so a client can see its
    choices and pick one when nothing is serving. `GET /v1/models` lists the whole
    registry with a `loaded` flag; asking for an unloaded one loads it.

What this cannot do: make a swap *instant*. Three things have to come back and only
two of them can be saved. KV state persists to disk; model weights reload in seconds,
and not reliably from page cache, because a streaming model reads enough to evict it;
the SSD-streaming expert cache is in-process RAM and is simply lost -- for V4.1 that
measured 70-123 GiB of re-reads. Treat a streaming model as a cold start every time,
which is why one can be marked `opt_in` and kept out of casual routing.

Endpoints:
  GET  /v1/models              every registered model, with `loaded`
  GET  /arbiter/status         registry, residency, in-flight counts, memory headroom
  POST /arbiter/load/<id>      load a model now (evicting as needed)
  POST /arbiter/unload/<id>    drain and stop a model
  POST /arbiter/pin/<id>|none  refuse to evict a model, or clear the pin
  anything else                proxied to the backend that serves the requested model

Binds loopback by default: the thing in front of it (mind-router) is what faces a LAN
and authenticates. Set MIND_ARBITER_KEY to require a key here too.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MIND_ARBITER_PORT", "8010"))
HOST = os.environ.get("MIND_ARBITER_HOST", "127.0.0.1")
REGISTRY_PATH = os.path.expanduser(
    os.environ.get("MIND_ARBITER_MODELS", "~/.config/mind/models.json"))
TIMEOUT = float(os.environ.get("MIND_ARBITER_UPSTREAM_TIMEOUT", "1800"))
PROBE_TTL = float(os.environ.get("MIND_ARBITER_PROBE_TTL", "10"))
# How long a request will wait for someone else's swap to finish before giving up.
# A streaming model can take minutes to come up, so this is generous by default and
# still bounded: a client gets 503 with a reason, never an indefinite hang.
SWAP_WAIT = float(os.environ.get("MIND_ARBITER_SWAP_WAIT", "1200"))
# How long to wait for a model's in-flight requests to finish before abandoning a
# swap. Short by design: if someone is actively generating, the swap is simply not
# safe yet, and failing fast is better than holding the lock for a long generation.
DRAIN_WAIT = float(os.environ.get("MIND_ARBITER_DRAIN_WAIT", "120"))
# Leave this much of the pool unused. The sweep on this box died at MemAvailable < 3
# GiB even though its plan said it fit, so the guard is deliberately not tight.
RESERVE_GIB = float(os.environ.get("MIND_ARBITER_RESERVE_GIB", "8"))


def _mem_available_gib() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0  # unknown: treated as "no headroom claim", never as "plenty"


def _read_key() -> str:
    key = os.environ.get("MIND_ARBITER_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.ds4-lan-key")) as handle:
            return handle.read().strip()
    except OSError:
        return ""


AUTH_KEY = _read_key()


def _backend_key(model: dict) -> str:
    """Resolve a backend's API key without ever putting it in the registry file.

    The registry is checked into a repo, so it names *where* the key lives, not the
    key: either an env var, or a KEY=VALUE file such as the one the MiniCPM unit
    already reads. A missing key is not an error -- a loopback backend that does not
    authenticate ignores the header.
    """
    name = model.get("key_env") or ""
    if name and os.environ.get(name):
        return os.environ[name].strip()
    path = model.get("key_env_file") or ""
    if path and name:
        try:
            with open(os.path.expanduser(path)) as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith(name + "="):
                        return line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            return ""
    return ""


class Model:
    """One registered model and the backend process that serves it."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.id = str(spec["id"])
        self.aliases = [str(a) for a in spec.get("aliases", [])]
        self.base = str(spec["base"]).rstrip("/")
        self.weight_gib = float(spec.get("weight_gib", 0))
        self.ready_timeout = float(spec.get("ready_timeout_s", 300))
        self.engine = str(spec.get("engine", "llama.cpp"))
        self.launcher = str(spec.get("launcher", "exec"))
        self.unit = spec.get("unit")
        self.argv = spec.get("argv")
        self.log = spec.get("log")
        self.slots = int(spec.get("slots", 0))
        self.slot_dir = spec.get("slot_save_path")
        self.opt_in = bool(spec.get("opt_in", False))
        self.external = bool(spec.get("external", False))
        self.proc: subprocess.Popen | None = None
        self.inflight = 0
        self._probe: tuple[bool, float] = (False, 0.0)

    # -- naming ------------------------------------------------------------------
    def matches(self, want: str) -> bool:
        return want == self.id or want in self.aliases

    def key(self) -> str:
        return _backend_key(self.spec)

    def headers(self, extra: dict | None = None) -> dict:
        key = self.key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        if extra:
            headers.update(extra)
        return headers

    # -- liveness ----------------------------------------------------------------
    def healthy(self, force: bool = False) -> bool:
        ok, at = self._probe
        if not force and time.time() - at <= PROBE_TTL:
            return ok
        try:
            req = urllib.request.Request(  # noqa: S310 - our own configured base
                self.base + "/v1/models", headers=self.headers())
            with urllib.request.urlopen(req, timeout=4) as resp:  # noqa: S310
                ok = 200 <= resp.status < 300
        except Exception:
            ok = False
        self._probe = (ok, time.time())
        return ok

    def running(self) -> bool:
        """Is the backend process up, regardless of whether it answers yet?

        Distinct from healthy(): a streaming model can be running and still minutes
        away from serving, and evicting it in that window would waste all of that
        work. Residency decisions use this; routing uses healthy().
        """
        if self.external:
            return self.healthy()
        if self.launcher == "systemd":
            return self._systemctl("is-active", check=False) == 0
        return self.proc is not None and self.proc.poll() is None

    # -- lifecycle ---------------------------------------------------------------
    def _systemctl(self, verb: str, check: bool = True) -> int:
        if not self.unit:
            return 1
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["systemctl", "--user", verb, self.unit],
                capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return 1
        if check and done.returncode != 0:
            LOG(f"{self.id}: systemctl {verb} failed rc={done.returncode} "
                f"{done.stderr.strip()[:200]}")
        return done.returncode

    def start(self) -> bool:
        if self.external:
            return self.healthy(force=True)
        if self.running():
            return True
        LOG(f"{self.id}: starting ({self.launcher})")
        if self.launcher == "systemd":
            if self._systemctl("start") != 0:
                return False
        else:
            if not self.argv:
                LOG(f"{self.id}: no argv and launcher is exec; cannot start")
                return False
            try:
                sink = open(os.path.expanduser(self.log), "ab") if self.log \
                    else subprocess.DEVNULL
            except OSError:
                sink = subprocess.DEVNULL
            try:
                self.proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    [os.path.expanduser(a) for a in self.argv],
                    stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True)
            except OSError as exc:
                LOG(f"{self.id}: exec failed: {exc}")
                return False
        return self.wait_ready()

    def wait_ready(self) -> bool:
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.healthy(force=True):
                LOG(f"{self.id}: ready")
                return True
            # A process that has already exited will never become ready; say so now
            # instead of burning the whole (possibly 40 minute) timeout on it.
            if not self.external and not self.running():
                LOG(f"{self.id}: backend exited before becoming ready")
                return False
            time.sleep(2)
        LOG(f"{self.id}: not ready within {self.ready_timeout:.0f}s")
        return False

    def stop(self) -> None:
        if self.external:
            return
        LOG(f"{self.id}: stopping")
        if self.launcher == "systemd":
            self._systemctl("stop")
        elif self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                LOG(f"{self.id}: did not exit on TERM; killing")
                self.proc.kill()
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            self.proc = None
        self._probe = (False, 0.0)

    # -- KV persistence ----------------------------------------------------------
    def save_kv(self) -> str:
        """Persist this model's prompt cache so a switch back does not re-prefill.

        ds4-server checkpoints on its own once `--kv-disk-dir` is set, so there is
        nothing to call. llama-server exposes per-slot save/restore instead, but only
        when started with `--slot-save-path`; without it the endpoint answers with an
        error, which is reported rather than retried.
        """
        if self.engine != "llama.cpp" or not self.slots:
            return "automatic" if self.engine == "ds4" else "unsupported"
        return self._slots("save")

    def restore_kv(self) -> str:
        if self.engine != "llama.cpp" or not self.slots:
            return "automatic" if self.engine == "ds4" else "unsupported"
        return self._slots("restore")

    def _slots(self, action: str) -> str:
        done, failed = 0, 0
        for slot in range(self.slots):
            body = json.dumps({"filename": f"{self.id}-slot{slot}.bin"}).encode()
            try:
                req = urllib.request.Request(  # noqa: S310 - our own configured base
                    f"{self.base}/slots/{slot}?action={action}", data=body,
                    method="POST",
                    headers=self.headers({"Content-Type": "application/json"}))
                with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                    done += 1 if 200 <= resp.status < 300 else 0
            except Exception:
                failed += 1
        return f"{action} {done}/{self.slots} ok" + (f", {failed} failed" if failed else "")


def _load_registry(path: str) -> list:
    with open(path) as handle:
        doc = json.load(handle)
    specs = doc.get("models") if isinstance(doc, dict) else doc
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{path}: no 'models' list")
    return [Model(s) for s in specs]


_log_lock = threading.Lock()


def LOG(message: str) -> None:  # noqa: N802 - a verb, used like a statement
    with _log_lock:
        print(f"[mind-arbiter] {time.strftime('%H:%M:%S')} {message}", flush=True)


MODELS: list = []
# One lock serialises every residency change, so two requests for two different
# unloaded models cannot both decide to evict the same third one.
_swap_lock = threading.Lock()
_state_lock = threading.Lock()
pinned: str = os.environ.get("MIND_ARBITER_PIN", "none")


def resolve(want: str):
    for model in MODELS:
        if model.matches(want):
            return model
    return None


def default_model():
    for model in MODELS:
        if not model.opt_in:
            return model
    return MODELS[0] if MODELS else None


def _drain(model, timeout: float) -> bool:
    """Wait for a model's in-flight requests to reach zero."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _state_lock:
            if model.inflight <= 0:
                return True
        time.sleep(0.5)
    with _state_lock:
        return model.inflight <= 0


def _resident_others(target):
    return [m for m in MODELS if m is not target and m.running() and not m.external]


def ensure(target) -> tuple[bool, str]:
    """Make `target` the (or a) resident model, evicting only what is necessary.

    Returns (ok, why). `why` is reported in a response header so a slow first request
    is explainable rather than mysterious.
    """
    if target.healthy():
        return True, "resident"

    if not _swap_lock.acquire(timeout=SWAP_WAIT):
        return False, "swap-busy"
    try:
        # Someone else may have loaded it while we waited for the lock.
        if target.healthy():
            return True, "resident-after-wait"
        if target.running():
            return (True, "waited-ready") if target.wait_ready() else (False, "start-timeout")

        others = _resident_others(target)
        headroom = _mem_available_gib() - RESERVE_GIB
        evicted = []
        for other in others:
            if headroom >= target.weight_gib:
                break  # what is already free is enough; leave the rest resident
            if other.id == pinned:
                return False, f"blocked-by-pin:{other.id}"
            if not _drain(other, DRAIN_WAIT):
                # Refusing is the right answer: the alternative is killing someone's
                # generation to satisfy a different client.
                return False, f"busy:{other.id}"
            note = other.save_kv()
            other.stop()
            evicted.append(f"{other.id}({note})")
            # Freeing is not instantaneous; let the pool settle before re-measuring.
            time.sleep(2)
            headroom = _mem_available_gib() - RESERVE_GIB

        if headroom < target.weight_gib:
            return False, (f"insufficient-memory:{headroom:.0f}GiB-free-"
                           f"{target.weight_gib:.0f}GiB-needed")

        if not target.start():
            return False, "start-failed"
        restored = target.restore_kv()
        why = "loaded"
        if evicted:
            why += "-evicted:" + ",".join(evicted)
        LOG(f"{target.id}: {why}; kv {restored}")
        return True, why
    finally:
        _swap_lock.release()


def client_key(headers) -> str:
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("x-api-key") or "").strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mind-arbiter"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return

    def _json(self, code: int, obj, extra: dict | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _authed(self) -> bool:
        if not AUTH_KEY or client_key(self.headers) == AUTH_KEY:
            return True
        self._json(401, {"error": {"type": "authentication_error",
                                   "message": "missing or wrong API key"}})
        return False

    def do_GET(self):  # noqa: N802 - stdlib name
        if not self._authed():
            return
        if self.path == "/arbiter/status":
            with _state_lock:
                inflight = {m.id: m.inflight for m in MODELS}
            return self._json(200, {
                "port": PORT,
                "pinned": pinned,
                "mem_available_gib": round(_mem_available_gib(), 1),
                "reserve_gib": RESERVE_GIB,
                "models": [{
                    "id": m.id, "base": m.base, "engine": m.engine,
                    "weight_gib": m.weight_gib, "opt_in": m.opt_in,
                    "external": m.external, "running": m.running(),
                    "healthy": m.healthy(), "inflight": inflight.get(m.id, 0),
                } for m in MODELS],
            })
        if self.path.rstrip("/") == "/v1/models":
            # Every registered model, loaded or not. This is what lets a client see
            # its options while nothing is serving and then ask for one by name.
            return self._json(200, {"object": "list", "data": [
                {"id": m.id, "object": "model", "owned_by": "local",
                 "loaded": m.healthy(), "opt_in": m.opt_in} for m in MODELS]})
        return self.forward("GET")

    def do_POST(self):  # noqa: N802 - stdlib name
        global pinned
        if not self._authed():
            return
        for verb in ("load", "unload", "pin"):
            prefix = f"/arbiter/{verb}/"
            if not self.path.startswith(prefix):
                continue
            want = self.path[len(prefix):].strip("/")
            if verb == "pin":
                if want not in ("none", "") and resolve(want) is None:
                    return self._json(400, {"error": f"unknown model {want!r}"})
                pinned = resolve(want).id if resolve(want) else "none"
                return self._json(200, {"pinned": pinned})
            model = resolve(want)
            if model is None:
                return self._json(404, {"error": f"unknown model {want!r}",
                                        "known": [m.id for m in MODELS]})
            if verb == "load":
                ok, why = ensure(model)
                return self._json(200 if ok else 503, {"model": model.id, "ok": ok, "why": why})
            if not _drain(model, DRAIN_WAIT):
                return self._json(409, {"model": model.id, "error": "busy",
                                        "inflight": model.inflight})
            note = model.save_kv()
            model.stop()
            return self._json(200, {"model": model.id, "stopped": True, "kv": note})
        return self.forward("POST")

    def forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else None
        ctype = self.headers.get("Content-Type", "application/json")

        body: dict = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    body = parsed
            except ValueError:
                pass  # relayed verbatim; the backend will reject it

        want = str(body.get("model") or "").strip()
        model = resolve(want) if want else default_model()
        if model is None:
            return self._json(400, {"error": {"type": "invalid_request",
                                              "message": f"unknown model {want!r}"},
                                    "known": [m.id for m in MODELS]})

        ok, why = ensure(model)
        if not ok:
            # 503 with a Retry-After, because every failure here is "not now" rather
            # than "never": a drain timed out, a pin blocked it, or memory is short.
            return self._json(503, {"error": {"type": "model_unavailable",
                                              "message": f"{model.id}: {why}"}},
                              extra={"Retry-After": "30", "x-mind-model": model.id,
                                     "x-mind-arbiter": why})

        with _state_lock:
            model.inflight += 1
        try:
            req = urllib.request.Request(  # noqa: S310 - our own configured base
                model.base + self.path, data=raw, method=method,
                headers=model.headers({"Content-Type": ctype}))
            try:
                resp = urllib.request.urlopen(req, timeout=TIMEOUT)  # noqa: S310
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                return self._json(exc.code, {"error": {"type": "upstream",
                                  "message": payload[:500].decode("utf-8", "replace")}},
                                  extra={"x-mind-model": model.id})
            except Exception as exc:
                return self._json(502, {"error": {"type": "upstream_unreachable",
                                  "message": f"{model.id}: {type(exc).__name__}"}},
                                  extra={"x-mind-model": model.id})

            self.send_response(resp.status)
            self.send_header("Content-Type",
                             resp.headers.get("Content-Type", "application/json"))
            self.send_header("x-mind-model", model.id)
            self.send_header("x-mind-arbiter", why)
            self.send_header("Connection", "close")
            if (resp.headers.get("Content-Type") or "").startswith("text/event-stream"):
                self.end_headers()
                while True:
                    chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        break
            else:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            resp.close()
            self.close_connection = True
        finally:
            # Must happen on every path, or a failed request would leave the model
            # permanently "busy" and unswappable.
            with _state_lock:
                model.inflight -= 1


if __name__ == "__main__":
    if shutil.which("systemctl") is None:
        LOG("systemctl not found; only launcher=exec models can be managed")
    MODELS = _load_registry(REGISTRY_PATH)
    LOG(f"{HOST}:{PORT} registry={REGISTRY_PATH} "
        + " ".join(f"{m.id}={m.weight_gib:.0f}GiB" + ("*" if m.opt_in else "")
                   for m in MODELS)
        + f" auth={'on' if AUTH_KEY else 'off'} reserve={RESERVE_GIB:.0f}GiB")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
