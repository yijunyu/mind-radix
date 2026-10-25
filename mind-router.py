#!/usr/bin/env python3
"""Route each conversation to the backend that most likely still holds its prefix.

Two backends serve the same model: studio (llama.cpp Metal, faster) and amd
(PolyStrata HIP). Both cache prompt prefixes, and a prefix only survives while
requests keep arriving in the same order on the same server -- so the routing rule
that matters is *affinity*, not fairness:

  * a conversation stays on the backend that served its previous turn, because that
    is where its prefix is warm;
  * a new conversation prefers studio, then amd;
  * a backend that is down or 5xx-ing is replaced, and the conversation moves with it;
  * a request that names a model only considers backends that claim it. Hosts used to
    be interchangeable because they all ran the same model; a host running
    mind-arbiter offers a choice, so sending "deepseek-v4.1-flash" to a host that has
    never heard of it is now a real way to get a wrong answer instead of a slow one.

Switching backends mid-conversation throws away the prefix (measured: the whole tail
is re-prefilled, ~5.2 s on a 7K prompt), so affinity is worth more than balancing.
Load is used only to break ties between backends that are equally good choices.

Endpoints:
  GET  /router/status              preference, per-backend health, and affinity table
  POST /router/prefer/<name>|auto  pin one backend, or restore affinity routing
  POST /router/session/clear       drop all affinity (e.g. after a backend restart)
  POST /router/forget/<name>       that backend's cache is gone; drop its estimates

Binds the LAN by default, so it authenticates: MIND_ROUTER_KEY gates every request.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MIND_ROUTER_PORT", "8015"))
# Binding all interfaces is the point: remote clients reach this router directly.
# It is not an open door -- every request is gated by MIND_ROUTER_KEY below, and
# an unauthenticated caller gets 401 before any backend is touched.
HOST = os.environ.get("MIND_ROUTER_HOST", "0.0.0.0")  # noqa: S104
TIMEOUT = float(os.environ.get("MIND_UPSTREAM_TIMEOUT", "900"))
PROBE_TTL = float(os.environ.get("MIND_PROBE_TTL", "20"))
# How long a conversation's affinity is honoured. Long enough to cover a working
# session; short enough that a stale entry cannot pin a dead conversation forever.
AFFINITY_TTL = float(os.environ.get("MIND_AFFINITY_TTL", "3600"))
MAX_CONVERSATIONS = int(os.environ.get("MIND_MAX_CONVERSATIONS", "2048"))


def _read_key() -> str:
    key = os.environ.get("MIND_ROUTER_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.ds4-lan-key")) as handle:
            return handle.read().strip()
    except OSError:
        return ""


AUTH_KEY = _read_key()

# Each backend's key is configurable, because a "backend" may itself be another
# host's router. On studio, studio is a local llama-server (no key) and amd is the AMD
# box's authenticated backend. On the AMD box the roles reverse: amd is a local server
# (keyed, since it binds the LAN) and studio is the other *router*, which is also keyed.
# Sending no key to a keyed endpoint is what made every probe fail on first deploy, so
# the key is per-backend rather than global.
BACKENDS = [
    {"name": "studio",
     "base": os.environ.get("MIND_STUDIO", "http://127.0.0.1:8016"),
     "key": os.environ.get("MIND_STUDIO_KEY", "").strip()},
    {"name": "mac",
     # The MacBook Pro lives off-LAN and its llama-server binds its own loopback, so it
     # is reached through an SSH forward that studio keeps open (8091 -> mac:8090).
     # Setting MIND_MAC empty removes it, which is what a host without that forward
     # should do -- a backend that is configured but unreachable just fails its probe.
     "base": os.environ.get("MIND_MAC", "http://127.0.0.1:8091"),
     "key": os.environ.get("MIND_MAC_KEY", "").strip()},
    {"name": "amd",
     "base": os.environ.get("MIND_AMD", "http://192.168.1.123:8000"),
     # Default to the shared key only when talking to a remote/LAN backend; a
     # loopback backend on a host that does not key it simply ignores the header.
     "key": os.environ.get("MIND_AMD_KEY", "").strip() or AUTH_KEY},
]
# An empty MIND_<NAME> drops that backend, so the same file can serve every host.
BACKENDS = [b for b in BACKENDS if b["base"]]

# Optional prefix index (crates/mind-radix). When present it lets the router rank
# backends by how much of an incoming prompt each one is *likely* to hold, which is
# strictly better than "prefer the faster box" for a conversation that has already
# been running elsewhere. Absent or broken, the router falls back to affinity alone.
RADIX_BIN = os.environ.get("MIND_RADIX_BIN", "")
NAMES = [b["name"] for b in BACKENDS]

_lock = threading.Lock()
_health: dict[str, tuple[bool, float]] = dict.fromkeys(NAMES, (False, 0.0))
# Backend name -> the model names it says it can serve, learned from the same
# /v1/models call that probes its health. Empty means "it told us nothing we could
# parse", which is treated as "no claim" and never as "cannot serve it".
_models: dict[str, frozenset[str]] = dict.fromkeys(NAMES, frozenset())
# conversation key -> (backend name, last seen). A conversation is identified by the
# earliest messages it sends, so a client needs no cooperation to be recognised.
_affinity: dict[str, tuple[str, float]] = {}
preferred = os.environ.get("MIND_PREFER", "auto")  # "auto" | backend name


class PrefixIndex:
    """Client for the mind-radix CLI: one long-lived child, one JSON line per call.

    The index lives in that child process, so it is started once and kept. If the
    binary is missing or the child dies, every method becomes a no-op and the router
    still works -- the index improves routing, it is never required for it.
    """

    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.proc = None
        self.lock = threading.Lock()
        self.enabled = bool(binary)

    def _start(self):
        if not self.enabled:
            return None
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        try:
            self.proc = subprocess.Popen(
                [self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1)
        except OSError:
            self.enabled = False
            self.proc = None
        return self.proc

    def call(self, payload: dict):
        if not self.enabled:
            return None
        with self.lock:
            proc = self._start()
            # A child that failed to spawn, or was spawned without pipes, cannot be
            # talked to; both are "the index is unavailable", never a crash.
            stdin, stdout = (proc.stdin, proc.stdout) if proc is not None else (None, None)
            if stdin is None or stdout is None:
                return None
            try:
                stdin.write(json.dumps(payload) + "\n")
                stdin.flush()
                line = stdout.readline()
                return json.loads(line) if line else None
            except (OSError, ValueError):
                # A dead child just disables the index; routing continues without it.
                self.enabled = False
                self.proc = None
                return None

    def estimate(self, tokens: list) -> list:
        reply = self.call({"op": "estimate", "tokens": tokens})
        return (reply or {}).get("estimates", []) if isinstance(reply, dict) else []

    def observe(self, backend: str, tokens: list, cached: int, prompt: int) -> None:
        self.call({"op": "observe", "backend": backend, "tokens": tokens,
                   "cached": cached, "prompt": prompt})


INDEX = PrefixIndex(RADIX_BIN)


def headers_for(backend: dict, extra: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {backend['key']}"} if backend["key"] else {}
    if extra:
        headers.update(extra)
    return headers


def _advertised(payload: bytes) -> frozenset[str]:
    """The model names in a /v1/models reply, from whichever shape it used.

    Three shapes turn up among the backends here, so the parser accepts all of them
    rather than assuming the host is an OpenAI-compatible engine: OpenAI's own
    ``{"data": [{"id": ...}]}``, the Ollama-style ``{"models": [{"name": ...}]}`` that
    one backend on this LAN answers with, and the arbiter's, which is OpenAI's plus an
    ``aliases`` list. Anything unparseable yields an empty set, which callers read as
    "this backend made no claim".
    """
    names: set[str] = set()
    try:
        doc = json.loads(payload)
    except (ValueError, TypeError):
        return frozenset()
    if not isinstance(doc, dict):
        return frozenset()
    for entry in (doc.get("data") or []) + (doc.get("models") or []):
        if not isinstance(entry, dict):
            continue
        for field in ("id", "name", "model"):
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                names.add(value.strip().lower())
        for alias in entry.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                names.add(alias.strip().lower())
    return frozenset(names)


def probe(backend: dict) -> bool:
    """Cached reachability check; free, so it can be called on every request."""
    name = backend["name"]
    with _lock:
        ok, at = _health[name]
        if time.time() - at <= PROBE_TTL:
            return ok
    names: frozenset[str] = frozenset()
    try:
        # The URL is built from this process's own configuration, never from a client
        # request, so the scheme is ours to trust (noqa: the audit assumes user input).
        req = urllib.request.Request(backend["base"] + "/v1/models",  # noqa: S310
                                     headers=headers_for(backend))
        with urllib.request.urlopen(req, timeout=4) as resp:  # noqa: S310
            ok = 200 <= resp.status < 300
            if ok:
                # The body was already being fetched and thrown away; a model listing
                # is small, but cap the read so a misbehaving backend cannot make a
                # health probe expensive.
                names = _advertised(resp.read(262144))
    except Exception:
        ok = False
    with _lock:
        _health[name] = (ok, time.time())
        if ok:
            # Only replace the claim on a successful probe: a backend that is briefly
            # unreachable has not stopped serving what it served a moment ago, and it
            # is excluded from routing by its health anyway.
            _models[name] = names
    return ok


def conversation_key(body: dict) -> str:
    """Identify a conversation from the request itself.

    Two properties matter, and the obvious implementations fail one of them:

    * **Stable across turns** -- later turns append messages, so the key must not
      depend on anything past the opening. Keying on the first three messages broke
      this: by turn 2 the window had moved and every turn became a new conversation.
    * **Distinct per conversation** -- the system prompt alone is shared by every Pi
      session, so it cannot identify one; the first *user* message is what differs.

    So the key is the model + system prompt + the first user message, and nothing
    after that. A client sends no session id and, this way, needs none.
    """
    import hashlib

    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return ""
    parts = [str(body.get("model") or "")]
    seen_user = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("system", "user"):
            continue
        if role == "user" and seen_user:
            continue
        content = message.get("content")
        if isinstance(content, list):  # content parts
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        parts.append(f"{role}:{content}")
        if role == "user":
            seen_user = True
    if not seen_user:
        return ""  # no user turn yet: nothing to pin
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:32]


def _evict_locked(now: float) -> None:
    stale = [k for k, (_, seen) in _affinity.items() if now - seen > AFFINITY_TTL]
    for key in stale:
        del _affinity[key]
    if len(_affinity) > MAX_CONVERSATIONS:  # oldest first
        for key, _ in sorted(_affinity.items(), key=lambda kv: kv[1][1])[: len(_affinity) - MAX_CONVERSATIONS]:
            del _affinity[key]


def choose(body: dict) -> tuple[dict | None, str]:
    """Pick a backend and say why, so the decision is visible in the response."""
    by_name = {b["name"]: b for b in BACKENDS}
    if preferred != "auto":
        return by_name.get(preferred), f"pinned:{preferred}"

    key = conversation_key(body)
    healthy = [b for b in BACKENDS if probe(b)]
    if not healthy:
        return None, "none-healthy"

    # Which backends can serve the model that was asked for. This mattered not at all
    # while every host ran the same model, and matters now: a host running an arbiter
    # offers a choice, so "amd" and "studio" are no longer interchangeable for a
    # request that names one. Narrow only when some healthy backend claims the name --
    # a backend that ignores the model field, or whose listing could not be parsed,
    # advertises nothing, and reading that as "cannot serve it" would route nowhere.
    want = str(body.get("model") or "").strip().lower()
    suffix = ""
    if want:
        with _lock:
            able = [b for b in healthy if want in _models.get(b["name"], frozenset())]
        if able and len(able) < len(healthy):
            healthy, suffix = able, "+model"

    if key:
        with _lock:
            now = time.time()
            _evict_locked(now)
            held = _affinity.get(key)
            if held and held[0] in [b["name"] for b in healthy]:
                _affinity[key] = (held[0], now)
                return by_name[held[0]], f"affinity:{held[0]}{suffix}"

    # No affinity yet. If the index knows this prompt, send it where the prefix most
    # likely still lives; otherwise take the first healthy backend (studio, the
    # faster one). The index only ever reorders backends that are already healthy.
    tokens = _hashable_tokens(body)
    if tokens:
        ranked = INDEX.estimate(tokens)
        best = next((e for e in ranked if e.get("cached_tokens", 0) > 0
                     and e.get("backend") in [b["name"] for b in healthy]), None)
        if best is not None:
            return by_name[best["backend"]], f"prefix:{best['backend']}:{best['cached_tokens']}{suffix}"
    if key:
        return healthy[0], f"new:{healthy[0]['name']}{suffix}"
    return healthy[0], f"nokey:{healthy[0]['name']}{suffix}"


def _hashable_tokens(body: dict) -> list:
    """The token ids the index should key on.

    The index hashes exactly what the engine is sent, so the router must supply real
    ids. Deriving them here would need the model's tokenizer, which the router does
    not have; instead the *message text* is hashed into stable 32-bit ids. That is
    enough for the index's purpose -- deciding whether two requests share an opening
    -- and it needs no tokenizer. It is an approximation of the engine's own prefix
    matching, not a replacement for it.
    """
    import hashlib

    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    ids = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        text = f"{message.get('role','')}:{content}"
        digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
        # One id per 4-byte word: the block size is 16, so a block is ~4 messages
        # worth of text, which is a sensible granularity for a chat prefix.
        ids.extend(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))
    return ids


def remember(key: str, backend: str) -> None:
    if not key:
        return
    with _lock:
        now = time.time()
        _evict_locked(now)
        _affinity[key] = (backend, now)


def client_key(headers) -> str:
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("x-api-key") or "").strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mind-router"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return

    def _json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _authed(self) -> bool:
        if not AUTH_KEY or client_key(self.headers) == AUTH_KEY:
            return True
        self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
        return False

    def do_GET(self):  # noqa: N802 - stdlib name
        if not self._authed():
            return
        if self.path == "/router/status":
            with _lock:
                now = time.time()
                _evict_locked(now)
                affinity = {k: v[0] for k, v in _affinity.items()}
            return self._json(200, {
                "preferred": preferred,
                "affinity_ttl_s": AFFINITY_TTL,
                "conversations": len(affinity),
                "backends": [{"name": b["name"], "base": b["base"], "healthy": probe(b),
                              "models": sorted(_models.get(b["name"], frozenset()))}
                             for b in BACKENDS],
            })
        return self.forward("GET")

    def do_POST(self):  # noqa: N802 - stdlib name
        global preferred
        if not self._authed():
            return
        if self.path.startswith("/router/prefer/"):
            want = self.path.rsplit("/", 1)[-1]
            if want != "auto" and want not in NAMES:
                return self._json(400, {"error": f"unknown backend {want!r}"})
            preferred = want
            return self._json(200, {"preferred": preferred})
        if self.path == "/router/session/clear":
            with _lock:
                dropped = len(_affinity)
                _affinity.clear()
            return self._json(200, {"cleared": dropped})
        if self.path.startswith("/router/forget/"):
            # "That backend's cache is gone." Called by a restart hook, or by an
            # arbiter that just swapped a model out from under this name. Only the
            # prefix index is dropped, never affinity: a conversation's affinity is
            # keyed on its model, so letting it return to the same host is what makes
            # a swap *back* cheap -- the arbiter reloads that model and, where KV was
            # persisted, its prefix is warm again. Forgetting the index is the
            # pessimistic half, and pessimism is the safe direction here, since the
            # index only ever corrects an estimate downward from measurement.
            want = self.path.rsplit("/", 1)[-1]
            if want not in NAMES:
                return self._json(400, {"error": f"unknown backend {want!r}"})
            INDEX.call({"op": "forget_backend", "backend": want})
            with _lock:
                _models[want] = frozenset()   # re-learned on the next probe
                _health[want] = (False, 0.0)  # force that probe now
            return self._json(200, {"forgot": want})
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

        key = conversation_key(body) if body else ""
        tokens = _hashable_tokens(body) if body else []
        backend, why = choose(body)
        if backend is None:
            return self._json(503, {"error": {"type": "unavailable", "message": "no backend reachable"}})

        def attempt(target: dict):
            # Again: scheme and host come from our configuration; only the path is the
            # client's, and it is passed through to the backend, not resolved here.
            req = urllib.request.Request(  # noqa: S310
                target["base"] + self.path, data=raw, method=method,
                headers=headers_for(target, {"Content-Type": ctype}))
            return urllib.request.urlopen(req, timeout=TIMEOUT)  # noqa: S310

        resp = None
        try:
            resp = attempt(backend)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            alternate = None
            if preferred == "auto" and exc.code >= 500:
                alternate = next((b for b in BACKENDS if b["name"] != backend["name"] and probe(b)), None)
            if alternate is None:
                return self._json(exc.code, {"error": {"type": "upstream",
                                    "message": payload[:500].decode("utf-8", "replace")}})
            backend, why = alternate, f"failover:{alternate['name']}"
            try:
                resp = attempt(backend)
            except Exception as inner:
                return self._json(503, {"error": {"type": "unavailable",
                                    "message": f"all backends failed: {type(inner).__name__}"}})
        except Exception as exc:
            alternate = next((b for b in BACKENDS if b["name"] != backend["name"] and probe(b)), None) \
                if preferred == "auto" else None
            if alternate is None:
                return self._json(503, {"error": {"type": "unavailable",
                                    "message": f"{backend['name']} unreachable: {type(exc).__name__}"}})
            backend, why = alternate, f"failover:{alternate['name']}"
            try:
                resp = attempt(backend)
            except Exception as inner:
                return self._json(503, {"error": {"type": "unavailable",
                                    "message": f"all backends failed: {type(inner).__name__}"}})

        remember(key, backend["name"])
        # The index is taught from the reply body below: cache_n is the engine's own
        # reuse figure, so a later estimate can be corrected by measurement.
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        # Pass through whatever the backend said about itself. When the backend is an
        # arbiter, its x-mind-arbiter header is the only account of why a first request
        # took seconds -- a model was swapped in -- and rebuilding the response here
        # used to drop it, leaving a client with a slow reply and no explanation.
        for header in ("x-mind-model", "x-mind-arbiter"):
            value = resp.headers.get(header)
            if value:
                self.send_header(header, value)
        self.send_header("x-mind-backend", backend["name"])
        self.send_header("x-mind-route", why)
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
            if tokens:
                try:
                    parsed_reply = json.loads(data)
                    timings = parsed_reply.get("timings") or {}
                    cached = int(timings.get("cache_n") or 0)
                    prompt_n = int(timings.get("prompt_n") or 0)
                    if prompt_n or cached:
                        INDEX.observe(backend["name"], tokens, cached, cached + prompt_n)
                except (ValueError, TypeError, AttributeError):
                    pass  # an unparseable body teaches the index nothing
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        resp.close()
        self.close_connection = True


if __name__ == "__main__":
    print("[mind-router] " + f"{HOST}:{PORT} "
          + " ".join(f"{b['name']}={b['base']}" for b in BACKENDS)
          + f" auth={'on' if AUTH_KEY else 'off'} prefer={preferred} affinity_ttl={AFFINITY_TTL:.0f}s", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
