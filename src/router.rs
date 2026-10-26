//! Which *host* should serve a request: the backend most likely to still hold its
//! prefix, and — since one host may now offer a choice of models — one that can serve
//! the model that was asked for.
//!
//! Normative behaviour is `SPEC.md` §2; requirement ids appear in comments where the
//! reason for a line is not obvious from the line.
//!
//! The decision is a pure function. [`choose`] takes a snapshot (which backends are
//! healthy, what each claims to serve, the affinity table, the index's estimates) and
//! returns a backend and the reason, with no I/O and no clock of its own. Everything
//! that probes, proxies and remembers lives in [`Router`] around it. That split is why
//! the routing rules can be tested exhaustively without a model or a socket.

use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::http::{self, Reply, Request, Response};
use crate::{Estimate, Observation, PrefixIndex};

/// Block size for the index. 16 token ids is ~4 messages of hashed text, a sensible
/// granularity for a chat prefix.
pub const BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backend {
    pub name: String,
    pub base: String,
    pub key: String,
}

impl Backend {
    pub fn new(name: &str, base: &str, key: &str) -> Self {
        Backend { name: name.into(), base: base.into(), key: key.into() }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub backends: Vec<Backend>,
    pub auth_key: String,
    pub probe_ttl: Duration,
    pub affinity_ttl: Duration,
    pub max_conversations: usize,
    pub upstream_timeout: Duration,
    pub preferred: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "0.0.0.0".into(),
            port: 8015,
            backends: Vec::new(),
            auth_key: String::new(),
            probe_ttl: Duration::from_secs(20),
            affinity_ttl: Duration::from_secs(3600),
            max_conversations: 2048,
            upstream_timeout: Duration::from_secs(900),
            preferred: "auto".into(),
        }
    }
}

/// What a backend says it can serve, from whichever `/v1/models` shape it used (R-8).
///
/// Three shapes are in use on this LAN and the parser takes all of them: OpenAI's
/// `{"data":[{"id":…}]}`, the Ollama-style `{"models":[{"name":…}]}` one backend
/// answers with, and the arbiter's, which is OpenAI's plus an `aliases` array.
/// Anything unparseable yields an empty set, which callers must read as "made no
/// claim" rather than "serves nothing" (R-9).
pub fn advertised(payload: &[u8]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(doc) = serde_json::from_slice::<Value>(payload) else {
        return names;
    };
    let entries = ["data", "models"]
        .iter()
        .filter_map(|key| doc.get(*key).and_then(Value::as_array))
        .flatten();
    for entry in entries {
        for field in ["id", "name", "model"] {
            if let Some(value) = entry.get(field).and_then(Value::as_str)
                && !value.trim().is_empty() {
                    names.insert(value.trim().to_ascii_lowercase());
                }
        }
        if let Some(aliases) = entry.get("aliases").and_then(Value::as_array) {
            for alias in aliases.iter().filter_map(Value::as_str) {
                if !alias.trim().is_empty() {
                    names.insert(alias.trim().to_ascii_lowercase());
                }
            }
        }
    }
    names
}

/// Text of a message's `content`, which may be a string or an array of content parts.
fn content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Identify a conversation from the request itself (R-2).
///
/// Two properties matter and the obvious implementations fail one of them. The key must
/// be **stable across turns**, so it cannot depend on anything past the opening — a
/// window over the most recent messages makes every turn a new conversation. And it
/// must be **distinct per conversation**, so the system prompt alone will not do, since
/// every session shares it. Model + system prompt + first user message satisfies both,
/// and a client needs to send no session id.
pub fn conversation_key(body: &Value) -> String {
    use sha2::{Digest, Sha256};

    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let mut parts = vec![body.get("model").and_then(Value::as_str).unwrap_or("").to_string()];
    let mut seen_user = false;
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role != "system" && role != "user" {
            continue;
        }
        if role == "user" && seen_user {
            continue;
        }
        parts.push(format!("{role}:{}", content_text(message)));
        if role == "user" {
            seen_user = true;
        }
    }
    if !seen_user {
        return String::new(); // no user turn yet: nothing to pin
    }
    let digest = Sha256::digest(parts.join("\0").as_bytes());
    digest.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

/// The token ids the index should key on.
///
/// The index hashes exactly what the engine is sent, so real ids would be ideal, but
/// deriving them needs the model's tokenizer, which a router does not have. Message
/// text is hashed into stable 32-bit ids instead: enough to decide whether two requests
/// share an opening, which is the index's only job. It is an approximation of the
/// engine's prefix matching, not a replacement for it.
pub fn hashable_tokens(body: &Value) -> Vec<u32> {
    use sha2::{Digest, Sha256};

    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let digest = Sha256::digest(format!("{role}:{}", content_text(message)).as_bytes());
        // One id per 4-byte word, so one message contributes 8 ids.
        for chunk in digest.chunks(4) {
            ids.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    ids
}

/// A routing decision: which backend, and why, in the form that reaches the client as
/// `x-mind-route` (R-16).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub backend: Option<usize>,
    pub why: String,
}

impl Decision {
    fn none(why: &str) -> Self {
        Decision { backend: None, why: why.to_string() }
    }
}

/// Pick a backend and say why. Pure: no clock, no sockets, no interior mutability.
///
/// `healthy` is parallel to `backends`. `claims` maps a backend name to what it says it
/// serves. `affinity` maps a conversation key to a backend name. `estimates` is what
/// the index returned for this prompt, best first.
#[allow(clippy::too_many_arguments)]
pub fn choose(
    backends: &[Backend],
    healthy: &[bool],
    claims: &HashMap<String, BTreeSet<String>>,
    affinity: &HashMap<String, (String, Instant)>,
    estimates: &[Estimate],
    preferred: &str,
    body: &Value,
) -> Decision {
    if preferred != "auto" {
        let found = backends.iter().position(|b| b.name == preferred);
        return Decision { backend: found, why: format!("pinned:{preferred}") };
    }

    let mut candidates: Vec<usize> = (0..backends.len()).filter(|i| healthy[*i]).collect();
    if candidates.is_empty() {
        return Decision::none("none-healthy");
    }

    // Which backends can serve the model that was asked for (R-10). Narrowing is
    // one-sided on purpose: a backend that ignores the `model` field, or whose listing
    // could not be parsed, claims nothing, and reading that as "cannot serve it" would
    // route nowhere. So narrow only when somebody does claim it.
    let want = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let mut suffix = "";
    if !want.is_empty() {
        let able: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|i| claims.get(&backends[*i].name).is_some_and(|set| set.contains(&want)))
            .collect();
        if !able.is_empty() && able.len() < candidates.len() {
            candidates = able;
            suffix = "+model";
        }
    }

    // Affinity comes *after* capability (R-11): a conversation held on a backend that
    // no longer claims this model must be re-routed, not honoured.
    let key = conversation_key(body);
    if !key.is_empty()
        && let Some((held, _)) = affinity.get(&key)
            && let Some(index) = candidates.iter().copied().find(|i| &backends[*i].name == held) {
                return Decision { backend: Some(index), why: format!("affinity:{held}{suffix}") };
            }

    // No affinity. If the index recognises this prompt, send it where the prefix most
    // likely still lives (R-3); the index only ever reorders backends already eligible.
    for estimate in estimates {
        if estimate.cached_tokens == 0 {
            continue;
        }
        if let Some(index) = candidates
            .iter()
            .copied()
            .find(|i| backends[*i].name == estimate.backend)
        {
            return Decision {
                backend: Some(index),
                why: format!("prefix:{}:{}{suffix}", estimate.backend, estimate.cached_tokens),
            };
        }
    }

    // Otherwise the first healthy backend in declaration order (R-4).
    let first = candidates[0];
    let reason = if key.is_empty() { "nokey" } else { "new" };
    Decision {
        backend: Some(first),
        why: format!("{reason}:{}{suffix}", backends[first].name),
    }
}

/// Drop expired conversations, then the oldest if the table is over its cap.
pub fn prune_affinity(
    affinity: &mut HashMap<String, (String, Instant)>,
    now: Instant,
    ttl: Duration,
    max: usize,
) {
    affinity.retain(|_, (_, seen)| now.duration_since(*seen) <= ttl);
    if affinity.len() > max {
        let mut by_age: Vec<(String, Instant)> =
            affinity.iter().map(|(k, (_, seen))| (k.clone(), *seen)).collect();
        by_age.sort_by_key(|(_, seen)| *seen);
        for (key, _) in by_age.into_iter().take(affinity.len() - max) {
            affinity.remove(&key);
        }
    }
}

struct State {
    health: HashMap<String, (bool, Instant)>,
    claims: HashMap<String, BTreeSet<String>>,
    affinity: HashMap<String, (String, Instant)>,
    preferred: String,
    index: PrefixIndex,
}

/// The router: configuration plus the mutable state behind one lock.
///
/// The index lives in this process rather than in a child reached over a pipe, as it
/// did when the router was Python. N-2 is therefore structural: there is no separate
/// process that can be missing or die mid-request.
pub struct Router {
    config: Config,
    state: Mutex<State>,
}

impl Router {
    pub fn new(config: Config) -> Self {
        let now = Instant::now() - Duration::from_secs(86400); // every probe starts stale
        let state = State {
            health: config
                .backends
                .iter()
                .map(|b| (b.name.clone(), (false, now)))
                .collect(),
            claims: config.backends.iter().map(|b| (b.name.clone(), BTreeSet::new())).collect(),
            affinity: HashMap::new(),
            preferred: config.preferred.clone(),
            index: PrefixIndex::new(BLOCK_SIZE),
        };
        Router { config, state: Mutex::new(state) }
    }

    fn headers_for(&self, backend: &Backend, extra: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if !backend.key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {}", backend.key)));
        }
        for (name, value) in extra {
            headers.push(((*name).to_string(), (*value).to_string()));
        }
        headers
    }

    /// Cached reachability check, which also learns what the backend serves (R-7).
    ///
    /// One request answers both questions, because the body of the health probe *is*
    /// the model listing. The Python fetched it and threw the body away.
    pub fn probe(&self, backend: &Backend) -> bool {
        {
            let state = self.state.lock().unwrap();
            if let Some((ok, at)) = state.health.get(&backend.name)
                && at.elapsed() <= self.config.probe_ttl {
                    return *ok;
                }
        }
        let mut ok = false;
        let mut names = BTreeSet::new();
        if let Ok(upstream) = http::fetch(
            &backend.base,
            "/v1/models",
            "GET",
            &self.headers_for(backend, &[]),
            None,
            Duration::from_secs(4),
        ) {
            ok = (200..300).contains(&upstream.status);
            if ok {
                // Cap the read: a model listing is small, and a health probe must not
                // become expensive because a backend misbehaves.
                let mut body = Vec::new();
                let mut reader = upstream.into_reader().take(256 * 1024);
                if reader.read_to_end(&mut body).is_ok() {
                    names = advertised(&body);
                }
            }
        }
        let mut state = self.state.lock().unwrap();
        state.health.insert(backend.name.clone(), (ok, Instant::now()));
        if ok {
            // Replace the claim only on a successful probe: a backend that is briefly
            // unreachable has not stopped serving what it served a moment ago, and its
            // health already excludes it from routing.
            state.claims.insert(backend.name.clone(), names);
        }
        ok
    }

    fn decide(&self, body: &Value) -> Decision {
        let healthy: Vec<bool> = self.config.backends.iter().map(|b| self.probe(b)).collect();
        let tokens = hashable_tokens(body);
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        prune_affinity(
            &mut state.affinity,
            now,
            self.config.affinity_ttl,
            self.config.max_conversations,
        );
        let estimates = if tokens.is_empty() {
            Vec::new()
        } else {
            state.index.estimate(&tokens)
        };
        let preferred = state.preferred.clone();
        choose(
            &self.config.backends,
            &healthy,
            &state.claims,
            &state.affinity,
            &estimates,
            &preferred,
            body,
        )
    }

    fn remember(&self, key: &str, backend: &str) {
        if key.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state.affinity.insert(key.to_string(), (backend.to_string(), Instant::now()));
    }

    /// Teach the index from a reply: `cache_n` is the engine's own reuse figure, so a
    /// later estimate is corrected by measurement rather than by belief (I-2).
    fn teach(&self, backend: &str, tokens: &[u32], reply: &[u8]) {
        if tokens.is_empty() {
            return;
        }
        let Ok(doc) = serde_json::from_slice::<Value>(reply) else {
            return; // an unparseable body teaches the index nothing
        };
        let timings = doc.get("timings");
        let cached = timings
            .and_then(|t| t.get("cache_n"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let prompt_n = timings
            .and_then(|t| t.get("prompt_n"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if cached == 0 && prompt_n == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state.index.observe(
            backend,
            tokens,
            Observation { cached_tokens: cached, prompt_tokens: cached + prompt_n },
        );
    }

    fn authed(&self, request: &Request) -> bool {
        self.config.auth_key.is_empty() || request.client_key() == self.config.auth_key
    }

    /// Everything the status endpoint reports, including what each backend claims to
    /// serve — which is how a misrouted model name gets diagnosed.
    fn status(&self) -> Value {
        let backends: Vec<Value> = self
            .config
            .backends
            .iter()
            .map(|b| {
                let healthy = self.probe(b);
                let state = self.state.lock().unwrap();
                serde_json::json!({
                    "name": b.name,
                    "base": b.base,
                    "healthy": healthy,
                    "models": state.claims.get(&b.name).map(|s| s.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
                })
            })
            .collect();
        let state = self.state.lock().unwrap();
        serde_json::json!({
            "preferred": state.preferred,
            "affinity_ttl_s": self.config.affinity_ttl.as_secs(),
            "conversations": state.affinity.len(),
            "backends": backends,
        })
    }

    pub fn handle(&self, request: Request) -> Reply {
        if !self.authed(&request) {
            return Response::error(401, "authentication_error", "missing or wrong API key").into();
        }
        let path = request.path().to_string();
        if request.method == "GET" && path == "/router/status" {
            return Response::json(200, &self.status()).into();
        }
        if request.method == "POST" {
            if let Some(want) = path.strip_prefix("/router/prefer/") {
                if want != "auto" && !self.config.backends.iter().any(|b| b.name == want) {
                    return Response::error(400, "invalid_request_error", &format!("unknown backend {want:?}")).into();
                }
                let mut state = self.state.lock().unwrap();
                state.preferred = want.to_string();
                return Response::json(200, &serde_json::json!({"preferred": want})).into();
            }
            if path == "/router/session/clear" {
                let mut state = self.state.lock().unwrap();
                let dropped = state.affinity.len();
                state.affinity.clear();
                return Response::json(200, &serde_json::json!({"cleared": dropped})).into();
            }
            if let Some(want) = path.strip_prefix("/router/forget/") {
                // "That backend's cache is gone." Called by a restart hook, or by an
                // arbiter that has just swapped a model out from under this name.
                if !self.config.backends.iter().any(|b| b.name == want) {
                    return Response::error(400, "invalid_request_error", &format!("unknown backend {want:?}")).into(); // R-15
                }
                let mut state = self.state.lock().unwrap();
                state.index.forget_backend(want); // R-13
                state.claims.insert(want.to_string(), BTreeSet::new());
                // Force the next probe, so a swapped-in model's names are learned now.
                state.health.insert(want.to_string(), (false, Instant::now() - Duration::from_secs(86400)));
                // Affinity is deliberately left alone (R-14).
                return Response::json(200, &serde_json::json!({"forgot": want})).into();
            }
        }
        self.forward(request)
    }

    fn forward(&self, request: Request) -> Reply {
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let body = if body.is_object() { body } else { Value::Null };
        let key = conversation_key(&body);
        let tokens = hashable_tokens(&body);

        let decision = self.decide(&body);
        let Some(index) = decision.backend else {
            return Response::error(503, "unavailable", "no backend reachable").into();
        };
        let mut backend = &self.config.backends[index];
        let mut why = decision.why;

        let content_type = request.header("content-type").unwrap_or("application/json").to_string();
        let payload = if request.body.is_empty() { None } else { Some(request.body.as_slice()) };
        let attempt = |target: &Backend| {
            http::fetch(
                &target.base,
                &request.target,
                &request.method,
                &self.headers_for(target, &[("Content-Type", &content_type)]),
                payload,
                self.config.upstream_timeout,
            )
        };

        let mut upstream = match attempt(backend) {
            Ok(up) if up.status < 500 => Some(up),
            other => {
                // A 5xx or an unreachable backend is retried once elsewhere (R-19).
                let alternate = self
                    .config
                    .backends
                    .iter()
                    .enumerate()
                    .find(|(i, b)| *i != index && self.probe(b))
                    .map(|(_, b)| b);
                match (other, alternate) {
                    (Ok(up), None) => Some(up), // nowhere else to go: relay the 5xx
                    (_, Some(alt)) => {
                        backend = alt;
                        why = format!("failover:{}", alt.name);
                        attempt(alt).ok()
                    }
                    (Err(_), None) => None,
                }
            }
        };

        let Some(up) = upstream.take() else {
            return Response::error(503, "unavailable", "all backends failed").into();
        };

        self.remember(&key, &backend.name);

        let status = up.status;
        let content_type = up.content_type().to_string();
        // Pass the backend's own account of itself through (R-17). When the backend is
        // an arbiter, x-mind-arbiter is the only explanation a client gets for a first
        // request that took seconds, and rebuilding the response here used to drop it.
        let mut headers = vec![("Content-Type".to_string(), content_type.clone())];
        for name in ["x-mind-model", "x-mind-arbiter"] {
            if let Some(value) = up.header(name) {
                headers.push((name.to_string(), value.to_string()));
            }
        }
        headers.push(("x-mind-backend".to_string(), backend.name.clone()));
        headers.push(("x-mind-route".to_string(), why));

        if up.is_event_stream() {
            // Streaming: the index learns nothing from this reply, because the usage
            // and timings arrive inside the stream rather than as a JSON body.
            return Reply::Stream { status, headers, source: up.into_reader() };
        }
        let data = up.read_body().unwrap_or_default();
        self.teach(&backend.name, &tokens, &data);
        Reply::Buffered(Response { status, headers, body: data })
    }

    pub fn run(self: Arc<Self>) -> std::io::Result<()> {
        let address = format!("{}:{}", self.config.host, self.config.port);
        http::serve(address, move |request| self.handle(request))
    }

    /// Serve on an already-bound listener, so a test can learn the port first.
    pub fn run_on(self: Arc<Self>, listener: std::net::TcpListener) -> std::io::Result<()> {
        http::serve_on(listener, move |request| self.handle(request))
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// Read configuration from the environment, matching the names the units already use.
///
/// An empty `MIND_<NAME>` removes that backend, so one binary serves every host
/// (R-22): a configured-but-unreachable backend would otherwise fail a probe forever.
pub fn config_from_env() -> Config {
    let env = |name: &str, fallback: &str| {
        std::env::var(name).map(|v| v.trim().to_string()).unwrap_or_else(|_| fallback.to_string())
    };
    let auth_key = {
        let configured = env("MIND_ROUTER_KEY", "");
        if configured.is_empty() {
            std::fs::read_to_string(
                std::path::Path::new(&env("HOME", "")).join(".ds4-lan-key"),
            )
            .map(|v| v.trim().to_string())
            .unwrap_or_default()
        } else {
            configured
        }
    };

    // Per-backend keys are separate because a backend may be another host's
    // authenticated router (R-23). `amd` falls back to the shared key: on this host it
    // is a local server that binds the LAN and therefore keys itself.
    let specs: Vec<(&str, &str, String)> = vec![
        ("studio", "MIND_STUDIO", env("MIND_STUDIO_KEY", "")),
        ("mac", "MIND_MAC", env("MIND_MAC_KEY", "")),
        (
            "amd",
            "MIND_AMD",
            {
                let own = env("MIND_AMD_KEY", "");
                if own.is_empty() { auth_key.clone() } else { own }
            },
        ),
    ];
    let defaults: HashMap<&str, &str> = HashMap::from([
        ("MIND_STUDIO", "http://127.0.0.1:8016"),
        ("MIND_MAC", "http://127.0.0.1:8091"),
        ("MIND_AMD", "http://192.168.1.123:8000"),
    ]);
    let backends = specs
        .into_iter()
        .filter_map(|(name, var, key)| {
            let base = env(var, defaults[var]);
            (!base.is_empty()).then(|| Backend::new(name, &base, &key))
        })
        .collect();

    let secs = |name: &str, fallback: u64| {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).map_or_else(
            || Duration::from_secs(fallback),
            Duration::from_secs_f64,
        )
    };
    Config {
        host: env("MIND_ROUTER_HOST", "0.0.0.0"),
        port: env("MIND_ROUTER_PORT", "8015").parse().unwrap_or(8015),
        backends,
        auth_key,
        probe_ttl: secs("MIND_PROBE_TTL", 20),
        affinity_ttl: secs("MIND_AFFINITY_TTL", 3600),
        max_conversations: env("MIND_MAX_CONVERSATIONS", "2048").parse().unwrap_or(2048),
        upstream_timeout: secs("MIND_UPSTREAM_TIMEOUT", 900),
        preferred: env("MIND_PREFER", "auto"),
    }
}
