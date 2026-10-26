//! Which *model* one host is currently holding.
//!
//! The router answers which host should serve a request, because every host serves the
//! same model. This answers a different question with a harder constraint: on a
//! unified-memory box the models do not fit together. On the gfx1151 box MiniCPM-V 4.7
//! needs ~35 GiB resident at 131072 ctx and DeepSeek V4.1 Flash ~88 GiB against a
//! ~122 GiB pool, so those two can never be co-resident.
//!
//! Normative behaviour is `SPEC.md` §3. Three rules are the whole design: swap only at
//! zero in-flight requests (A-7), evict only as much as necessary (A-9), and advertise
//! unloaded models so a client can see its choices while nothing is serving (A-21).
//!
//! Everything that touches the machine — systemd, processes, `/proc/meminfo`, the
//! engine's HTTP — is behind the [`Host`] trait. [`Arbiter`] holds only decisions. That
//! is what lets the eviction and drain rules be tested without a GPU, a 35 GiB model,
//! or a swap that takes forty minutes.

use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::http::{self, Reply, Request, Response};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launcher {
    /// `systemctl --user` owns the lifecycle.
    Systemd,
    /// This process spawns the server directly, for a host without systemd.
    Exec,
    /// Another host owns the lifecycle: never started, never stopped, never counted
    /// against local memory (A-4).
    External,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub aliases: Vec<String>,
    pub engine: String,
    pub launcher: Launcher,
    pub unit: Option<String>,
    pub argv: Vec<String>,
    pub base: String,
    pub log: Option<String>,
    pub key_env: Option<String>,
    pub key_env_file: Option<String>,
    pub weight_gib: f64,
    pub ready_timeout: Duration,
    pub slots: usize,
    pub opt_in: bool,
}

impl Model {
    /// Addressable by id or any alias (A-1), case-insensitively.
    pub fn matches(&self, want: &str) -> bool {
        let want = want.trim().to_ascii_lowercase();
        self.id.to_ascii_lowercase() == want
            || self.aliases.iter().any(|a| a.to_ascii_lowercase() == want)
    }

    pub fn external(&self) -> bool {
        self.launcher == Launcher::External
    }

    /// Resolve this backend's API key without ever putting it in the registry file.
    ///
    /// The registry is checked into a repo, so it names *where* the key lives, not the
    /// key (A-2): an environment variable, or a `KEY=VALUE` file such as the one the
    /// MiniCPM unit already reads. A missing key is not an error — a loopback backend
    /// that does not authenticate ignores the header.
    pub fn resolve_key(&self) -> String {
        let Some(name) = self.key_env.as_deref() else {
            return String::new();
        };
        if let Ok(value) = std::env::var(name)
            && !value.trim().is_empty() {
                return value.trim().to_string();
            }
        let Some(path) = self.key_env_file.as_deref() else {
            return String::new();
        };
        let path = expand_home(path);
        let Ok(text) = std::fs::read_to_string(path) else {
            return String::new();
        };
        for line in text.lines() {
            if let Some(value) = line.trim().strip_prefix(&format!("{name}=")) {
                return value.trim().trim_matches(['"', '\'']).to_string();
            }
        }
        String::new()
    }
}

fn expand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path.to_string(),
    }
}

/// Parse a registry document. The first non-`opt_in` entry becomes the default (A-6),
/// which is a property of order, so the list order is preserved exactly.
pub fn parse_registry(doc: &Value) -> Result<Vec<Model>, String> {
    let specs = doc
        .get("models")
        .or(Some(doc))
        .and_then(Value::as_array)
        .ok_or_else(|| "no 'models' list".to_string())?;
    if specs.is_empty() {
        return Err("no 'models' list".to_string());
    }
    let mut models = Vec::new();
    for spec in specs {
        let id = spec
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "a model has no id".to_string())?
            .to_string();
        let external = spec.get("external").and_then(Value::as_bool).unwrap_or(false);
        let launcher = match spec.get("launcher").and_then(Value::as_str) {
            _ if external => Launcher::External,
            Some("exec") => Launcher::Exec,
            _ => Launcher::Systemd,
        };
        models.push(Model {
            id,
            aliases: spec
                .get("aliases")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default(),
            engine: spec
                .get("engine")
                .and_then(Value::as_str)
                .unwrap_or("llama.cpp")
                .to_string(),
            launcher,
            unit: spec.get("unit").and_then(Value::as_str).map(str::to_string),
            argv: spec
                .get("argv")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default(),
            base: spec.get("base").and_then(Value::as_str).unwrap_or("").to_string(),
            log: spec.get("log").and_then(Value::as_str).map(str::to_string),
            key_env: spec.get("key_env").and_then(Value::as_str).map(str::to_string),
            key_env_file: spec.get("key_env_file").and_then(Value::as_str).map(str::to_string),
            weight_gib: spec.get("weight_gib").and_then(Value::as_f64).unwrap_or(0.0),
            ready_timeout: Duration::from_secs_f64(
                spec.get("ready_timeout_s").and_then(Value::as_f64).unwrap_or(300.0),
            ),
            slots: spec.get("slots").and_then(Value::as_u64).unwrap_or(0) as usize,
            opt_in: spec.get("opt_in").and_then(Value::as_bool).unwrap_or(false),
        });
    }
    Ok(models)
}

/// Everything that touches the machine, so the decisions above it can be tested.
pub trait Host: Send + Sync {
    fn running(&self, model: &Model) -> bool;
    fn healthy(&self, model: &Model) -> bool;
    /// Start, then wait until healthy or `ready_timeout` elapses.
    fn start(&self, model: &Model) -> bool;
    fn stop(&self, model: &Model);
    /// `MemAvailable`, in GiB. An unknown value MUST be reported as 0.0, which is read
    /// as "no headroom claim" and never as plenty (A-13).
    fn mem_available_gib(&self) -> f64;
    fn save_kv(&self, model: &Model) -> String;
    fn restore_kv(&self, model: &Model) -> String;
    /// Tell a router that this host's prompt cache is gone (A-14). Best-effort.
    fn notify_forget(&self, why: &str);
    fn log(&self, message: &str);
    /// Settle time after a stop: freeing is not instantaneous.
    fn settle(&self) {
        std::thread::sleep(Duration::from_secs(2));
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub auth_key: String,
    pub reserve_gib: f64,
    pub swap_wait: Duration,
    pub drain_wait: Duration,
    pub probe_ttl: Duration,
    pub upstream_timeout: Duration,
    pub pinned: String,
    pub router_base: String,
    pub router_backend: String,
    pub router_key: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "127.0.0.1".into(),
            port: 8010,
            auth_key: String::new(),
            reserve_gib: 8.0,
            swap_wait: Duration::from_secs(1200),
            drain_wait: Duration::from_secs(120),
            probe_ttl: Duration::from_secs(10),
            upstream_timeout: Duration::from_secs(1800),
            pinned: "none".into(),
            router_base: String::new(),
            router_backend: String::new(),
            router_key: String::new(),
        }
    }
}

/// Why a swap did or did not happen, as it reaches a client in `x-mind-arbiter`.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Resident,
    ResidentAfterWait,
    Loaded { evicted: Vec<String>, kv: String },
    SwapBusy,
    Busy(String),
    BlockedByPin(String),
    InsufficientMemory { free: f64, needed: f64 },
    StartFailed,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        matches!(self, Outcome::Resident | Outcome::ResidentAfterWait | Outcome::Loaded { .. })
    }

    /// The short form that goes in the response header (A-25).
    pub fn label(&self) -> String {
        match self {
            Outcome::Resident => "resident".into(),
            Outcome::ResidentAfterWait => "resident-after-wait".into(),
            Outcome::Loaded { evicted, .. } if evicted.is_empty() => "loaded".into(),
            Outcome::Loaded { evicted, .. } => format!("loaded-evicted:{}", evicted.join(",")),
            Outcome::SwapBusy => "swap-busy".into(),
            Outcome::Busy(id) => format!("busy:{id}"),
            Outcome::BlockedByPin(id) => format!("blocked-by-pin:{id}"),
            Outcome::InsufficientMemory { free, needed } => {
                format!("insufficient-memory:{free:.0}GiB-free-{needed:.0}GiB-needed")
            }
            Outcome::StartFailed => "start-failed".into(),
        }
    }
}

/// A request in flight against one model, which decrements however the request ends
/// (A-27).
///
/// The Python did this in a `finally`; a guard makes it structural, so a path added
/// later cannot forget it and leave a model permanently unswappable. It is public
/// because holding one is what "a request is in flight" *means* here: the proxy takes
/// one for the duration of a request, and a test takes one to stand for that request
/// without needing a model to generate anything.
pub struct Inflight(Arc<AtomicUsize>);

impl Inflight {
    fn acquire(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Inflight(Arc::clone(counter))
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct Arbiter<H: Host> {
    models: Vec<Model>,
    inflight: Vec<Arc<AtomicUsize>>,
    config: Config,
    host: H,
    pinned: Mutex<String>,
    /// One lock serialises every residency change (A-11), so two requests for two
    /// different unloaded models cannot both decide to evict the same third one.
    swap: Mutex<()>,
}

impl<H: Host> Arbiter<H> {
    pub fn new(models: Vec<Model>, config: Config, host: H) -> Self {
        let inflight = models.iter().map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let pinned = Mutex::new(config.pinned.clone());
        Arbiter { models, inflight, config, host, pinned, swap: Mutex::new(()) }
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn inflight(&self, index: usize) -> usize {
        self.inflight[index].load(Ordering::SeqCst)
    }

    /// Count a request against a model until the returned handle is dropped.
    pub fn hold(&self, index: usize) -> Inflight {
        Inflight::acquire(&self.inflight[index])
    }

    /// The host this arbiter drives. Lets a caller that supplied a [`Host`] inspect it
    /// afterwards, which is how the spec tests assert what was started and stopped.
    pub fn host_ref(&self) -> &H {
        &self.host
    }

    pub fn resolve(&self, want: &str) -> Option<usize> {
        self.models.iter().position(|m| m.matches(want))
    }

    /// First non-`opt_in` entry (A-6).
    pub fn default_model(&self) -> Option<usize> {
        self.models.iter().position(|m| !m.opt_in)
    }

    fn headroom(&self) -> f64 {
        self.host.mem_available_gib() - self.config.reserve_gib
    }

    /// Wait until nothing is in flight for this model, or give up (A-8).
    fn drain(&self, index: usize, wait: Duration) -> bool {
        let deadline = Instant::now() + wait;
        loop {
            if self.inflight(index) == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(200).min(wait));
        }
    }

    /// Make `target` resident, evicting only what is necessary.
    ///
    /// The ordering here is the safety property: a model is drained *before* it is
    /// stopped, and a drain that does not finish abandons the swap rather than the
    /// request (A-8). The caller turns a failure into 503 with `Retry-After`, so a
    /// client retries instead of a generation being killed.
    pub fn ensure(&self, target: usize) -> Outcome {
        if self.host.healthy(&self.models[target]) {
            return Outcome::Resident;
        }

        // std::sync::Mutex has no timed acquire, so poll until swap_wait elapses. A
        // swap already in progress can legitimately take many minutes.
        let deadline = Instant::now() + self.config.swap_wait;
        let guard = loop {
            if let Ok(guard) = self.swap.try_lock() {
                break guard;
            }
            if Instant::now() >= deadline {
                return Outcome::SwapBusy;
            }
            std::thread::sleep(Duration::from_millis(250));
        };
        let _guard = guard;

        // Another request may have loaded it while we waited for the lock.
        if self.host.healthy(&self.models[target]) {
            return Outcome::ResidentAfterWait;
        }

        let pinned = self.pinned.lock().unwrap().clone();
        let mut evicted = Vec::new();
        let mut headroom = self.headroom();
        let needed = self.models[target].weight_gib;

        for other in 0..self.models.len() {
            if headroom >= needed {
                break; // minimal eviction (A-9): stop as soon as it fits
            }
            if other == target || self.models[other].external() || !self.host.running(&self.models[other]) {
                continue;
            }
            if self.models[other].id == pinned {
                return Outcome::BlockedByPin(self.models[other].id.clone()); // A-10
            }
            if !self.drain(other, self.config.drain_wait) {
                return Outcome::Busy(self.models[other].id.clone()); // A-8
            }
            let note = self.host.save_kv(&self.models[other]);
            self.host.stop(&self.models[other]);
            evicted.push(format!("{}({note})", self.models[other].id));
            self.host.settle();
            headroom = self.headroom();
        }

        if headroom < needed {
            return Outcome::InsufficientMemory { free: headroom, needed }; // A-12
        }
        if !self.host.start(&self.models[target]) {
            return Outcome::StartFailed;
        }
        let kv = self.host.restore_kv(&self.models[target]);
        let outcome = Outcome::Loaded { evicted: evicted.clone(), kv: kv.clone() };
        self.host.log(&format!("{}: {}; kv {kv}", self.models[target].id, outcome.label()));
        if !evicted.is_empty() {
            // Only when something was actually stopped: loading into free memory leaves
            // the other models' caches intact, so there is nothing to forget (A-14).
            self.host.notify_forget(&format!("swap-to:{}", self.models[target].id));
        }
        outcome
    }

    /// Drain and stop a model on request.
    pub fn unload(&self, index: usize) -> Result<Value, Value> {
        if !self.drain(index, self.config.drain_wait) {
            return Err(serde_json::json!({
                "model": self.models[index].id,
                "error": "busy",
                "inflight": self.inflight(index),
            }));
        }
        let note = self.host.save_kv(&self.models[index]);
        self.host.stop(&self.models[index]);
        self.host.notify_forget(&format!("unload:{}", self.models[index].id));
        Ok(serde_json::json!({"model": self.models[index].id, "stopped": true, "kv": note}))
    }

    pub fn set_pin(&self, want: &str) -> String {
        let resolved = if want == "none" || want.is_empty() {
            "none".to_string()
        } else {
            match self.resolve(want) {
                Some(index) => self.models[index].id.clone(),
                None => return String::new(),
            }
        };
        *self.pinned.lock().unwrap() = resolved.clone();
        resolved
    }

    pub fn pinned(&self) -> String {
        self.pinned.lock().unwrap().clone()
    }

    /// The whole registry with a `loaded` flag and aliases (A-21, A-22).
    pub fn model_list(&self) -> Value {
        let data: Vec<Value> = self
            .models
            .iter()
            .map(|m| {
                let mut aliases = m.aliases.clone();
                aliases.sort();
                serde_json::json!({
                    "id": m.id,
                    "object": "model",
                    "owned_by": "local",
                    "aliases": aliases,
                    "loaded": self.host.healthy(m),
                    "opt_in": m.opt_in,
                })
            })
            .collect();
        serde_json::json!({"object": "list", "data": data})
    }

    pub fn status(&self) -> Value {
        let models: Vec<Value> = self
            .models
            .iter()
            .enumerate()
            .map(|(index, m)| {
                serde_json::json!({
                    "id": m.id,
                    "base": m.base,
                    "engine": m.engine,
                    "weight_gib": m.weight_gib,
                    "opt_in": m.opt_in,
                    "external": m.external(),
                    "running": self.host.running(m),
                    "healthy": self.host.healthy(m),
                    "inflight": self.inflight(index),
                })
            })
            .collect();
        serde_json::json!({
            "port": self.config.port,
            "pinned": self.pinned(),
            "mem_available_gib": (self.host.mem_available_gib() * 10.0).round() / 10.0,
            "reserve_gib": self.config.reserve_gib,
            "models": models,
        })
    }

    fn authed(&self, request: &Request) -> bool {
        self.config.auth_key.is_empty() || request.client_key() == self.config.auth_key
    }

    pub fn handle(&self, request: Request) -> Reply {
        if !self.authed(&request) {
            return Response::error(401, "authentication_error", "missing or wrong API key").into();
        }
        let path = request.path().trim_end_matches('/').to_string();
        if request.method == "GET" {
            if path == "/arbiter/status" {
                return Response::json(200, &self.status()).into();
            }
            if path == "/v1/models" {
                return Response::json(200, &self.model_list()).into();
            }
        }
        if request.method == "POST" {
            for verb in ["load", "unload", "pin"] {
                let prefix = format!("/arbiter/{verb}/");
                let Some(want) = path.strip_prefix(&prefix) else { continue };
                let want = want.trim_matches('/');
                if verb == "pin" {
                    let pinned = self.set_pin(want);
                    if pinned.is_empty() {
                        return Response::error(400, "invalid_request_error", &format!("unknown model {want:?}")).into();
                    }
                    return Response::json(200, &serde_json::json!({"pinned": pinned})).into();
                }
                let Some(index) = self.resolve(want) else {
                    return Response::json(
                        404,
                        &serde_json::json!({
                            "error": format!("unknown model {want:?}"),
                            "known": self.models.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
                        }),
                    )
                    .into();
                };
                if verb == "load" {
                    let outcome = self.ensure(index);
                    let status = if outcome.ok() { 200 } else { 503 };
                    let reply = Response::json(
                        status,
                        &serde_json::json!({
                            "model": self.models[index].id,
                            "ok": outcome.ok(),
                            "why": outcome.label(),
                        }),
                    );
                    return if status == 503 { reply.with_header("Retry-After", "30") } else { reply }.into();
                }
                return match self.unload(index) {
                    Ok(body) => Response::json(200, &body).into(),
                    Err(body) => Response::json(409, &body).into(),
                };
            }
        }
        self.forward(request)
    }

    fn forward(&self, request: Request) -> Reply {
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let want = body.get("model").and_then(Value::as_str).unwrap_or("").trim().to_string();

        let index = if want.is_empty() {
            match self.default_model() {
                Some(index) => index,
                None => return Response::error(503, "unavailable", "no default model").into(),
            }
        } else {
            match self.resolve(&want) {
                Some(index) => index,
                // An unknown name is a client error, not a reason to serve something
                // else under a name the client did not ask for (A-24). `known` sits
                // beside `error` rather than inside it, which is where the Python put
                // it: the shape is what a deployed client would have parsed, and this
                // is meant to be a drop-in for that service.
                None => {
                    return Response::json(
                        400,
                        &serde_json::json!({
                            "error": {
                                "type": "invalid_request",
                                "message": format!("unknown model {want:?}"),
                            },
                            "known": self.models.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
                        }),
                    )
                    .into();
                }
            }
        };

        let outcome = self.ensure(index);
        if !outcome.ok() {
            // Any "not now" is a 503 with Retry-After, never a hang (A-26).
            return Response::error(503, "unavailable", &outcome.label())
                .with_header("Retry-After", "30")
                .with_header("x-mind-model", &self.models[index].id)
                .with_header("x-mind-arbiter", &outcome.label())
                .into();
        }

        let model = &self.models[index];
        let _inflight = Inflight::acquire(&self.inflight[index]);
        let key = model.resolve_key();
        let mut headers = vec![(
            "Content-Type".to_string(),
            request.header("content-type").unwrap_or("application/json").to_string(),
        )];
        if !key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        let payload = if request.body.is_empty() { None } else { Some(request.body.as_slice()) };
        let upstream = http::fetch(
            &model.base,
            &request.target,
            &request.method,
            &headers,
            payload,
            self.config.upstream_timeout,
        );
        let Ok(up) = upstream else {
            return Response::error(502, "upstream_error", &format!("{} unreachable", model.id))
                .with_header("x-mind-model", &model.id)
                .into();
        };

        let status = up.status;
        let mut out = vec![
            ("Content-Type".to_string(), up.content_type().to_string()),
            ("x-mind-model".to_string(), model.id.clone()),
            ("x-mind-arbiter".to_string(), outcome.label()),
        ];
        if up.is_event_stream() {
            // The guard must outlive the stream, or the model becomes swappable while
            // it is still generating. Chaining it onto the reader keeps it alive for
            // exactly as long as the body is being copied.
            let source = GuardedReader { inner: up.into_reader(), _guard: _inflight };
            out.push(("x-mind-stream".to_string(), "1".to_string()));
            return Reply::Stream { status, headers: out, source: Box::new(source) };
        }
        let data = up.read_body().unwrap_or_default();
        Reply::Buffered(Response { status, headers: out, body: data })
    }
}

/// A reader that holds an in-flight guard for as long as it is being read.
struct GuardedReader {
    inner: Box<dyn Read + Send>,
    _guard: Inflight,
}

impl Read for GuardedReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buffer)
    }
}

impl<H: Host + 'static> Arbiter<H> {
    pub fn run(self: Arc<Self>) -> std::io::Result<()> {
        let address = format!("{}:{}", self.config.host, self.config.port);
        http::serve(address, move |request| self.handle(request))
    }

    /// Serve on an already-bound listener, so a test can learn the port first.
    pub fn run_on(self: Arc<Self>, listener: std::net::TcpListener) -> std::io::Result<()> {
        http::serve_on(listener, move |request| self.handle(request))
    }
}

// ---------------------------------------------------------------------------------
// The real host.
// ---------------------------------------------------------------------------------

/// Drives systemd, spawns processes, reads `/proc/meminfo`, and talks to engines.
pub struct SystemHost {
    config: Config,
    children: Mutex<HashMap<String, std::process::Child>>,
    probes: Mutex<HashMap<String, (bool, Instant)>>,
}

impl SystemHost {
    pub fn new(config: Config) -> Self {
        SystemHost {
            config,
            children: Mutex::new(HashMap::new()),
            probes: Mutex::new(HashMap::new()),
        }
    }

    /// `total_slots` from an engine's `/props`, or `None` if it cannot be asked.
    pub fn total_slots(&self, model: &Model) -> Option<usize> {
        let key = model.resolve_key();
        let mut headers = Vec::new();
        if !key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        let upstream =
            http::fetch(&model.base, "/props", "GET", &headers, None, Duration::from_secs(4)).ok()?;
        if !(200..300).contains(&upstream.status) {
            return None;
        }
        let body = upstream.read_body().ok()?;
        let doc: Value = serde_json::from_slice(&body).ok()?;
        doc.get("total_slots").and_then(Value::as_u64).map(|v| v as usize)
    }

    /// Warn about any model whose `slots` disagrees with its engine (A-17).
    ///
    /// Best-effort and advisory: a model that is down cannot be asked, and a mismatch
    /// is a configuration error to report rather than a reason to refuse to serve.
    pub fn report_slot_counts(&self, models: &[Model]) {
        for model in models {
            if model.engine != "llama.cpp" || model.slots == 0 || !self.healthy(model) {
                continue;
            }
            if let Some(warning) = slot_mismatch(model, self.total_slots(model)) {
                self.log(&format!("WARNING {warning}"));
            }
        }
    }

    fn systemctl(&self, unit: &str, action: &str) -> bool {
        Command::new("systemctl")
            .args(["--user", action, unit])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// One slot save or restore per slot, reporting cold separately from failed.
    fn slots(&self, model: &Model, action: &str) -> String {
        let key = model.resolve_key();
        let mut done = 0;
        let mut cold = 0;
        let mut failed = 0;
        for slot in 0..model.slots {
            let body = serde_json::json!({"filename": format!("{}-slot{slot}.bin", model.id)})
                .to_string()
                .into_bytes();
            let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
            if !key.is_empty() {
                headers.push(("Authorization".to_string(), format!("Bearer {key}")));
            }
            match http::fetch(
                &model.base,
                &format!("/slots/{slot}?action={action}"),
                "POST",
                &headers,
                Some(&body),
                Duration::from_secs(300),
            ) {
                Ok(up) if (200..300).contains(&up.status) => done += 1,
                // Restoring a slot that was never saved is the ordinary first case, and
                // llama-server reports it as 400 with the same message it uses for a
                // corrupt file. Counting that as a failure would make every first
                // swap-in look broken (A-18).
                Ok(up) if action == "restore" && up.status == 400 => cold += 1,
                _ => failed += 1,
            }
        }
        let mut parts = vec![format!("{action} {done}/{} ok", model.slots)];
        if cold > 0 {
            parts.push(format!("{cold} cold"));
        }
        if failed > 0 {
            parts.push(format!("{failed} failed"));
        }
        parts.join(", ")
    }
}

/// Does a model's configured `slots` match what the engine reports (A-17)?
///
/// Worth checking rather than trusting, because the failure is silent: `--parallel`
/// defaults to *auto*, which chose 4 on this box rather than 1, and a `slots` that is
/// merely plausible persists some slots and leaves the rest cold. Nothing else in the
/// system notices — swaps keep working, and only prefill time says anything is wrong.
/// `None` for `reported` means `/props` could not be read, which is not a mismatch.
pub fn slot_mismatch(model: &Model, reported: Option<usize>) -> Option<String> {
    if model.engine != "llama.cpp" || model.slots == 0 {
        return None;
    }
    match reported {
        Some(actual) if actual != model.slots => Some(format!(
            "{}: registry says slots={} but the engine reports total_slots={actual}; \
             {} slot(s) will never be saved or restored",
            model.id,
            model.slots,
            actual.abs_diff(model.slots),
        )),
        _ => None,
    }
}

/// `MemAvailable` in GiB, or 0.0 if it cannot be read (A-13).
pub fn mem_available_gib() -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0.0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:")
            && let Some(kb) = rest.split_whitespace().next()
                && let Ok(kb) = kb.parse::<f64>() {
                    return kb / 1_048_576.0;
                }
    }
    0.0
}

impl Host for SystemHost {
    fn running(&self, model: &Model) -> bool {
        match model.launcher {
            Launcher::External => true,
            Launcher::Systemd => model
                .unit
                .as_deref()
                .is_some_and(|unit| self.systemctl(unit, "is-active")),
            Launcher::Exec => {
                let mut children = self.children.lock().unwrap();
                match children.get_mut(&model.id) {
                    Some(child) => matches!(child.try_wait(), Ok(None)),
                    None => false,
                }
            }
        }
    }

    fn healthy(&self, model: &Model) -> bool {
        {
            let probes = self.probes.lock().unwrap();
            if let Some((ok, at)) = probes.get(&model.id)
                && at.elapsed() <= self.config.probe_ttl {
                    return *ok;
                }
        }
        let key = model.resolve_key();
        let mut headers = Vec::new();
        if !key.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        let ok = http::fetch(&model.base, "/v1/models", "GET", &headers, None, Duration::from_secs(4))
            .map(|up| (200..300).contains(&up.status))
            .unwrap_or(false);
        self.probes.lock().unwrap().insert(model.id.clone(), (ok, Instant::now()));
        ok
    }

    fn start(&self, model: &Model) -> bool {
        if model.external() {
            return self.healthy(model);
        }
        self.log(&format!("{}: starting ({:?})", model.id, model.launcher));
        match model.launcher {
            Launcher::Systemd => {
                let Some(unit) = model.unit.as_deref() else { return false };
                if !self.systemctl(unit, "start") {
                    return false;
                }
            }
            Launcher::Exec => {
                let Some((program, args)) = model.argv.split_first() else { return false };
                let log = model
                    .log
                    .as_deref()
                    .map(expand_home)
                    .and_then(|path| std::fs::File::create(path).ok());
                let (out, err) = match log {
                    Some(file) => match file.try_clone() {
                        Ok(clone) => (Stdio::from(file), Stdio::from(clone)),
                        Err(_) => (Stdio::null(), Stdio::null()),
                    },
                    None => (Stdio::null(), Stdio::null()),
                };
                let Ok(child) = Command::new(program).args(args).stdout(out).stderr(err).spawn()
                else {
                    return false;
                };
                self.children.lock().unwrap().insert(model.id.clone(), child);
            }
            Launcher::External => unreachable!("handled above"),
        }
        let deadline = Instant::now() + model.ready_timeout;
        while Instant::now() < deadline {
            self.probes.lock().unwrap().remove(&model.id);
            if self.healthy(model) {
                self.log(&format!("{}: ready", model.id));
                return true;
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        self.log(&format!("{}: not ready within {:.0}s", model.id, model.ready_timeout.as_secs_f64()));
        false
    }

    fn stop(&self, model: &Model) {
        if model.external() {
            return;
        }
        self.log(&format!("{}: stopping", model.id));
        match model.launcher {
            Launcher::Systemd => {
                if let Some(unit) = model.unit.as_deref() {
                    self.systemctl(unit, "stop");
                }
            }
            Launcher::Exec => {
                if let Some(mut child) = self.children.lock().unwrap().remove(&model.id) {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
            Launcher::External => {}
        }
        self.probes.lock().unwrap().insert(model.id.clone(), (false, Instant::now()));
    }

    fn mem_available_gib(&self) -> f64 {
        mem_available_gib()
    }

    fn save_kv(&self, model: &Model) -> String {
        // ds4-server checkpoints on its own once --kv-disk-dir is set, so there is
        // nothing to call (A-15).
        if model.engine == "ds4" {
            return "automatic".into();
        }
        if model.slots == 0 {
            return "unsupported".into(); // A-19
        }
        self.slots(model, "save")
    }

    fn restore_kv(&self, model: &Model) -> String {
        if model.engine == "ds4" {
            return "automatic".into();
        }
        if model.slots == 0 {
            return "unsupported".into();
        }
        self.slots(model, "restore")
    }

    fn notify_forget(&self, why: &str) {
        if self.config.router_base.is_empty() || self.config.router_backend.is_empty() {
            return;
        }
        let base = self.config.router_base.clone();
        let backend = self.config.router_backend.clone();
        let key = self.config.router_key.clone();
        let why = why.to_string();
        // Fire and forget: the swap has already happened by the time this runs, so
        // there is nothing a failure could undo, and a missed notification self-heals
        // because the index only ever corrects an estimate downward (A-14).
        std::thread::spawn(move || {
            let mut headers = Vec::new();
            if !key.is_empty() {
                headers.push(("Authorization".to_string(), format!("Bearer {key}")));
            }
            let path = format!("/router/forget/{backend}");
            let result = http::fetch(&base, &path, "POST", &headers, Some(b""), Duration::from_secs(4));
            let note = match result {
                Ok(up) if (200..300).contains(&up.status) => format!("router: forget {backend} ({why})"),
                Ok(up) => format!("router: forget {backend} refused: HTTP {}", up.status),
                Err(error) => format!("router: forget {backend} failed: {}", error.kind()),
            };
            log_line(&note);
        });
    }

    fn log(&self, message: &str) {
        log_line(message);
    }
}

fn log_line(message: &str) {
    use std::io::Write;
    // Seconds since the epoch rather than a formatted clock: date formatting is not
    // worth a dependency, and journald already timestamps every line.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "[mind-arbiter] t={now} {message}");
    let _ = out.flush();
}

/// Read configuration from the environment, matching the names the unit already uses.
pub fn config_from_env() -> Config {
    let env = |name: &str, fallback: &str| {
        std::env::var(name).map(|v| v.trim().to_string()).unwrap_or_else(|_| fallback.to_string())
    };
    let secs = |name: &str, fallback: f64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or_else(|| Duration::from_secs_f64(fallback), Duration::from_secs_f64)
    };
    let auth_key = {
        let configured = env("MIND_ARBITER_KEY", "");
        if configured.is_empty() {
            std::fs::read_to_string(std::path::Path::new(&env("HOME", "")).join(".ds4-lan-key"))
                .map(|v| v.trim().to_string())
                .unwrap_or_default()
        } else {
            configured
        }
    };
    let router_key = {
        let own = env("MIND_ARBITER_ROUTER_KEY", "");
        if own.is_empty() { auth_key.clone() } else { own }
    };
    Config {
        host: env("MIND_ARBITER_HOST", "127.0.0.1"),
        port: env("MIND_ARBITER_PORT", "8010").parse().unwrap_or(8010),
        auth_key,
        reserve_gib: env("MIND_ARBITER_RESERVE_GIB", "8").parse().unwrap_or(8.0),
        swap_wait: secs("MIND_ARBITER_SWAP_WAIT", 1200.0),
        drain_wait: secs("MIND_ARBITER_DRAIN_WAIT", 120.0),
        probe_ttl: secs("MIND_ARBITER_PROBE_TTL", 10.0),
        upstream_timeout: secs("MIND_ARBITER_UPSTREAM_TIMEOUT", 1800.0),
        pinned: env("MIND_ARBITER_PIN", "none"),
        router_base: env("MIND_ARBITER_ROUTER", "").trim_end_matches('/').to_string(),
        router_backend: env("MIND_ARBITER_ROUTER_BACKEND", ""),
        router_key,
    }
}

pub fn load_registry(path: &str) -> Result<Vec<Model>, String> {
    let text = std::fs::read_to_string(expand_home(path)).map_err(|e| format!("{path}: {e}"))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    parse_registry(&doc).map_err(|e| format!("{path}: {e}"))
}
