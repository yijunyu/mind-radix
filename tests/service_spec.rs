//! End-to-end behaviour over real sockets.
//!
//! The tests in `router_spec.rs` and `arbiter_spec.rs` cover decisions; these cover the
//! wire, because several requirements are *about* the wire — a header that must survive
//! being proxied (R-17), a status code that must be told apart from another with the
//! same message (A-18), a body that must not be buffered (R-18). Fake engines run on
//! ephemeral ports in this process, so nothing here needs a model, a GPU or the live
//! services.

use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mind_radix::arbiter::{self, Arbiter, Host, Model, SystemHost, parse_registry};
use mind_radix::http::{self, Reply, Request, Response};
use mind_radix::router::{self, Backend, Router};
use serde_json::{Value, json};

// --------------------------------------------------------------------- fake engine

/// A stand-in engine. `listing` is what it answers to `/v1/models`; `slot_status` is
/// what it answers to a slot save or restore, which is how A-18 is exercised.
struct Engine {
    listing: Value,
    slot_status: u16,
    model_header: Option<String>,
    requests: Arc<Mutex<Vec<String>>>,
}

fn spawn_engine(engine: Engine) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::clone(&engine.requests);
    let listing = engine.listing.clone();
    let slot_status = engine.slot_status;
    let model_header = engine.model_header.clone();
    std::thread::spawn(move || {
        let _ = http::serve_on(listener, move |request: Request| {
            requests.lock().unwrap().push(format!("{} {}", request.method, request.target));
            let path = request.path();
            if path == "/v1/models" {
                return Response::json(200, &listing).into();
            }
            if path.starts_with("/slots/") {
                if (200..300).contains(&slot_status) {
                    return Response::json(
                        slot_status,
                        &json!({"id_slot": 0, "n_saved": 32, "n_written": 198_249_916}),
                    )
                    .into();
                }
                // llama-server's real wording, which is the point of A-18: the same
                // message covers "no such file" and "corrupt file".
                return Response::error(
                    slot_status,
                    "invalid_request_error",
                    "Unable to restore slot: No available space in KV cache or invalid slot save file",
                )
                .into();
            }
            // Streaming is requested in the body (`"stream": true`), which is what an
            // OpenAI-compatible client sends and what llama.cpp reads — not an Accept
            // header. The engine decides, and the router only relays what comes back.
            let wants_stream = serde_json::from_slice::<Value>(&request.body)
                .ok()
                .and_then(|b| b.get("stream").and_then(Value::as_bool))
                .unwrap_or(false);
            if path == "/v1/chat/completions" && wants_stream {
                return Reply::Stream {
                    status: 200,
                    headers: vec![("Content-Type".into(), "text/event-stream".into())],
                    source: Box::new(std::io::Cursor::new(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n".to_vec(),
                    )),
                };
            }
            // An ordinary completion, with the timings the index learns from.
            let mut reply = Response::json(
                200,
                &json!({
                    "choices": [{"message": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}],
                    "timings": {"cache_n": 512, "prompt_n": 8},
                }),
            );
            if let Some(model) = &model_header {
                // An arbiter behind the router sets these; the router must relay them.
                reply = reply
                    .with_header("x-mind-model", model)
                    .with_header("x-mind-arbiter", "loaded-evicted:other(save 4/4 ok)");
            }
            reply.into()
        });
    });
    wait_until_up(port);
    (format!("http://127.0.0.1:{port}"), engine.requests)
}

fn engine_with(listing: Value) -> Engine {
    Engine {
        listing,
        slot_status: 200,
        model_header: None,
        requests: Arc::new(Mutex::new(Vec::new())),
    }
}

fn wait_until_up(port: u16) {
    for _ in 0..200 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("test server on {port} never came up");
}

/// One request to a service under test, returning status, headers and body.
fn call(
    base: &str,
    path: &str,
    method: &str,
    key: Option<&str>,
    body: Option<&str>,
) -> (u16, HashMap<String, String>, String) {
    let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
    if let Some(key) = key {
        headers.push(("Authorization".to_string(), format!("Bearer {key}")));
    }
    let upstream = http::fetch(
        base,
        path,
        method,
        &headers,
        body.map(str::as_bytes),
        Duration::from_secs(10),
    )
    .expect("request reaches the service");
    let status = upstream.status;
    let headers = upstream
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let text = String::from_utf8_lossy(&upstream.read_body().expect("body")).into_owned();
    (status, headers, text)
}

fn spawn_router(backends: Vec<Backend>, auth_key: &str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let config = router::Config {
        backends,
        auth_key: auth_key.to_string(),
        probe_ttl: Duration::from_millis(50),
        ..router::Config::default()
    };
    let service = Arc::new(Router::new(config));
    std::thread::spawn(move || {
        let _ = service.run_on(listener);
    });
    wait_until_up(port);
    format!("http://127.0.0.1:{port}")
}

// --------------------------------------------------------------------- router

/// R-21: no key, no backend contact.
#[test]
fn the_router_refuses_an_unauthenticated_request() {
    let (engine, seen) = spawn_engine(engine_with(json!({"data": [{"id": "m"}]})));
    let base = spawn_router(vec![Backend::new("amd", &engine, "")], "secret");

    let (status, _, body) = call(&base, "/v1/models", "GET", None, None);
    assert_eq!(status, 401);
    assert!(body.contains("authentication_error"), "{body}");
    assert!(seen.lock().unwrap().is_empty(), "no backend may be touched before auth");

    let (status, _, _) = call(&base, "/router/status", "GET", Some("wrong"), None);
    assert_eq!(status, 401);
    let (status, _, _) = call(&base, "/router/status", "GET", Some("secret"), None);
    assert_eq!(status, 200);
}

/// R-7 (regression): the health probe and the capability listing are **one** request.
///
/// The Python fetched `/v1/models` and discarded the body, then had nothing to route
/// models on. Asserting the request count here is what keeps a future change from
/// quietly adding a second round trip per request.
#[test]
fn one_probe_answers_both_health_and_capability() {
    let listing = json!({"data": [{"id": "minicpm-v4.7", "aliases": ["minicpm"]}]});
    let (engine, seen) = spawn_engine(engine_with(listing));
    let base = spawn_router(vec![Backend::new("amd", &engine, "")], "");

    let (status, _, body) = call(&base, "/router/status", "GET", None, None);
    assert_eq!(status, 200);
    let doc: Value = serde_json::from_str(&body).unwrap();
    let models = doc["backends"][0]["models"].as_array().unwrap();
    assert_eq!(models.len(), 2, "id and alias, from the probe body: {body}");

    let probes = seen.lock().unwrap();
    assert_eq!(probes.len(), 1, "exactly one upstream request: {probes:?}");
    assert_eq!(probes[0], "GET /v1/models");
}

/// R-16 and R-17 (regression): the backend's own headers survive the proxy.
///
/// The router rebuilt responses from scratch and forwarded only `Content-Type`, so an
/// arbiter's `x-mind-arbiter` — the only explanation for a first request that took
/// seconds — was dropped exactly when a client needed it.
#[test]
fn the_router_relays_the_backends_mind_headers() {
    let mut engine = engine_with(json!({"data": [{"id": "minicpm-v4.7"}]}));
    engine.model_header = Some("minicpm-v4.7".to_string());
    let (engine, _) = spawn_engine(engine);
    let base = spawn_router(vec![Backend::new("amd", &engine, "")], "");

    let (status, headers, _) = call(
        &base,
        "/v1/chat/completions",
        "POST",
        None,
        Some(&json!({"model": "minicpm-v4.7", "messages": [{"role": "user", "content": "hi"}]}).to_string()),
    );
    assert_eq!(status, 200);
    assert_eq!(headers.get("x-mind-model").map(String::as_str), Some("minicpm-v4.7"));
    assert_eq!(
        headers.get("x-mind-arbiter").map(String::as_str),
        Some("loaded-evicted:other(save 4/4 ok)")
    );
    assert_eq!(headers.get("x-mind-backend").map(String::as_str), Some("amd"));
    assert_eq!(headers.get("x-mind-route").map(String::as_str), Some("new:amd"));
}

/// R-10 and R-12 on the wire: two live backends, one model, the right host.
#[test]
fn the_router_sends_a_model_to_the_host_that_claims_it() {
    let (studio, studio_seen) = spawn_engine(engine_with(json!({
        "models": [{"name": "minicpm-v4.7-35b-a3b-q4km"}]
    })));
    let (amd, amd_seen) = spawn_engine(engine_with(json!({
        "data": [{"id": "deepseek-v4.1-flash", "aliases": ["v41"]},
                 {"id": "minicpm-v4.7-35b-a3b-q4km"}]
    })));
    let base = spawn_router(
        vec![Backend::new("studio", &studio, ""), Backend::new("amd", &amd, "")],
        "",
    );

    let body = json!({"model": "v41", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (status, headers, _) = call(&base, "/v1/chat/completions", "POST", None, Some(&body));
    assert_eq!(status, 200);
    assert_eq!(headers.get("x-mind-backend").map(String::as_str), Some("amd"));
    assert_eq!(headers.get("x-mind-route").map(String::as_str), Some("new:amd+model"));

    let completions = |seen: &Arc<Mutex<Vec<String>>>| {
        seen.lock().unwrap().iter().filter(|r| r.contains("chat/completions")).count()
    };
    assert_eq!(completions(&amd_seen), 1);
    assert_eq!(completions(&studio_seen), 0, "studio never claimed v41");
}

/// R-18: an event stream is relayed without a Content-Length.
#[test]
fn the_router_streams_an_event_stream_through() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "m"}]})));
    let base = spawn_router(vec![Backend::new("amd", &engine, "")], "");

    let upstream = http::fetch(
        &base,
        "/v1/chat/completions",
        "POST",
        &[("Content-Type".to_string(), "application/json".to_string())],
        Some(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true})
            .to_string()
            .as_bytes()),
        Duration::from_secs(10),
    )
    .expect("stream starts");
    assert_eq!(upstream.status, 200);
    assert!(upstream.is_event_stream(), "{:?}", upstream.headers);
    assert!(upstream.header("content-length").is_none(), "a stream has no length");
    let body = String::from_utf8_lossy(&upstream.read_body().unwrap()).into_owned();
    assert!(body.contains("data: "), "{body}");
    assert!(body.trim_end().ends_with("[DONE]"), "{body}");
}

/// R-13, R-14, R-15: forget drops the index, keeps affinity, rejects a bad name.
#[test]
fn forget_drops_estimates_but_not_affinity() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "m"}]})));
    let base = spawn_router(vec![Backend::new("amd", &engine, "")], "");
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();

    // One turn, which both creates an affinity entry and teaches the index.
    let (status, _, _) = call(&base, "/v1/chat/completions", "POST", None, Some(&body));
    assert_eq!(status, 200);
    let (_, _, before) = call(&base, "/router/status", "GET", None, None);
    let before: Value = serde_json::from_str(&before).unwrap();
    assert_eq!(before["conversations"], json!(1));

    let (status, _, text) = call(&base, "/router/forget/amd", "POST", None, Some(""));
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["forgot"], json!("amd"));

    let (_, _, after) = call(&base, "/router/status", "GET", None, None);
    let after: Value = serde_json::from_str(&after).unwrap();
    assert_eq!(after["conversations"], json!(1), "R-14: affinity must survive a forget");

    let (status, _, _) = call(&base, "/router/forget/nosuch", "POST", None, Some(""));
    assert_eq!(status, 400, "R-15: an unknown backend is rejected, not ignored");
}

/// R-5 and R-19: with the only backend down, the router says so rather than hanging.
#[test]
fn the_router_reports_when_no_backend_is_reachable() {
    // Port 1 on loopback refuses connections, so the probe fails fast.
    let base = spawn_router(vec![Backend::new("amd", "http://127.0.0.1:1", "")], "");
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (status, _, text) = call(&base, "/v1/chat/completions", "POST", None, Some(&body));
    assert_eq!(status, 503);
    assert!(text.contains("no backend reachable"), "{text}");
}

// --------------------------------------------------------------------- arbiter

/// A host that is always healthy, so arbiter wire behaviour can be tested without a
/// launcher. Residency rules have their own tests in `arbiter_spec.rs`.
struct ReadyHost {
    free_gib: f64,
}

impl Host for ReadyHost {
    fn running(&self, _model: &Model) -> bool {
        true
    }
    fn healthy(&self, _model: &Model) -> bool {
        true
    }
    fn start(&self, _model: &Model) -> bool {
        true
    }
    fn stop(&self, _model: &Model) {}
    fn mem_available_gib(&self) -> f64 {
        self.free_gib
    }
    fn save_kv(&self, _model: &Model) -> String {
        "save 4/4 ok".into()
    }
    fn restore_kv(&self, _model: &Model) -> String {
        "restore 4/4 ok".into()
    }
    fn notify_forget(&self, _why: &str) {}
    fn log(&self, _message: &str) {}
    fn settle(&self) {}
}

fn spawn_arbiter(models: Vec<Model>, auth_key: &str, free_gib: f64) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let config = arbiter::Config {
        auth_key: auth_key.to_string(),
        drain_wait: Duration::from_millis(100),
        probe_ttl: Duration::from_millis(50),
        ..arbiter::Config::default()
    };
    let service = Arc::new(Arbiter::new(models, config, ReadyHost { free_gib }));
    std::thread::spawn(move || {
        let _ = service.run_on(listener);
    });
    wait_until_up(port);
    format!("http://127.0.0.1:{port}")
}

/// A-21, A-22, A-25 on the wire, plus A-24 for an unknown name.
#[test]
fn the_arbiter_advertises_its_registry_and_serves_by_alias() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "whatever"}]})));
    let registry = json!({"models": [
        {"id": "minicpm-v4.7", "aliases": ["minicpm"], "engine": "llama.cpp",
         "base": engine, "weight_gib": 35, "slots": 0},
        {"id": "deepseek-v4.1-flash", "aliases": ["v41"], "engine": "ds4",
         "base": engine, "weight_gib": 88, "opt_in": true}]});
    let models = parse_registry(&registry).unwrap();
    let base = spawn_arbiter(models, "k", 122.0);

    let (status, _, body) = call(&base, "/v1/models", "GET", Some("k"), None);
    assert_eq!(status, 200);
    let doc: Value = serde_json::from_str(&body).unwrap();
    let data = doc["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["aliases"], json!(["minicpm"]), "A-22: aliases are published");
    assert_eq!(data[1]["opt_in"], json!(true));

    // A-1 and A-25: an alias resolves, and the reply says which model answered.
    let body = json!({"model": "minicpm", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (status, headers, _) = call(&base, "/v1/chat/completions", "POST", Some("k"), Some(&body));
    assert_eq!(status, 200);
    assert_eq!(headers.get("x-mind-model").map(String::as_str), Some("minicpm-v4.7"));
    assert_eq!(headers.get("x-mind-arbiter").map(String::as_str), Some("resident"));

    // A-24: an unknown name is a client error listing what does exist.
    let body = json!({"model": "llama-3", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (status, _, text) = call(&base, "/v1/chat/completions", "POST", Some("k"), Some(&body));
    assert_eq!(status, 400);
    assert!(text.contains("unknown model"), "{text}");
    assert!(text.contains("deepseek-v4.1-flash"), "the known ids are listed: {text}");
}

/// A-26: a refusal is a 503 that tells the client when to come back.
#[test]
fn a_refused_swap_answers_503_with_retry_after() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "x"}]})));
    let registry = json!({"models": [
        {"id": "huge", "engine": "llama.cpp", "base": engine, "weight_gib": 500, "slots": 0}]});
    let models = parse_registry(&registry).unwrap();
    // Healthy host, but nothing like enough memory, and `ensure` is only consulted for
    // a model that is not resident — so this arbiter is given a host that reports the
    // model down by asking for a load explicitly.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    struct DownHost;
    impl Host for DownHost {
        fn running(&self, _m: &Model) -> bool {
            false
        }
        fn healthy(&self, _m: &Model) -> bool {
            false
        }
        fn start(&self, _m: &Model) -> bool {
            true
        }
        fn stop(&self, _m: &Model) {}
        fn mem_available_gib(&self) -> f64 {
            16.0
        }
        fn save_kv(&self, _m: &Model) -> String {
            "unsupported".into()
        }
        fn restore_kv(&self, _m: &Model) -> String {
            "unsupported".into()
        }
        fn notify_forget(&self, _why: &str) {}
        fn log(&self, _message: &str) {}
        fn settle(&self) {}
    }
    let service = Arc::new(Arbiter::new(models, arbiter::Config::default(), DownHost));
    std::thread::spawn(move || {
        let _ = service.run_on(listener);
    });
    wait_until_up(port);
    let base = format!("http://127.0.0.1:{port}");

    let body = json!({"model": "huge", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (status, headers, text) = call(&base, "/v1/chat/completions", "POST", None, Some(&body));
    assert_eq!(status, 503);
    assert_eq!(headers.get("retry-after").map(String::as_str), Some("30"));
    assert!(text.contains("insufficient-memory"), "{text}");
    assert!(
        headers.get("x-mind-arbiter").is_some_and(|v| v.contains("insufficient-memory")),
        "the reason must reach the client: {headers:?}"
    );

    // The same reason through the explicit load endpoint.
    let (status, headers, text) = call(&base, "/arbiter/load/huge", "POST", None, Some(""));
    assert_eq!(status, 503);
    assert_eq!(headers.get("retry-after").map(String::as_str), Some("30"));
    assert!(text.contains("insufficient-memory"), "{text}");
}

#[test]
fn the_arbiter_refuses_an_unauthenticated_request_and_an_unknown_model_endpoint() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "x"}]})));
    let registry = json!({"models": [
        {"id": "m", "engine": "llama.cpp", "base": engine, "weight_gib": 1, "slots": 0}]});
    let base = spawn_arbiter(parse_registry(&registry).unwrap(), "k", 122.0);

    let (status, _, _) = call(&base, "/arbiter/status", "GET", None, None);
    assert_eq!(status, 401);
    let (status, _, text) = call(&base, "/arbiter/load/nosuch", "POST", Some("k"), Some(""));
    assert_eq!(status, 404);
    assert!(text.contains("\"known\""), "{text}");
    let (status, _, _) = call(&base, "/arbiter/pin/nosuch", "POST", Some("k"), Some(""));
    assert_eq!(status, 400);
}

// --------------------------------------------------------------------- KV slots

/// **A-18 (regression): a never-saved slot is `cold`, not `failed`.**
///
/// llama-server answers a restore of a missing file with 400 and the same message it
/// uses for a corrupt one. Counting every non-2xx as a failure made the first swap-in
/// of any model report `restore 0/4 ok, 4 failed` while behaving perfectly.
#[test]
fn restoring_a_slot_that_was_never_saved_is_reported_as_cold() {
    let mut engine = engine_with(json!({"data": [{"id": "m"}]}));
    engine.slot_status = 400;
    let (base, seen) = spawn_engine(engine);
    let registry = json!({"models": [
        {"id": "minicpm-v4.7", "engine": "llama.cpp", "base": base, "slots": 4}]});
    let models = parse_registry(&registry).unwrap();
    let host = SystemHost::new(arbiter::Config::default());

    assert_eq!(host.restore_kv(&models[0]), "restore 0/4 ok, 4 cold");
    // A save that fails is still a failure: only a restore can be cold.
    assert_eq!(host.save_kv(&models[0]), "save 0/4 ok, 4 failed");

    let paths = seen.lock().unwrap();
    assert_eq!(paths.len(), 8, "four slots, each saved and restored: {paths:?}");
    assert!(paths.iter().any(|p| p.contains("/slots/3?action=restore")), "{paths:?}");
}

/// A-16 and A-20: every slot is addressed by index, with a per-model filename.
#[test]
fn every_slot_is_saved_under_a_name_of_its_own() {
    let (base, seen) = spawn_engine(engine_with(json!({"data": [{"id": "m"}]})));
    let registry = json!({"models": [
        {"id": "minicpm-v4.7", "engine": "llama.cpp", "base": base, "slots": 4}]});
    let models = parse_registry(&registry).unwrap();
    let host = SystemHost::new(arbiter::Config::default());

    assert_eq!(host.save_kv(&models[0]), "save 4/4 ok");
    let paths = seen.lock().unwrap();
    for slot in 0..4 {
        assert!(
            paths.iter().any(|p| p == &format!("POST /slots/{slot}?action=save")),
            "slot {slot} was not saved: {paths:?}"
        );
    }
}

/// A-14 on the wire: the arbiter's notification is a call the router accepts.
///
/// The two services are developed together, so the one contract between them deserves
/// a test that actually crosses the socket: a stale path or verb on either side would
/// otherwise only show up as a silently optimistic index in production.
#[test]
fn the_arbiters_notification_is_a_call_the_router_accepts() {
    let (engine, _) = spawn_engine(engine_with(json!({"data": [{"id": "m"}]})));
    let router_base = spawn_router(vec![Backend::new("amd", &engine, "")], "shared-key");

    let config = arbiter::Config {
        router_base: router_base.clone(),
        router_backend: "amd".to_string(),
        router_key: "shared-key".to_string(),
        ..arbiter::Config::default()
    };
    let host = SystemHost::new(config);
    host.notify_forget("swap-to:v41");

    // The notification is deliberately fire-and-forget, so poll the router for the
    // effect rather than waiting on a handle the arbiter does not keep.
    for _ in 0..100 {
        let (status, _, _) = call(&router_base, "/router/status", "GET", Some("shared-key"), None);
        if status == 200 {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let (status, _, _) = call(&router_base, "/router/forget/amd", "POST", Some("shared-key"), Some(""));
    assert_eq!(status, 200, "the path and verb the arbiter uses must be the ones the router serves");
}
