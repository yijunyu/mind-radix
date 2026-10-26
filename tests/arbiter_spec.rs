//! Swap, eviction and registry rules, against `SPEC.md` §3.
//!
//! Every rule here used to be checkable only by evicting a 35 GiB model on the real
//! box, which takes seconds when it works and forty minutes when the model being loaded
//! streams from disk. [`Host`] puts all of that behind a trait, so a fake machine
//! stands in: memory that actually drops when a model stops, a start that can be made
//! to fail, and a record of every call. The rules are then exhaustively testable in
//! milliseconds.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mind_radix::arbiter::{
    Arbiter, Config, Host, Launcher, Model, Outcome, SystemHost, parse_registry, slot_mismatch,
};
use serde_json::{Value, json};

/// A machine that behaves like the real one in the respects the rules care about:
/// stopping a model frees its weight, starting it consumes it.
struct FakeHost {
    pool_gib: f64,
    running: Mutex<HashMap<String, bool>>,
    calls: Mutex<Vec<String>>,
    start_fails: Mutex<bool>,
    /// Highest number of threads inside `start` at once, for A-11.
    concurrent: AtomicUsize,
    peak_concurrent: AtomicUsize,
    /// `true` makes `mem_available_gib` report nothing readable, for A-13.
    memory_unknown: bool,
}

impl FakeHost {
    fn new(pool_gib: f64, running: &[&str]) -> Self {
        FakeHost {
            pool_gib,
            running: Mutex::new(running.iter().map(|id| ((*id).to_string(), true)).collect()),
            calls: Mutex::new(Vec::new()),
            start_fails: Mutex::new(false),
            concurrent: AtomicUsize::new(0),
            peak_concurrent: AtomicUsize::new(0),
            memory_unknown: false,
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn weights() -> HashMap<String, f64> {
        HashMap::from([
            ("minicpm".to_string(), 35.0),
            ("v41".to_string(), 88.0),
            ("small".to_string(), 10.0),
            ("studio".to_string(), 0.0),
        ])
    }
}

impl Host for FakeHost {
    fn running(&self, model: &Model) -> bool {
        *self.running.lock().unwrap().get(&model.id).unwrap_or(&false)
    }

    fn healthy(&self, model: &Model) -> bool {
        self.running(model)
    }

    fn start(&self, model: &Model) -> bool {
        let now = self.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_concurrent.fetch_max(now, Ordering::SeqCst);
        self.calls.lock().unwrap().push(format!("start:{}", model.id));
        std::thread::sleep(Duration::from_millis(20));
        let ok = !*self.start_fails.lock().unwrap();
        if ok {
            self.running.lock().unwrap().insert(model.id.clone(), true);
        }
        self.concurrent.fetch_sub(1, Ordering::SeqCst);
        ok
    }

    fn stop(&self, model: &Model) {
        self.calls.lock().unwrap().push(format!("stop:{}", model.id));
        self.running.lock().unwrap().insert(model.id.clone(), false);
    }

    fn mem_available_gib(&self) -> f64 {
        if self.memory_unknown {
            return 0.0; // A-13: unreadable means no headroom claim
        }
        let weights = Self::weights();
        let used: f64 = self
            .running
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, up)| **up)
            .map(|(id, _)| weights.get(id).copied().unwrap_or(0.0))
            .sum();
        self.pool_gib - used
    }

    fn save_kv(&self, model: &Model) -> String {
        self.calls.lock().unwrap().push(format!("save:{}", model.id));
        "save 4/4 ok".to_string()
    }

    fn restore_kv(&self, model: &Model) -> String {
        self.calls.lock().unwrap().push(format!("restore:{}", model.id));
        "restore 4/4 ok".to_string()
    }

    fn notify_forget(&self, why: &str) {
        self.calls.lock().unwrap().push(format!("notify:{why}"));
    }

    fn log(&self, _message: &str) {}

    /// No sleeping in tests; the real host waits for the pool to settle.
    fn settle(&self) {}
}

fn registry() -> Value {
    json!({"models": [
        {"id": "minicpm", "aliases": ["mini", "minicpm-v4.7"], "engine": "llama.cpp",
         "launcher": "systemd", "unit": "minicpm.service", "base": "http://127.0.0.1:8000",
         "weight_gib": 35, "slots": 4},
        {"id": "v41", "aliases": ["ds41"], "engine": "ds4", "launcher": "exec",
         "argv": ["/bin/true"], "base": "http://127.0.0.1:8001",
         "weight_gib": 88, "opt_in": true},
        {"id": "small", "engine": "llama.cpp", "launcher": "systemd", "unit": "small.service",
         "base": "http://127.0.0.1:8002", "weight_gib": 10},
        {"id": "studio", "engine": "llama.cpp", "external": true,
         "base": "http://192.168.1.92:8015", "weight_gib": 0}
    ]})
}

fn models() -> Vec<Model> {
    parse_registry(&registry()).expect("registry parses")
}

fn config() -> Config {
    Config {
        reserve_gib: 8.0,
        drain_wait: Duration::from_millis(200),
        swap_wait: Duration::from_secs(5),
        ..Config::default()
    }
}

fn arbiter(pool_gib: f64, running: &[&str]) -> Arbiter<FakeHost> {
    Arbiter::new(models(), config(), FakeHost::new(pool_gib, running))
}

// ---------------------------------------------------------------- the registry

/// A-1: a model answers to its id and every alias, case-insensitively.
#[test]
fn a_model_is_addressable_by_id_and_alias() {
    let a = arbiter(122.0, &[]);
    assert_eq!(a.resolve("minicpm"), Some(0));
    assert_eq!(a.resolve("mini"), Some(0));
    assert_eq!(a.resolve("MiniCPM-V4.7"), Some(0));
    assert_eq!(a.resolve("ds41"), Some(1));
    assert_eq!(a.resolve("nope"), None);
}

/// A-6: the default is the first non-`opt_in` entry, so order matters.
#[test]
fn the_default_model_skips_opt_in_entries() {
    let a = arbiter(122.0, &[]);
    assert_eq!(a.default_model(), Some(0));

    let opt_in_first = json!({"models": [
        {"id": "v41", "opt_in": true, "base": "http://x:1", "weight_gib": 88},
        {"id": "minicpm", "base": "http://x:2", "weight_gib": 35}]});
    let models = parse_registry(&opt_in_first).unwrap();
    let a = Arbiter::new(models, config(), FakeHost::new(122.0, &[]));
    assert_eq!(a.default_model(), Some(1), "an opt_in model is never the default");
}

/// A-4: `external` implies the arbiter owns nothing about its lifecycle.
#[test]
fn an_external_model_is_not_launched_locally() {
    let models = models();
    let studio = models.iter().find(|m| m.id == "studio").unwrap();
    assert!(studio.external());
    assert_eq!(studio.launcher, Launcher::External);
}

/// A-2: a registry carries no key, only where one lives.
#[test]
fn a_registry_entry_names_where_its_key_lives() {
    let doc = json!({"models": [{"id": "m", "base": "http://x:1",
        "key_env": "SPEC_TEST_KEY", "key_env_file": "/nonexistent"}]});
    let models = parse_registry(&doc).unwrap();
    assert_eq!(models[0].resolve_key(), "", "no variable set, no file: no key, no panic");
    // SAFETY: this variable is unique to this test.
    unsafe { std::env::set_var("SPEC_TEST_KEY", "from-env") };
    assert_eq!(models[0].resolve_key(), "from-env");
    unsafe { std::env::remove_var("SPEC_TEST_KEY") };
}

#[test]
fn a_key_is_read_from_a_key_equals_value_file() {
    let dir = std::env::temp_dir().join(format!("mind-spec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lan-service.env");
    std::fs::write(&path, "# comment\nSPEC_FILE_KEY='quoted-value'\n").unwrap();
    let doc = json!({"models": [{"id": "m", "base": "http://x:1",
        "key_env": "SPEC_FILE_KEY", "key_env_file": path.to_str().unwrap()}]});
    let models = parse_registry(&doc).unwrap();
    assert_eq!(models[0].resolve_key(), "quoted-value", "quotes are stripped");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_empty_or_malformed_registry_is_an_error_not_a_panic() {
    assert!(parse_registry(&json!({"models": []})).is_err());
    assert!(parse_registry(&json!({"models": [{"no_id": true}]})).is_err());
    assert!(parse_registry(&json!(42)).is_err());
}

// ---------------------------------------------------------------- residency

/// A-7: already resident means no swap at all.
#[test]
fn a_resident_model_is_served_without_a_swap() {
    let a = arbiter(122.0, &["minicpm"]);
    assert_eq!(a.ensure(0), Outcome::Resident);
    assert!(a.host_calls().is_empty(), "nothing was started or stopped");
}

/// A-9: models that genuinely fit together both stay up.
#[test]
fn loading_into_free_memory_evicts_nothing() {
    // small (10) alongside minicpm (35) in a 122 GiB pool: 122-35-8 = 79 free.
    let a = arbiter(122.0, &["minicpm"]);
    let outcome = a.ensure(2);
    assert_eq!(outcome, Outcome::Loaded { evicted: vec![], kv: "restore 4/4 ok".into() });
    assert_eq!(outcome.label(), "loaded");
    let calls = a.host_calls();
    assert!(calls.contains(&"start:small".to_string()));
    assert!(!calls.iter().any(|c| c.starts_with("stop:")), "{calls:?}");
}

/// A-14 (regression): loading into free memory must not tell the router to forget.
///
/// Nothing was evicted, so no cache died. Notifying anyway would throw away the
/// router's estimates for this host on every ordinary load.
#[test]
fn a_load_without_eviction_does_not_notify_the_router() {
    let a = arbiter(122.0, &["minicpm"]);
    a.ensure(2);
    assert!(
        !a.host_calls().iter().any(|c| c.starts_with("notify:")),
        "{:?}",
        a.host_calls()
    );
}

/// A-9 and A-14: V4.1 cannot fit beside MiniCPM, so MiniCPM is drained, saved, stopped,
/// and the router is told the cache is gone.
#[test]
fn an_incoming_model_that_does_not_fit_evicts_the_resident_one() {
    let a = arbiter(122.0, &["minicpm"]);
    let outcome = a.ensure(1);
    match &outcome {
        Outcome::Loaded { evicted, .. } => {
            assert_eq!(evicted, &vec!["minicpm(save 4/4 ok)".to_string()]);
        }
        other => panic!("expected a load, got {other:?}"),
    }
    assert_eq!(outcome.label(), "loaded-evicted:minicpm(save 4/4 ok)");
    let calls = a.host_calls();
    let order: Vec<&String> = calls.iter().filter(|c| !c.starts_with("notify")).collect();
    assert_eq!(
        order,
        vec!["save:minicpm", "stop:minicpm", "start:v41", "restore:v41"],
        "KV must be saved before the stop, and restored after the start"
    );
    assert!(calls.contains(&"notify:swap-to:v41".to_string()));
}

/// A-9: eviction is minimal — it stops one model and no more than it must.
#[test]
fn eviction_stops_as_soon_as_the_incoming_model_fits() {
    // Pool 122, resident small (10) + minicpm (35) = 77 free, 69 after reserve.
    // v41 needs 88, so freeing minicpm alone (104 free, 96 usable) is enough.
    let a = arbiter(122.0, &["small", "minicpm"]);
    let outcome = a.ensure(1);
    assert!(outcome.ok(), "{outcome:?}");
    let stopped: Vec<String> = a
        .host_calls()
        .into_iter()
        .filter(|c| c.starts_with("stop:"))
        .collect();
    assert_eq!(stopped, vec!["stop:minicpm"], "small was not touched: it was not needed");
}

/// A-4: an external model is never stopped to make room, and never counted.
#[test]
fn an_external_model_is_never_evicted() {
    let a = arbiter(122.0, &["minicpm", "studio"]);
    let outcome = a.ensure(1);
    assert!(outcome.ok(), "{outcome:?}");
    assert!(
        !a.host_calls().contains(&"stop:studio".to_string()),
        "another host owns studio's lifecycle"
    );
}

/// A-10 (regression): a pinned model is not evicted; the swap fails instead.
///
/// This is the guard that turns a stray request for a 40-minute model into an instant
/// refusal rather than an outage on the model that is actually serving.
#[test]
fn a_pinned_model_blocks_a_swap_instead_of_being_evicted() {
    let a = arbiter(122.0, &["minicpm"]);
    assert_eq!(a.set_pin("minicpm"), "minicpm");
    let outcome = a.ensure(1);
    assert_eq!(outcome, Outcome::BlockedByPin("minicpm".into()));
    assert_eq!(outcome.label(), "blocked-by-pin:minicpm");
    assert!(!outcome.ok());
    let calls = a.host_calls();
    assert!(!calls.iter().any(|c| c.starts_with("stop:")), "{calls:?}");
    assert!(!calls.iter().any(|c| c.starts_with("start:")), "{calls:?}");
}

#[test]
fn a_pin_can_be_cleared_and_an_unknown_pin_is_refused() {
    let a = arbiter(122.0, &[]);
    assert_eq!(a.set_pin("mini"), "minicpm", "an alias pins the model it names");
    assert_eq!(a.pinned(), "minicpm");
    assert_eq!(a.set_pin("nope"), "", "an unknown name is refused");
    assert_eq!(a.pinned(), "minicpm", "and leaves the existing pin alone");
    assert_eq!(a.set_pin("none"), "none");
}

/// A-12: when nothing more can be freed, the model is not started.
#[test]
fn an_impossible_load_fails_without_starting_anything() {
    // A 50 GiB pool cannot hold V4.1's 88 GiB however much is evicted.
    let a = arbiter(50.0, &["minicpm"]);
    let outcome = a.ensure(1);
    match outcome {
        Outcome::InsufficientMemory { needed, .. } => assert_eq!(needed, 88.0),
        other => panic!("expected insufficient memory, got {other:?}"),
    }
    assert!(
        !a.host_calls().contains(&"start:v41".to_string()),
        "a model that cannot fit must never be started"
    );
}

/// A-13 (regression): an unreadable `MemAvailable` means no headroom, not plenty.
///
/// Treating "unknown" as "fine" is how an arbiter OOMs a box: the one measurement that
/// bounds the decision is missing, so the safe reading is zero.
#[test]
fn unknown_memory_is_treated_as_no_headroom() {
    let mut host = FakeHost::new(122.0, &[]);
    host.memory_unknown = true;
    let a = Arbiter::new(models(), config(), host);
    let outcome = a.ensure(2); // small, 10 GiB, would fit in a real 122 GiB pool
    assert!(matches!(outcome, Outcome::InsufficientMemory { .. }), "{outcome:?}");
}

/// A-26 and A-12: a start that fails is reported, not papered over.
#[test]
fn a_failed_start_is_reported() {
    let a = arbiter(122.0, &[]);
    *a.host().start_fails.lock().unwrap() = true;
    let outcome = a.ensure(2);
    assert_eq!(outcome, Outcome::StartFailed);
    assert!(!outcome.ok());
}

// ---------------------------------------------------------------- draining

/// **A-7 and A-8 (regression): a swap waits for in-flight work, then abandons itself.**
///
/// Measured on the box: an 8.68 s drain let a 900-token generation finish, and with a
/// 2 s window the unload was refused `409 busy` while the model stayed up. This is the
/// same property in milliseconds: while a request is held against MiniCPM, a swap to
/// V4.1 must fail with `busy:minicpm` and MiniCPM must still be running afterwards.
#[test]
fn a_swap_is_abandoned_rather_than_killing_a_request() {
    let a = arbiter(122.0, &["minicpm"]);
    let hold = a.hold(0); // stands for a generation in progress
    assert_eq!(a.inflight(0), 1);

    let outcome = a.ensure(1);
    assert_eq!(outcome, Outcome::Busy("minicpm".into()));
    assert_eq!(outcome.label(), "busy:minicpm");
    assert!(!outcome.ok(), "the caller must get a 503, not a backend");
    let calls = a.host_calls();
    assert!(!calls.contains(&"stop:minicpm".to_string()), "the request was not killed");
    assert!(!calls.contains(&"start:v41".to_string()), "and the swap did not proceed");

    // Once the request finishes, the same swap succeeds.
    drop(hold);
    assert_eq!(a.inflight(0), 0);
    assert!(a.ensure(1).ok());
    assert!(a.host_calls().contains(&"stop:minicpm".to_string()));
}

/// A-27 (regression): the count returns to zero on every path, including a panic.
///
/// A leaked count makes a model permanently unswappable, which is invisible until
/// someone tries to switch models and gets `busy` forever.
#[test]
fn the_in_flight_count_always_returns_to_zero() {
    let a = Arc::new(arbiter(122.0, &["minicpm"]));
    {
        let _hold = a.hold(0);
        assert_eq!(a.inflight(0), 1);
    }
    assert_eq!(a.inflight(0), 0, "dropped normally");

    let unwound = Arc::clone(&a);
    let result = std::panic::catch_unwind(move || {
        let _hold = unwound.hold(0);
        panic!("a request handler blew up");
    });
    assert!(result.is_err());
    assert_eq!(a.inflight(0), 0, "dropped while unwinding");
}

/// A-8: an unload is refused while a request is in flight, and the model stays up.
#[test]
fn an_unload_is_refused_while_a_request_is_in_flight() {
    let a = arbiter(122.0, &["minicpm"]);
    let hold = a.hold(0);
    let error = a.unload(0).expect_err("must be refused");
    assert_eq!(error["error"], json!("busy"));
    assert_eq!(error["inflight"], json!(1));
    assert!(!a.host_calls().contains(&"stop:minicpm".to_string()));

    drop(hold);
    let ok = a.unload(0).expect("must succeed once idle");
    assert_eq!(ok["stopped"], json!(true));
    assert_eq!(ok["kv"], json!("save 4/4 ok"));
    // A-14: an unload kills a cache, so the router is told even with no swap.
    assert!(a.host_calls().contains(&"notify:unload:minicpm".to_string()));
}

/// A-11: residency changes are serialised, so two swaps cannot interleave.
#[test]
fn residency_changes_are_serialised() {
    let a = Arc::new(arbiter(122.0, &[]));
    let mut threads = Vec::new();
    for target in [0usize, 2usize] {
        let a = Arc::clone(&a);
        threads.push(std::thread::spawn(move || a.ensure(target)));
    }
    for thread in threads {
        assert!(thread.join().unwrap().ok());
    }
    assert_eq!(
        a.host().peak_concurrent.load(Ordering::SeqCst),
        1,
        "two models must not be started at the same time"
    );
}

// ---------------------------------------------------------------- KV reporting

/// A-15: a ds4 engine persists on its own policy, so there is nothing to call.
#[test]
fn a_ds4_engine_reports_automatic_kv() {
    let host = SystemHost::new(Config::default());
    let models = models();
    let v41 = models.iter().find(|m| m.id == "v41").unwrap();
    assert_eq!(host.save_kv(v41), "automatic");
    assert_eq!(host.restore_kv(v41), "automatic");
}

/// A-19: `slots: 0` disables persistence honestly instead of failing every swap.
#[test]
fn zero_slots_reports_unsupported() {
    let doc = json!({"models": [{"id": "m", "engine": "llama.cpp",
        "base": "http://127.0.0.1:1", "slots": 0}]});
    let models = parse_registry(&doc).unwrap();
    let host = SystemHost::new(Config::default());
    assert_eq!(host.save_kv(&models[0]), "unsupported");
    assert_eq!(host.restore_kv(&models[0]), "unsupported");
}

// ---------------------------------------------------------------- the listing

/// A-21 and A-22: the whole registry, with `loaded` and aliases.
#[test]
fn the_model_list_shows_unloaded_models_too() {
    let a = arbiter(122.0, &["minicpm"]);
    let list = a.model_list();
    let data = list["data"].as_array().unwrap();
    assert_eq!(data.len(), 4, "every registered model appears, loaded or not");

    let minicpm = &data[0];
    assert_eq!(minicpm["id"], json!("minicpm"));
    assert_eq!(minicpm["loaded"], json!(true));
    assert_eq!(minicpm["aliases"], json!(["mini", "minicpm-v4.7"]));

    let v41 = &data[1];
    assert_eq!(v41["loaded"], json!(false), "an unloaded model is still advertised");
    assert_eq!(v41["opt_in"], json!(true));
    assert_eq!(v41["aliases"], json!(["ds41"]));
}

#[test]
fn status_reports_residency_and_headroom() {
    let a = arbiter(122.0, &["minicpm"]);
    let status = a.status();
    assert_eq!(status["reserve_gib"], json!(8.0));
    assert_eq!(status["mem_available_gib"], json!(87.0), "122 - 35 resident");
    let models = status["models"].as_array().unwrap();
    assert_eq!(models[0]["running"], json!(true));
    assert_eq!(models[0]["inflight"], json!(0));
    assert_eq!(models[1]["running"], json!(false));
    assert_eq!(models[3]["external"], json!(true));
}

/// Test-only accessors, kept here so the library exposes no test hooks of its own.
trait Peek {
    fn host(&self) -> &FakeHost;
    fn host_calls(&self) -> Vec<String>;
}

impl Peek for Arbiter<FakeHost> {
    fn host(&self) -> &FakeHost {
        self.host_ref()
    }
    fn host_calls(&self) -> Vec<String> {
        self.host_ref().calls()
    }
}

// ---------------------------------------------------------------- loading on demand

/// A-5 and A-23: an `opt_in` model is not the default, but asking for it loads it.
///
/// Both halves matter. Keeping V4.1 out of the default route is what stops a casual
/// request triggering a 40-minute load; being loadable by name is the entire feature.
#[test]
fn an_opt_in_model_is_not_default_but_is_loadable_by_name() {
    let a = arbiter(200.0, &[]);
    assert_eq!(a.default_model(), Some(0), "minicpm, not the opt_in v41");
    assert_eq!(a.resolve("v41"), Some(1));

    let outcome = a.ensure(1);
    assert!(outcome.ok(), "{outcome:?}");
    assert!(a.host_calls().contains(&"start:v41".to_string()));
}

/// A-3: eviction is judged against `weight_gib` and nothing else.
///
/// A 200 GiB pool holds both models even though one of them is the model the box
/// cannot actually co-host at its real weight — the decision follows the declared
/// number, which is why the registry comment calls an honest `weight_gib` the thing
/// that keeps a swap from OOMing the machine.
#[test]
fn eviction_follows_the_declared_weight() {
    let a = arbiter(200.0, &["minicpm"]);
    let outcome = a.ensure(1);
    assert_eq!(outcome, Outcome::Loaded { evicted: vec![], kv: "restore 4/4 ok".into() });
    assert!(!a.host_calls().iter().any(|c| c.starts_with("stop:")), "{:?}", a.host_calls());
}

// ---------------------------------------------------------------- slot counting

/// A-17 (regression): a `slots` that disagrees with the engine is reported.
///
/// `--parallel` defaults to auto, which chose 4 on this box rather than 1. A plausible
/// `slots: 1` would have persisted one slot of four and left the rest cold, and nothing
/// would have said so — swaps keep working and only prefill time suffers.
#[test]
fn a_slot_count_that_disagrees_with_the_engine_is_reported() {
    let doc = json!({"models": [
        {"id": "minicpm", "engine": "llama.cpp", "base": "http://x:1", "slots": 1},
        {"id": "off", "engine": "llama.cpp", "base": "http://x:2", "slots": 0},
        {"id": "ds4", "engine": "ds4", "base": "http://x:3", "slots": 4}]});
    let models = parse_registry(&doc).unwrap();

    let warning = slot_mismatch(&models[0], Some(4)).expect("a mismatch is reported");
    assert!(warning.contains("slots=1"), "{warning}");
    assert!(warning.contains("total_slots=4"), "{warning}");
    assert!(warning.contains("3 slot(s)"), "{warning}");

    assert_eq!(slot_mismatch(&models[0], Some(1)), None, "agreement is silent");
    assert_eq!(
        slot_mismatch(&models[0], None),
        None,
        "an unreadable /props is not a mismatch"
    );
    assert_eq!(
        slot_mismatch(&models[1], Some(4)),
        None,
        "slots: 0 means persistence is off, not misconfigured"
    );
    assert_eq!(
        slot_mismatch(&models[2], Some(1)),
        None,
        "a ds4 engine has no slots to count"
    );
}

// ---------------------------------------------------------------- key hygiene

/// N-3: a key must not appear in anything the service publishes or logs.
#[test]
fn no_key_appears_in_what_the_arbiter_publishes() {
    let secret = "ds4-SPEC-TEST-SECRET-VALUE";
    // SAFETY: a variable unique to this test.
    unsafe { std::env::set_var("SPEC_HYGIENE_KEY", secret) };
    let doc = json!({"models": [{"id": "m", "engine": "llama.cpp", "base": "http://127.0.0.1:1",
        "key_env": "SPEC_HYGIENE_KEY", "weight_gib": 1, "slots": 0}]});
    let models = parse_registry(&doc).unwrap();
    assert_eq!(models[0].resolve_key(), secret, "the key is resolvable...");

    let config = Config { auth_key: secret.to_string(), ..config() };
    let a = Arbiter::new(models, config, FakeHost::new(122.0, &["m"]));
    for published in [a.status().to_string(), a.model_list().to_string()] {
        assert!(!published.contains(secret), "...but must never be published: {published}");
    }
    unsafe { std::env::remove_var("SPEC_HYGIENE_KEY") };
}
