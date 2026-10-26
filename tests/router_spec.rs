//! Routing decisions, against `SPEC.md` §2.
//!
//! [`mind_radix::router::choose`] is a pure function over a snapshot, so every rule
//! here is tested without a socket, a clock, or a model. Each test names the
//! requirement it pins, so a failure says which rule broke rather than which line.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use mind_radix::router::{self, Backend, advertised, choose, conversation_key, hashable_tokens};
use mind_radix::Estimate;
use serde_json::{Value, json};

fn backends() -> Vec<Backend> {
    vec![
        Backend::new("studio", "http://studio:8015", ""),
        Backend::new("amd", "http://127.0.0.1:8010", "k"),
    ]
}

/// studio serves one spelling; amd is an arbiter offering both models under aliases.
fn claims() -> HashMap<String, BTreeSet<String>> {
    HashMap::from([
        (
            "studio".to_string(),
            BTreeSet::from(["minicpm-v4.7-35b-a3b-q4km".to_string()]),
        ),
        (
            "amd".to_string(),
            BTreeSet::from([
                "minicpm-v4.7".to_string(),
                "minicpm".to_string(),
                "minicpm-v4.7-35b-a3b-q4km".to_string(),
                "deepseek-v4.1-flash".to_string(),
                "v41".to_string(),
            ]),
        ),
    ])
}

fn body(model: &str, user: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": user}]})
}

fn decide(body: &Value) -> (Option<String>, String) {
    decide_with(body, &[true, true], &HashMap::new(), &[])
}

fn decide_with(
    body: &Value,
    healthy: &[bool],
    affinity: &HashMap<String, (String, Instant)>,
    estimates: &[Estimate],
) -> (Option<String>, String) {
    let all = backends();
    let decision = choose(&all, healthy, &claims(), affinity, estimates, "auto", body);
    (
        decision.backend.map(|i| all[i].name.clone()),
        decision.why,
    )
}

// ---------------------------------------------------------------- model listings

/// R-8: all three shapes in use on this LAN are accepted.
#[test]
fn reads_every_model_listing_shape() {
    let openai = br#"{"object":"list","data":[{"id":"Qwen-7B"}]}"#;
    assert_eq!(advertised(openai), BTreeSet::from(["qwen-7b".to_string()]));

    let ollama = br#"{"models":[{"name":"minicpm-v4.7-35b-a3b-q4km","model":"minicpm-v4.7-35b-a3b-q4km"}]}"#;
    assert_eq!(
        advertised(ollama),
        BTreeSet::from(["minicpm-v4.7-35b-a3b-q4km".to_string()])
    );

    let arbiter = br#"{"object":"list","data":[
        {"id":"minicpm-v4.7","aliases":["minicpm","minicpm-v4.7-35b-a3b-q4km"]},
        {"id":"deepseek-v4.1-flash","aliases":["v41","ds41"]}]}"#;
    assert_eq!(
        advertised(arbiter),
        BTreeSet::from([
            "deepseek-v4.1-flash".to_string(),
            "ds41".to_string(),
            "minicpm".to_string(),
            "minicpm-v4.7".to_string(),
            "minicpm-v4.7-35b-a3b-q4km".to_string(),
            "v41".to_string(),
        ])
    );
}

/// R-8: names are matched case-insensitively, so they are stored folded.
#[test]
fn model_names_are_case_folded() {
    assert_eq!(
        advertised(br#"{"data":[{"id":"MiniCPM-V4.7"}]}"#),
        BTreeSet::from(["minicpm-v4.7".to_string()])
    );
}

/// R-9 (regression): junk must mean "made no claim", never "serves nothing".
///
/// The distinction is the whole of R-10: an empty set leaves a backend eligible. If
/// this ever returned a non-empty set for unparseable input, or if callers treated
/// empty as a negative claim, a backend that answers a listing the router cannot read
/// would silently stop receiving traffic.
#[test]
fn unparseable_listings_make_no_claim() {
    for payload in [
        &b"not json at all"[..],
        &b"[1,2,3]"[..],
        &b"{}"[..],
        &br#"{"object":"list","data":[]}"#[..],
        &br#"{"data":[{"id":""},{"id":"  "}]}"#[..],
        &b""[..],
    ] {
        assert!(
            advertised(payload).is_empty(),
            "expected no claim from {:?}",
            String::from_utf8_lossy(payload)
        );
    }
}

// ---------------------------------------------------------------- conversations

/// R-2: the key must survive later turns being appended.
#[test]
fn conversation_key_is_stable_as_a_conversation_grows() {
    let turn1 = json!({"model": "m", "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"}]});
    let turn2 = json!({"model": "m", "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "and now something else entirely"}]});
    assert_eq!(conversation_key(&turn1), conversation_key(&turn2));
}

/// R-2: distinct conversations must not collide, and the model is part of the key.
#[test]
fn conversation_key_separates_conversations_and_models() {
    let a = json!({"model": "m", "messages": [{"role": "user", "content": "alpha"}]});
    let b = json!({"model": "m", "messages": [{"role": "user", "content": "beta"}]});
    let c = json!({"model": "other", "messages": [{"role": "user", "content": "alpha"}]});
    assert_ne!(conversation_key(&a), conversation_key(&b));
    assert_ne!(conversation_key(&a), conversation_key(&c));
}

/// R-2: with no user turn there is nothing to pin.
#[test]
fn conversation_key_is_empty_without_a_user_turn() {
    assert_eq!(conversation_key(&json!({"messages": []})), "");
    assert_eq!(
        conversation_key(&json!({"messages": [{"role": "system", "content": "s"}]})),
        ""
    );
    assert_eq!(conversation_key(&json!({})), "");
}

/// Content parts must be read, or a multimodal client would look like a new
/// conversation on every turn.
#[test]
fn conversation_key_reads_content_parts() {
    let parts = json!({"model": "m", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}]});
    let plain = json!({"model": "m", "messages": [{"role": "user", "content": "hello"}]});
    assert_eq!(conversation_key(&parts), conversation_key(&plain));
}

#[test]
fn hashable_tokens_grow_with_the_conversation_and_share_a_prefix() {
    let one = hashable_tokens(&json!({"messages": [{"role": "user", "content": "a"}]}));
    let two = hashable_tokens(&json!({"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}));
    assert_eq!(one.len(), 8, "one message is one 32-byte digest");
    assert_eq!(two.len(), 16);
    assert_eq!(&two[..8], &one[..], "a longer prompt must extend the same prefix");
    assert!(hashable_tokens(&json!({})).is_empty());
}

// ---------------------------------------------------------------- choosing

/// R-4: no affinity, no estimate, first healthy backend in declaration order.
#[test]
fn a_new_conversation_takes_the_first_healthy_backend() {
    let (backend, why) = decide(&body("minicpm-v4.7-35b-a3b-q4km", "hello"));
    assert_eq!(backend.as_deref(), Some("studio"));
    assert_eq!(why, "new:studio");
}

/// R-5: an unhealthy backend is never chosen.
#[test]
fn an_unhealthy_backend_is_skipped() {
    let (backend, why) = decide_with(
        &body("minicpm-v4.7-35b-a3b-q4km", "hello"),
        &[false, true],
        &HashMap::new(),
        &[],
    );
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "new:amd");

    let (none, why) = decide_with(&body("m", "hello"), &[false, false], &HashMap::new(), &[]);
    assert_eq!(none, None);
    assert_eq!(why, "none-healthy");
}

/// R-1: a conversation stays where its prefix is warm.
#[test]
fn affinity_keeps_a_conversation_on_its_backend() {
    let body = body("minicpm-v4.7-35b-a3b-q4km", "hello");
    let affinity = HashMap::from([(conversation_key(&body), ("amd".to_string(), Instant::now()))]);
    let (backend, why) = decide_with(&body, &[true, true], &affinity, &[]);
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "affinity:amd");
}

/// R-1 and R-5 together: affinity to a dead backend must not strand a conversation.
#[test]
fn affinity_to_an_unhealthy_backend_is_abandoned() {
    let body = body("minicpm-v4.7-35b-a3b-q4km", "hello");
    let affinity = HashMap::from([(conversation_key(&body), ("amd".to_string(), Instant::now()))]);
    let (backend, _) = decide_with(&body, &[true, false], &affinity, &[]);
    assert_eq!(backend.as_deref(), Some("studio"));
}

/// R-3: a non-zero estimate outranks declaration order.
#[test]
fn the_index_outranks_declaration_order() {
    let estimates = vec![Estimate {
        backend: "amd".into(),
        cached_tokens: 512,
        shared_blocks: 32,
        measured: true,
    }];
    let (backend, why) = decide_with(
        &body("minicpm-v4.7-35b-a3b-q4km", "hello"),
        &[true, true],
        &HashMap::new(),
        &estimates,
    );
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "prefix:amd:512");
}

/// R-3: a zero estimate is not evidence of anything and must not steer routing.
#[test]
fn a_zero_estimate_is_ignored() {
    let estimates = vec![Estimate {
        backend: "amd".into(),
        cached_tokens: 0,
        shared_blocks: 0,
        measured: false,
    }];
    let (backend, why) = decide_with(
        &body("minicpm-v4.7-35b-a3b-q4km", "hello"),
        &[true, true],
        &HashMap::new(),
        &estimates,
    );
    assert_eq!(backend.as_deref(), Some("studio"));
    assert_eq!(why, "new:studio");
}

/// R-6: a pin bypasses affinity, the index and even health.
#[test]
fn a_pin_overrides_everything() {
    let all = backends();
    let decision = choose(
        &all,
        &[false, false],
        &claims(),
        &HashMap::new(),
        &[],
        "amd",
        &body("anything", "hello"),
    );
    assert_eq!(decision.backend.map(|i| all[i].name.clone()).as_deref(), Some("amd"));
    assert_eq!(decision.why, "pinned:amd");
}

#[test]
fn a_pin_on_an_unknown_backend_chooses_nothing() {
    let all = backends();
    let decision = choose(&all, &[true, true], &claims(), &HashMap::new(), &[], "nope", &body("m", "x"));
    assert_eq!(decision.backend, None);
    assert_eq!(decision.why, "pinned:nope");
}

// ---------------------------------------------------------------- model awareness

/// R-10 and R-12: only amd claims V4.1, so the request goes there and says why.
#[test]
fn a_request_goes_to_the_only_host_claiming_the_model() {
    let (backend, why) = decide(&body("deepseek-v4.1-flash", "hello"));
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "new:amd+model");
}

/// A-1 and R-8: an alias is as good as the id.
#[test]
fn an_alias_resolves_like_an_id() {
    let (backend, why) = decide(&body("v41", "hello"));
    assert_eq!(backend.as_deref(), Some("amd"));
    assert!(why.ends_with("+model"), "{why}");
}

/// R-10 (regression): nobody claims it, so nothing is narrowed.
///
/// This is the one-sided rule. A backend that ignores the `model` field advertises
/// nothing; if "no claim" were read as "cannot serve it", the candidate set would empty
/// and a perfectly serviceable request would be refused.
#[test]
fn an_unclaimed_model_name_narrows_nothing() {
    let (backend, why) = decide(&body("llama-3-70b", "hello"));
    assert_eq!(backend.as_deref(), Some("studio"));
    assert_eq!(why, "new:studio", "no narrowing, so no +model suffix");
}

/// R-10: when every healthy backend claims it, there is nothing to narrow either.
#[test]
fn a_model_everyone_claims_narrows_nothing() {
    let (_, why) = decide(&body("minicpm-v4.7-35b-a3b-q4km", "hello"));
    assert!(!why.contains("+model"), "{why}");
}

/// R-10: a request naming no model is not narrowed.
#[test]
fn a_request_without_a_model_narrows_nothing() {
    let body = json!({"messages": [{"role": "user", "content": "hello"}]});
    let (backend, why) = decide(&body);
    assert_eq!(backend.as_deref(), Some("studio"));
    assert_eq!(why, "new:studio");
}

/// **R-11 (regression): capability is checked before affinity.**
///
/// The conversation's previous turn was served by studio, but it now asks for a model
/// only amd has. Honouring affinity here would send a request for DeepSeek to a host
/// that has never heard of it — a wrong answer, where the whole point of affinity is
/// to avoid a merely slow one.
#[test]
fn capability_beats_affinity() {
    let body = body("deepseek-v4.1-flash", "hello");
    let affinity = HashMap::from([(conversation_key(&body), ("studio".to_string(), Instant::now()))]);
    let (backend, why) = decide_with(&body, &[true, true], &affinity, &[]);
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "new:amd+model");
}

/// R-11: affinity is still honoured when the held backend can serve the model.
#[test]
fn affinity_survives_a_model_both_hosts_claim() {
    let body = body("minicpm-v4.7-35b-a3b-q4km", "hello");
    let affinity = HashMap::from([(conversation_key(&body), ("amd".to_string(), Instant::now()))]);
    let (backend, why) = decide_with(&body, &[true, true], &affinity, &[]);
    assert_eq!(backend.as_deref(), Some("amd"));
    assert_eq!(why, "affinity:amd");
}

/// R-10 and R-3: narrowing constrains the index, not the other way round.
#[test]
fn the_index_cannot_route_to_a_backend_that_lacks_the_model() {
    let estimates = vec![Estimate {
        backend: "studio".into(),
        cached_tokens: 4096,
        shared_blocks: 256,
        measured: true,
    }];
    let (backend, why) = decide_with(&body("v41", "hello"), &[true, true], &HashMap::new(), &estimates);
    assert_eq!(backend.as_deref(), Some("amd"), "a warm prefix is worthless on a host without the model");
    assert_eq!(why, "new:amd+model");
}

/// R-8: the client's spelling is folded too, not just the backend's.
#[test]
fn a_requested_model_name_is_matched_case_insensitively() {
    let (backend, _) = decide(&body("DeepSeek-V4.1-Flash", "hello"));
    assert_eq!(backend.as_deref(), Some("amd"));
}

// ---------------------------------------------------------------- affinity table

#[test]
fn expired_conversations_are_dropped() {
    let mut affinity = HashMap::from([
        ("fresh".to_string(), ("amd".to_string(), Instant::now())),
        (
            "stale".to_string(),
            ("amd".to_string(), Instant::now() - Duration::from_secs(7200)),
        ),
    ]);
    router::prune_affinity(&mut affinity, Instant::now(), Duration::from_secs(3600), 2048);
    assert!(affinity.contains_key("fresh"));
    assert!(!affinity.contains_key("stale"));
}

#[test]
fn the_affinity_table_is_capped_oldest_first() {
    let now = Instant::now();
    let mut affinity = HashMap::new();
    for i in 0..10u64 {
        affinity.insert(
            format!("c{i}"),
            ("amd".to_string(), now - Duration::from_secs(10 - i)),
        );
    }
    router::prune_affinity(&mut affinity, now, Duration::from_secs(3600), 4);
    assert_eq!(affinity.len(), 4);
    assert!(affinity.contains_key("c9"), "the newest must survive");
    assert!(!affinity.contains_key("c0"), "the oldest must go first");
}

// ---------------------------------------------------------------- configuration

/// R-22 (regression): an empty `MIND_<NAME>` removes that backend.
///
/// This box has no SSH forward to the MacBook, so leaving `mac` configured at its
/// default would make every request pay for a probe that can only fail. The variables
/// are process-global, so this test is the one place they are set and it asserts all
/// three cases at once.
#[test]
fn an_empty_backend_variable_drops_the_backend() {
    // SAFETY: single-threaded within this test; no other test reads these.
    unsafe {
        std::env::set_var("MIND_STUDIO", "http://studio:8015");
        std::env::set_var("MIND_MAC", "");
        std::env::set_var("MIND_AMD", "http://127.0.0.1:8010");
        std::env::set_var("MIND_ROUTER_KEY", "test-key");
        std::env::set_var("MIND_AMD_KEY", "");
    }
    let config = router::config_from_env();
    let names: Vec<&str> = config.backends.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, vec!["studio", "amd"], "mac must be absent entirely");
    assert_eq!(config.auth_key, "test-key");

    // R-23: amd falls back to the shared key, studio keeps its own (here, none).
    let amd = config.backends.iter().find(|b| b.name == "amd").unwrap();
    assert_eq!(amd.key, "test-key");
    let studio = config.backends.iter().find(|b| b.name == "studio").unwrap();
    assert_eq!(studio.key, "");
}
