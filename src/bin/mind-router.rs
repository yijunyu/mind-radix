//! Prefix-aware router: one endpoint in front of several hosts.
//!
//! Configuration is entirely environmental, with the same variable names the systemd
//! unit already sets, so this binary is a drop-in for the Python it replaces. See
//! `SPEC.md` §2 for what it must do and `src/router.rs` for how.

use std::sync::Arc;

use mind_radix::router::{self, Router};

fn main() {
    let config = router::config_from_env();
    if config.backends.is_empty() {
        eprintln!("[mind-router] no backends configured: set MIND_STUDIO / MIND_AMD / MIND_MAC");
        std::process::exit(2);
    }
    let summary: Vec<String> = config
        .backends
        .iter()
        .map(|b| format!("{}={}", b.name, b.base))
        .collect();
    println!(
        "[mind-router] {}:{} {} auth={} prefer={} affinity_ttl={}s",
        config.host,
        config.port,
        summary.join(" "),
        if config.auth_key.is_empty() { "off" } else { "on" },
        config.preferred,
        config.affinity_ttl.as_secs(),
    );
    if config.auth_key.is_empty() && config.host != "127.0.0.1" {
        // R-21 and N-4: a LAN-bound router with no key would be an open model.
        eprintln!("[mind-router] WARNING: bound to {} with no key", config.host);
    }

    let router = Arc::new(Router::new(config));
    if let Err(error) = router.run() {
        eprintln!("[mind-router] bind failed: {error}");
        std::process::exit(1);
    }
}
