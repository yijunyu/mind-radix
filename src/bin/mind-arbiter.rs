//! Local model arbiter: one endpoint offering every model this host can serve,
//! swapping them only while idle.
//!
//! Sits *behind* mind-router — point that host's `MIND_AMD` (or `MIND_STUDIO`) at this
//! port instead of at a single model's server, and the router keeps doing prefix-aware
//! host routing with no change. See `SPEC.md` §3.

use std::sync::Arc;

use mind_radix::arbiter::{self, Arbiter, SystemHost};

fn main() {
    let config = arbiter::config_from_env();
    let path = std::env::var("MIND_ARBITER_MODELS")
        .unwrap_or_else(|_| "~/.config/mind/models.json".to_string());
    let models = match arbiter::load_registry(&path) {
        Ok(models) => models,
        Err(error) => {
            eprintln!("[mind-arbiter] {error}");
            std::process::exit(2);
        }
    };

    let summary: Vec<String> = models
        .iter()
        .map(|m| format!("{}={:.0}GiB{}", m.id, m.weight_gib, if m.opt_in { "*" } else { "" }))
        .collect();
    println!(
        "[mind-arbiter] {}:{} registry={} {} auth={} reserve={:.0}GiB{}",
        config.host,
        config.port,
        path,
        summary.join(" "),
        if config.auth_key.is_empty() { "off" } else { "on" },
        config.reserve_gib,
        if config.router_base.is_empty() {
            String::new()
        } else {
            format!(" router={}/{}", config.router_base, config.router_backend)
        },
    );

    let host = SystemHost::new(config.clone());
    // Check the one setting whose failure is invisible (A-17), in the background so a
    // model that is down cannot delay the bind.
    {
        let checked = models.clone();
        let checker = SystemHost::new(config.clone());
        std::thread::spawn(move || checker.report_slot_counts(&checked));
    }
    let arbiter = Arc::new(Arbiter::new(models, config, host));
    if let Err(error) = arbiter.run() {
        eprintln!("[mind-arbiter] bind failed: {error}");
        std::process::exit(1);
    }
}
