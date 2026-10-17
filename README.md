# mind-radix

Router-side prefix index and router for llama.cpp / ggml servers, so a request can be
sent to the backend that most likely still holds its prompt prefix.

## The problem

llama.cpp reuses the longest common prefix of an incoming prompt, and that reuse is
worth a great deal: measured on this setup, a repeated prefix prefills in **85 ms
instead of 1,672 ms** on Metal, and **19 ms instead of 1,785 ms** on ROCm. But
llama.cpp publishes nothing about what it currently holds — `/slots` reports a slot's
cache only while that slot is processing, there is no KV-event stream, and `/props`
says nothing about cached content. So a router cannot ask an engine *"do you already
have this prefix?"* before deciding where to send work.

There are two consequences, and this crate addresses both:

1. **Reuse is prefix-only.** Diverging anywhere invalidates the whole tail. Measured:
   editing one mid-conversation turn in a 7,311-token prompt left only the 245-token
   system prefix reusable and re-prefilled ~7,065 tokens (~5.2 s), at *every* divergence
   depth tried. `--cache-reuse` would rescue this, but it is **silently disabled** for
   this model (`cache_reuse is not supported by this context`), so any implementation
   must emit prompts in a stable prefix order rather than rely on KV shifting.
2. **Routing must be sticky.** Switching backends mid-conversation throws the prefix
   away, so per-conversation affinity is worth more than fairness.

## Two pieces

**`src/lib.rs` — the index.** vLLM-style chained block hashes
(`h_i = H(h_{i-1} ‖ block_i)`), so a block's identity depends on everything before it
and identical prefixes converge on the same keys regardless of which conversation
produced them. Those hashes are indexed in a radix tree to answer "how much of this
prompt has this backend seen?". A measurement from the engine (`cache_n`) corrects the
estimate **downward only** — a request the engine happened to serve cold is not
evidence that the prefix was absent.

This is also the exact value Dynamo's KV router consumes as
`ExternalSequenceBlockHash`, so an engine-side integration later can emit these hashes
unchanged. Only `sha2` is a dependency: no engine bindings, no transport.

**`mind-router.py` — the router.** One endpoint in front of two backends:

* a conversation stays on the backend that served its previous turn (prefix affinity),
  keyed on model + system prompt + first user message, so no client cooperation is
  needed;
* a new conversation is ranked by the index when it recognises the opening, otherwise
  it takes the fastest healthy backend;
* a backend that is down or 5xx-ing is replaced and the conversation follows.

Every reply carries `x-mind-backend` (which host answered) and `x-mind-route` (why).
Binds the LAN by default and **requires an API key**, so an open router is never
mistaken for an open model. If the index binary is missing or dies, routing degrades to
affinity alone — the index improves routing and is never required for it.

## Layout

| Path | What |
| --- | --- |
| `src/lib.rs` | hashing + radix tree + estimates |
| `src/bin/mind-radix-cli.rs` | line-oriented JSON CLI, so the router needs no binding layer |
| `mind-router.py` | the router (stdlib only, Python 3.9+) |
| `mind-router.sh` | start/stop/status helper |
| `systemd/mind-router.service` | Linux user unit |

## Use

```bash
cargo test                                  # 17 tests
cargo build --release
cargo clippy --all-targets                  # clean

printf '%s\n%s\n' \
  '{"op":"observe","backend":"studio","tokens":[1,2,3],"cached":0,"prompt":3}' \
  '{"op":"estimate","tokens":[1,2,3]}' | ./target/release/mind-radix-cli

# Router: set the backends for THIS host; a backend may be another host's router.
MIND_STUDIO=http://127.0.0.1:8016 MIND_AMD=http://192.168.1.123:8000 \
MIND_AMD_KEY=$(cat ~/.ds4-lan-key) MIND_RADIX_BIN=$PWD/target/release/mind-radix-cli \
  ./mind-router.sh up
```

Both hosts run their own router, because a router can only see the caches it has talked
to: studio's router knows studio's cache, and the AMD box's router knows AMD's. On the
AMD box the "studio" backend is studio's *router*, so a prefix living on AMD stays
local instead of always crossing the network to a router that cannot see it.

## Limits

* Tokens are hashed from message text, not a real tokenizer (the router has none). That
  is enough to decide whether two requests share an opening; it is an approximation of
  the engine's own prefix matching, not a replacement.
* A backend restart loses its cache; call `forget_backend` or `POST /router/session/clear`.
* No engine-side event stream exists for llama.cpp, so estimates are built from prompts
  the router sent. Precise routing would need llama.cpp to emit KV events; the hash
  chain here is the prerequisite for that and is already written.
