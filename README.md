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

## Three pieces, one crate

Everything is Rust now. The index was always a crate; the router and the arbiter were
stdlib Python until they were ported, which bought three things worth the rewrite: the
index moved in-process so routing no longer speaks to a child over a pipe, in-flight
accounting became a drop guard rather than a `finally` (a leaked count makes a model
permanently unswappable), and the decisions became unit-testable — **91 tests**, where
the swap and eviction rules previously could only be checked by evicting a 35 GiB model
on the real box.

`SPEC.md` is normative for all three and gives every requirement an id; the tests name
those ids, so a failure says which rule broke rather than which line.

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

**`src/router.rs` — the router.** One endpoint in front of several backends:

* a conversation stays on the backend that served its previous turn (prefix affinity),
  keyed on model + system prompt + first user message, so no client cooperation is
  needed;
* a new conversation is ranked by the index when it recognises the opening, otherwise
  it takes the fastest healthy backend;
* a backend that is down or 5xx-ing is replaced and the conversation follows;
* a request that names a model only considers backends that claim it, learned from the
  `/v1/models` listing each health probe already fetches. Hosts were interchangeable
  while they all ran one model; a host running the arbiter offers a choice, so this is
  now the difference between a slow answer and a wrong one. Capability is checked
  *before* affinity, or a conversation would keep returning to a host that no longer
  holds what it is asking for. Narrowing happens only when some healthy backend claims
  the name — a backend that ignores the `model` field advertises nothing, and reading
  that as "cannot serve it" would route nowhere.

Every reply carries `x-mind-backend` (which host answered) and `x-mind-route` (why).
Binds the LAN by default and **requires an API key**, so an open router is never
mistaken for an open model. [`router::choose`] is a pure function over a snapshot of
health, claims, affinity and estimates, which is why all of §2 of the spec can be
tested without a socket or a model.

## Choosing a model, not just a host: `src/arbiter.rs`

The router answers *which host*, because every host serves the same model. The
arbiter answers *which model this host is currently holding*, which is a different
problem with a harder constraint: on a unified-memory box the models do not fit
together. Measured on the gfx1151 box (~122 GiB usable):

| Model | Resident need |
| --- | --- |
| MiniCPM-V 4.7 Q4_K_M | ~35 GB at 131072 ctx |
| Qwen3.8-Flash-Next Q2 | 137 GiB file, SSD-streamed |
| DeepSeek V4.1 Flash Q2 | ~88 GiB at a 76 GB expert cache (auto sizing plans 101) |

So MiniCPM and V4.1 can never be co-resident. The arbiter puts one OpenAI-compatible
endpoint in front of all of them and swaps on demand:

* **it swaps only at zero in-flight requests.** A model being evicted is drained
  first, and a drain that does not finish inside `MIND_ARBITER_DRAIN_WAIT` abandons
  the swap rather than the request — a client gets `503` with `Retry-After`, never a
  killed generation;
* **it evicts only as much as it must.** Eviction stops as soon as `MemAvailable`
  minus a reserve covers the incoming model's `weight_gib`, so models that genuinely
  fit together stay up;
* **it keeps per-model prompt caches across a swap.** `ds4-server` persists KV on a
  policy with `--kv-disk-dir`; `llama-server` needs `--slot-save-path` plus explicit
  `/slots/{id}?action=save|restore`, which the arbiter calls around the swap. Measured
  on the gfx1151 box: a 3,926-token prompt costs **3,029 ms** to prefill cold, and
  after a full stop/start with KV persisted the same prefix re-entered the engine as
  **4 tokens in 56 ms**. Saving all four slots took 82 ms;
* **it advertises unloaded models.** `GET /v1/models` lists the whole registry with a
  `loaded` flag, so a client can see its choices while nothing is serving and pick
  one; asking for an unloaded model loads it.

Every reply carries `x-mind-model` and `x-mind-arbiter` (`resident`, `loaded`,
`loaded-evicted:<id>(<kv>)`, …), and the router passes both through, so a first
request that took seconds still says why.

Composition is the point: set `MIND_AMD=http://127.0.0.1:8010` and the router needs
no change at all — it keeps ranking hosts by prefix, and the host it picks now
happens to offer a choice of models. Two optional settings close the loop between
them: `MIND_ARBITER_ROUTER` and `MIND_ARBITER_ROUTER_BACKEND` let the arbiter call
`POST /router/forget/<backend>` after a swap, because a swap is a backend restart as
far as a prefix index can tell. That drops the index's estimates for the host and not
its affinity — a conversation is keyed on its model, so letting it come back to the
same host is exactly what makes a swap *back* cheap.

```bash
cp models.example.json ~/.config/mind/models.json   # then edit; never put a key in it
cargo build --release && ./target/release/mind-arbiter

curl -H "Authorization: Bearer $KEY" localhost:8010/v1/models        # choices
curl -H "Authorization: Bearer $KEY" localhost:8010/arbiter/status   # residency + headroom
curl -XPOST -H "Authorization: Bearer $KEY" localhost:8010/arbiter/load/v41
curl -XPOST -H "Authorization: Bearer $KEY" localhost:8010/arbiter/pin/minicpm   # refuse eviction
```

Measured here: unload MiniCPM, then one chat request reloaded it and answered in
**4.19 s** end to end, reporting `x-mind-arbiter: loaded`. With `--slot-save-path` on
the engine and `slots` set to its `total_slots`, that reload also brought the prompt
cache back — the long prefix that cost 3 s to prefill came back as 4 tokens. Note that
`--parallel` defaults to *auto*, which chose 4 slots here, not 1: `slots` must match
`total_slots` from `/props` or the extra slots are silently left cold.

What it cannot do is make a swap instant. Three things must come back and only two
can be saved: KV state persists to disk, weights reload in seconds and not reliably
from page cache (a streaming model reads enough to evict it), and an SSD-streaming
expert cache is in-process RAM that is simply lost — for V4.1 that measured 70–123
GiB of re-reads. Mark such a model `opt_in` so it stays out of casual routing.

## Layout

| Path | What |
| --- | --- |
| `src/lib.rs` | hashing + radix tree + estimates |
| `src/http.rs` | enough HTTP/1.1 for a JSON and SSE reverse proxy, in the standard library |
| `src/router.rs` | prefix- and model-aware host routing |
| `src/arbiter.rs` | local model arbitration and swapping |
| `src/bin/mind-router.rs`, `src/bin/mind-arbiter.rs` | the two services |
| `src/bin/mind-radix-cli.rs` | line-oriented JSON CLI over the index alone |
| `SPEC.md` | normative behaviour, one id per requirement |
| `tests/router_spec.rs` | routing decisions (§2) |
| `tests/arbiter_spec.rs` | swap, eviction and registry rules (§3), against a fake machine |
| `tests/service_spec.rs` | the wire: headers, status codes, streaming, fake engines on real sockets |
| `models.example.json` | arbiter registry template |
| `mind-router.sh` | start/stop/status helper |
| `systemd/mind-router.service` | Linux user unit |
| `systemd/mind-arbiter.service` | Linux user unit for the arbiter |
| `mind-router.py`, `mind-arbiter.py` | the previous implementation, kept only until the live units run the binaries |

## Use

```bash
cargo test                                  # 91 tests
cargo build --release
cargo clippy --all-targets                  # clean

printf '%s\n%s\n' \
  '{"op":"observe","backend":"studio","tokens":[1,2,3],"cached":0,"prompt":3}' \
  '{"op":"estimate","tokens":[1,2,3]}' | ./target/release/mind-radix-cli

# Router: set the backends for THIS host; a backend may be another host's router,
# or its arbiter. An empty MIND_<NAME> drops that backend entirely.
MIND_STUDIO=http://127.0.0.1:8016 MIND_MAC= MIND_AMD=http://127.0.0.1:8010 \
MIND_AMD_KEY=$(cat ~/.ds4-lan-key) ./target/release/mind-router
```

Both hosts run their own router, because a router can only see the caches it has talked
to: studio's router knows studio's cache, and the AMD box's router knows AMD's. On the
AMD box the "studio" backend is studio's *router*, so a prefix living on AMD stays
local instead of always crossing the network to a router that cannot see it.

## Limits

* Tokens are hashed from message text, not a real tokenizer (the router has none). That
  is enough to decide whether two requests share an opening; it is an approximation of
  the engine's own prefix matching, not a replacement.
* A backend restart loses its cache; call `POST /router/forget/<backend>` (or
  `/router/session/clear` to drop affinity as well). An arbiter swap *is* such a
  restart as far as the router can tell, and the arbiter now notifies it when
  `MIND_ARBITER_ROUTER` is set — best-effort and in a background thread, since the
  swap has already happened and a failed notification self-heals: the index only ever
  corrects an estimate downward from measurement. A related subtlety the notification
  does *not* fix: the index keys on message text and not on the model, so two models
  behind one backend name share estimates. Affinity is unaffected — a conversation
  key already includes `model`.
* Model names must agree across hosts, now that they are routed on. The same model
  advertised as `minicpm-v4.7` by one host and `minicpm-v4.7-35b-a3b-q4km` by another
  looks like two models, and a request using the first spelling will skip the second
  host even when it could serve it faster. Alias lists are the fix, which the arbiter
  publishes in its own `/v1/models`; a plain `llama-server` backend cannot, so give it
  the spelling its clients use.
* No engine-side event stream exists for llama.cpp, so estimates are built from prompts
  the router sent. Precise routing would need llama.cpp to emit KV events; the hash
  chain here is the prerequisite for that and is already written.
