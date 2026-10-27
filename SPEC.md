# mind-radix: specification

Normative behaviour of the two services and the index they share. Each requirement has
an ID; the Rust tests name these IDs, so a failing test says which requirement broke.

**MUST** / **MUST NOT** / **SHOULD** carry their usual force. "Engine" means a
llama.cpp-family or ds4 server that caches prompt prefixes. "Backend" means a host the
router may send a request to, which may itself be another router or an arbiter.

Measured figures are from the gfx1151 box (Radeon 8060S, ~122 GiB unified pool) and are
recorded because several requirements exist only because of them.

---

## 1. The index (`PrefixIndex`)

| ID | Requirement |
| --- | --- |
| **I-1** | Block hashes MUST chain: `h_i = H(h_{i-1} ‖ block_i)`, so identical prefixes converge on identical keys regardless of which conversation produced them. |
| **I-2** | An `observe` with a measured `cached_tokens` MUST lower an existing estimate and MUST NOT raise it. A request an engine happened to serve cold is not evidence that the prefix was absent. |
| **I-3** | `forget_backend` MUST drop every block for that backend and leave other backends untouched. |
| **I-4** | An estimate for an unknown prompt MUST be empty rather than a guess. |

## 2. Routing (`mind-router`)

### 2.1 Choosing a backend

| ID | Requirement |
| --- | --- |
| **R-1** | A conversation MUST stay on the backend that served its previous turn while that backend is healthy (affinity). Measured reason: switching mid-conversation re-prefills the whole tail — ~5.2 s on a 7,311-token prompt. |
| **R-2** | A conversation MUST be identified from the request alone, by model + system prompt + first user message, and MUST NOT depend on anything after the first user turn. Keying on a sliding window of recent messages breaks affinity at turn 2. |
| **R-3** | With no affinity, the index MUST be consulted, and a backend with a non-zero estimate MUST be preferred over declaration order. |
| **R-4** | With no affinity and no estimate, the first healthy backend in declaration order MUST be chosen. |
| **R-5** | An unhealthy backend MUST NOT be chosen. Health MUST be cached for `MIND_PROBE_TTL` so it costs nothing per request. |
| **R-6** | `MIND_PREFER=<name>` or `POST /router/prefer/<name>` MUST pin every request to that backend, bypassing affinity and the index. `auto` MUST restore normal routing. |

### 2.2 Model-aware routing

Hosts were interchangeable while they all served one model. A host running an arbiter
offers a choice, so a request naming a model can reach a host that has never heard of
it — a wrong answer rather than a slow one.

| ID | Requirement |
| --- | --- |
| **R-7** | The router MUST learn which model names each backend answers to from the `/v1/models` response it already fetches for the health probe, and MUST NOT make an extra request for it. |
| **R-8** | That parse MUST accept all three shapes in use: OpenAI `{"data":[{"id":…}]}`, Ollama-style `{"models":[{"name":…}]}`, and the arbiter's (OpenAI plus an `aliases` array). Matching MUST be case-insensitive. |
| **R-9** | An unparseable or absent listing MUST yield "no claim", never "serves nothing". |
| **R-10** | **Narrowing is one-sided.** When a request names a model, candidates MUST be narrowed to backends claiming it **only if at least one healthy backend claims it**. Otherwise every healthy backend stays eligible. A backend that ignores the `model` field advertises nothing, and treating that as "cannot serve it" would route nowhere. |
| **R-11** | **Capability MUST be checked before affinity.** A conversation held on a backend that no longer claims the requested model MUST be re-routed, not honoured. |
| **R-12** | When narrowing changed the candidate set, the route reason MUST say so (suffix `+model`). |

### 2.3 Cache invalidation

| ID | Requirement |
| --- | --- |
| **R-13** | `POST /router/forget/<backend>` MUST drop that backend's index estimates and force a fresh capability probe. |
| **R-14** | It MUST NOT drop affinity. A conversation key includes its model, so letting a conversation return to the same host is what makes a swap *back* cheap. |
| **R-15** | An unknown backend name MUST be rejected with 400 rather than silently ignored. |

### 2.4 Proxying

| ID | Requirement |
| --- | --- |
| **R-16** | Every reply MUST carry `x-mind-backend` (who answered) and `x-mind-route` (why). |
| **R-17** | `x-mind-model` and `x-mind-arbiter` from the backend MUST be passed through. They are the only account of why a first request took seconds. |
| **R-18** | `text/event-stream` MUST be streamed through without buffering the body. |
| **R-19** | A 5xx or an unreachable backend MUST be retried once on another healthy backend, and the conversation's affinity MUST follow the backend that answered. |
| **R-20** | Only the request path and body come from the client; scheme and host MUST come from this process's configuration. |

### 2.5 Configuration and auth

| ID | Requirement |
| --- | --- |
| **R-21** | The router MUST require an API key when one is configured, and MUST answer 401 before contacting any backend. |
| **R-22** | An empty `MIND_<NAME>` MUST remove that backend entirely, so one file can serve every host. A configured-but-unreachable backend would otherwise fail a probe on every request. |
| **R-23** | Per-backend keys MUST be separate (`MIND_<NAME>_KEY`), because a backend may be another host's authenticated router. |

## 3. Arbitration (`mind-arbiter`)

On a unified-memory box the models do not fit together: MiniCPM-V 4.7 needs ~35 GiB at
131072 ctx and DeepSeek V4.1 Flash ~88 GiB, against ~122 GiB.

### 3.1 The registry

| ID | Requirement |
| --- | --- |
| **A-1** | A model MUST be addressable by its `id` or any of its `aliases`. |
| **A-2** | The registry MUST NOT contain an API key. It names where one lives (`key_env`, `key_env_file`), and the arbiter resolves it at use. |
| **A-3** | `weight_gib` is what the model needs resident, including KV and any expert cache. It MUST be the only quantity eviction is judged against. It is **not** the GGUF file size: llama.cpp preallocates the whole `-c` cache at startup, so the context is already in the resident figure and must be declared with it. Nanbeige4.3-33B checks out to the GiB — 19.36 weights + 13.81 KV (131072 × q8_0) + 1.43 compute buffers = 34.60 measured at load, against 36 declared. Declaring the 19.36 file size instead would have let the arbiter admit it beside a model it cannot actually share the box with. |
| **A-4** | `external: true` MUST mean the arbiter never starts or stops it and never counts it against local memory. |
| **A-5** | `opt_in: true` MUST keep a model out of the default route; it MUST still be loadable by name. |
| **A-6** | The default model for a request naming none MUST be the first non-`opt_in` entry. |

### 3.2 Swapping

| ID | Requirement |
| --- | --- |
| **A-7** | **A swap MUST happen only at zero in-flight requests for the model being stopped.** |
| **A-8** | A model being evicted MUST be drained, and a drain that does not finish within `MIND_ARBITER_DRAIN_WAIT` MUST abandon **the swap**, not the request. The caller gets 503 with `Retry-After`; the running generation completes. Verified: an 8.68 s drain let a 900-token generation finish with `finish_reason: length`, and with a 2 s window the unload was refused `409 {"error":"busy","inflight":1}` while the model stayed up. |
| **A-9** | Eviction MUST be minimal: it stops models one at a time and MUST stop as soon as `MemAvailable − MIND_ARBITER_RESERVE_GIB ≥ weight_gib` of the incoming model. Models that genuinely fit together stay resident. |
| **A-10** | A pinned model MUST NOT be evicted; the swap MUST fail with `blocked-by-pin:<id>` instead. |
| **A-11** | Residency changes MUST be serialised, so two requests for two different unloaded models cannot both decide to evict the same third one. |
| **A-12** | If headroom is still insufficient after evicting everything evictable, the swap MUST fail with `insufficient-memory:…` and MUST NOT start the model. |
| **A-13** | An unknown `MemAvailable` MUST be treated as no headroom, never as plenty. |
| **A-14** | After a swap that stopped something, the arbiter SHOULD notify a configured router (`MIND_ARBITER_ROUTER`, `MIND_ARBITER_ROUTER_BACKEND`). It MUST be best-effort: never blocking the swap, never failing it. Loading into free memory harms no other cache and MUST NOT notify. Both branches are confirmed live, in one run: admitting Nanbeige4.3-33B (36 GiB) against 81 GiB of headroom left MiniCPM resident and answering and sent no forget, while the explicit unload of the same model sent `forget amd`. |

### 3.3 KV persistence

| ID | Requirement |
| --- | --- |
| **A-15** | A `ds4` engine persists KV on its own policy once `--kv-disk-dir` is set; the arbiter MUST report `automatic` and call nothing. |
| **A-16** | A `llama.cpp` engine MUST be saved and restored per slot via `POST /slots/{id}?action=save\|restore`, which requires the server to have been started with `--slot-save-path`. The round trip is verified end-to-end against the real adapter, not only the test host: on Nanbeige4.3-33B the arbiter logged `kv restore 2/2 ok` on an on-demand load and `save 2/2 ok` on the unload, and the *first* request after the load reported `cached_tokens: 25` of a 40-token prompt. Nothing but the restored slot can supply those tokens, so this is evidence the cache is reused and not merely read back. |
| **A-17** | `slots` MUST equal the engine's `total_slots` from `/props`. `--parallel` defaults to *auto*, which chose **4** on this box, not 1 — a plausible-but-wrong value silently leaves the other slots cold. |
| **A-18** | **Restoring a slot that was never saved MUST be reported as `cold`, not as a failure.** llama-server answers 400 with the same message it uses for a corrupt file, so counting non-2xx as failure makes every first swap-in look broken. |
| **A-19** | `slots: 0` MUST disable KV persistence and report `unsupported` rather than failing every swap. |
| **A-20** | KV files MUST be named per model and slot and overwritten in place, so disk use is bounded by slot count. Measured on MiniCPM-V 4.7: a 3,926-token slot wrote ~330 MiB, of which ~190 MiB is context checkpoints; marginal KV ~20 KiB/token, so four full 131072-token slots project to ~10 GiB. **That rate is the model's attention shape, not a constant of this design**, and the bound is only as good as the registry's `slots`: Nanbeige4.3-33B measured 110.6 KiB/token with a `q8_0` cache and 208 KiB/token at f16, ten times MiniCPM's, so two full 65536-token slots project to ~14 GiB. Size a `--slot-save-path` from the model in hand. |

### 3.4 Serving

| ID | Requirement |
| --- | --- |
| **A-21** | `GET /v1/models` MUST list the **whole registry** with a `loaded` flag, so a client can see its choices while nothing is serving. |
| **A-22** | It MUST publish each model's `aliases`, because the arbiter is the authority on what names it answers to and a router in front of it cannot read the registry. |
| **A-23** | A request for an unloaded model MUST load it (subject to §3.2) and then serve it. |
| **A-24** | A request naming an unknown model MUST be answered 400 with the known ids listed, not routed to a default. |
| **A-25** | Every reply MUST carry `x-mind-model` and `x-mind-arbiter` (`resident`, `loaded`, `loaded-evicted:<id>(<kv>)`, …). |
| **A-26** | Any "not now" MUST be a 503 with `Retry-After`, never a hang and never a killed request. |
| **A-27** | In-flight accounting MUST be decremented on every path, including error paths, or a crashed request would block swaps forever. |

### 3.5 What it cannot do

Recorded so the limits are not mistaken for defects:

* A swap is not instant. Three things must come back and only two can be saved: KV
  persists to disk; weights reload in seconds but not reliably from page cache, since a
  streaming model reads enough to evict it; and an SSD-streaming expert cache is
  in-process RAM that is simply lost — for V4.1 that measured 70–123 GiB of re-reads.
  Such a model belongs behind `opt_in`.
* The index keys on message text, not on the model, so two models behind one backend
  name share estimates. Affinity is unaffected (R-2 includes the model).
* Model names must agree across hosts (R-8). The same model advertised as
  `minicpm-v4.7` by one host and `minicpm-v4.7-35b-a3b-q4km` by another looks like two
  models, and a request using one spelling skips the other host.

## 4. Non-functional

| ID | Requirement |
| --- | --- |
| **N-1** | Both services MUST run from the repository with no network fetch at run time and no transport dependency beyond the standard library plus JSON. |
| **N-2** | A missing or broken index MUST NOT fail a request. This was a real failure mode when the index ran as a child process reached over a pipe; it now lives in the router's own address space, so the requirement is satisfied structurally and there is nothing left to degrade to. |
| **N-3** | Neither service may log an API key, and neither may write one to a file it creates. |
| **N-4** | The arbiter MUST bind loopback by default; the router MUST bind the LAN and therefore MUST require a key (R-21). |
