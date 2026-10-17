#!/usr/bin/env python3
"""Matched decode benchmark for the Local Mind router on any host.

Same design as the Studio/AMD measurements so the numbers are comparable: three
interleaved A/B/B/A pairs over two prompts, 256 generated tokens, one warm-up run
excluded, spread reported.

    python3 bench-mac.py                 # defaults to the Mac's tunnel on :8020
    python3 bench-mac.py 8020            # or another port
"""
import http.client
import json
import os
import statistics
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8020
HOST = "127.0.0.1"
MODEL = "minicpm-v4.7-35b-a3b-q4km"
KEYFILE = os.path.expanduser("~/.ds4-lan-key")
ARMS = {
    "A": "Write a detailed tutorial on hash tables and collision handling in more than 500 words.",
    "B": "Summarise how binary search trees maintain balance, in more than 500 words.",
}

try:
    KEY = open(KEYFILE).read().strip()
except OSError:
    print(f"missing {KEYFILE}; this must match the AMD box's key")
    raise SystemExit(1)


def run(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 256, "reasoning_effort": "none"}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    started = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                     {"Content-Type": "application/json",
                      "Authorization": f"Bearer {KEY}"})
        with conn.getresponse() as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {resp.read().decode()[:300]}")
                raise SystemExit(1)
            reply = json.load(resp)
            backend = resp.getheader("x-mind-backend") or "?"
            route = resp.getheader("x-mind-route") or "?"
    finally:
        conn.close()
    wall = time.monotonic() - started
    timings = reply.get("timings") or {}
    served = reply.get("model")
    return wall, timings.get("predicted_per_second"), timings.get("prompt_per_second"), backend, route, served


# A warm-up is excluded so the first measured arm is not the cold one.
w, tps, pps, be, why, served = run(ARMS["A"])
print(f"warm-up (excluded): {tps:.1f} tok/s, prefill {pps:.1f} tok/s, backend={be}")
print(f"served model: {served}\n")

results = []
for rep in range(3):
    for arm in (["A", "B"] if rep % 2 == 0 else ["B", "A"]):
        w, tps, pps, be, why, served = run(ARMS[arm])
        results.append(tps)
        print(f"  {arm} rep{rep + 1}: {tps:6.1f} tok/s  ({w:5.2f}s)  backend={be:7} route={why}")

print(f"\nMEDIAN {statistics.median(results):.1f} tok/s | spread {max(results) - min(results):.1f} "
      f"| n={len(results)}")
print("compare: studio 82.7 tok/s (llama.cpp Metal, Q4_K_M) | amd 60.5 tok/s (PolyStrata HIP, mixed-Q3)")
