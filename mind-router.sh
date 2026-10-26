#!/usr/bin/env bash
# Start the Local Mind router on this host.
#
# The router prefers whichever backend most likely still holds an incoming prompt's
# prefix, then the faster one. Each host runs its OWN router because a router can only
# see the caches it has talked to: studio's router knows studio's cache, and the AMD
# box's router knows AMD's. On AMD the "studio" backend is studio's router, so a
# prefix that lives on AMD is still reachable locally instead of always crossing the
# network to a router that cannot see it.
#
# For a host without systemd; where the user manager is available, the unit in
# systemd/ is better. Expects the release binary installed as ~/mind-router.
#
#   mind-router.sh up|down|status
set -uo pipefail
PORT="${MIND_ROUTER_PORT:-8015}"
HOST="${MIND_ROUTER_HOST:-0.0.0.0}"
KEY="${MIND_ROUTER_KEY:-$(cat "$HOME/.ds4-lan-key" 2>/dev/null)}"
LOG="$HOME/mind-router.log"
up() { curl -sf -m 4 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/router/status" >/dev/null 2>&1; }
case "${1:-status}" in
  up)
    if up; then echo "already on $PORT"; exit 0; fi
    pkill -x "mind-router" 2>/dev/null; sleep 2
    MIND_ROUTER_PORT="$PORT" MIND_ROUTER_HOST="$HOST" MIND_ROUTER_KEY="$KEY" \
      nohup "$HOME/mind-router" >>"$LOG" 2>&1 &
    echo "started pid $!"
    for _ in $(seq 1 20); do up && { echo "ready"; exit 0; }; sleep 1; done
    echo "not ready" >&2; tail -5 "$LOG" >&2; exit 1 ;;
  down) pkill -x "mind-router" && echo stopped || echo "not running" ;;
  status) up && echo "router serving on $PORT" || echo "router down"
          curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/router/status" 2>/dev/null | python3 -m json.tool 2>/dev/null ;;
esac
