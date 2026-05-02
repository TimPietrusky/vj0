#!/usr/bin/env bash
# Snipes an N-GPU RTX 5090 pod in RunPod EU-RO-1 (the only DC with UDP, mandatory for WebRTC),
# attached to the existing network volume so models + Inductor cache come along for free.
# Polls every $INTERVAL seconds; when stock appears, creates the pod and exits.
#
# USAGE
#   bash snipe-pod.sh                # try 4, fall back to 3 — every 60s, forever
#   bash snipe-pod.sh 4              # only 4, no fallback
#   bash snipe-pod.sh 4 3 2          # try 4 → 3 → 2 each cycle
#   INTERVAL=30 bash snipe-pod.sh 4  # every 30s instead of 60s
#
# OUTPUT
#   stdout: timestamped log
#   /tmp/sniped-pod-id.txt: created pod id when sniped (so other tools can pick it up)
#   macOS notification + terminal bell on success
#
# Run in the background (will keep going until success or you kill it):
#   nohup bash snipe-pod.sh > /tmp/snipe.log 2>&1 &
#   tail -F /tmp/snipe.log
#
# Stop:
#   pkill -f snipe-pod.sh

set -uo pipefail

VOLUME_ID=${VOLUME_ID:-jtgc1lxkx3}
DC=${DC:-EU-RO-1}
IMAGE=${IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}
GPU_ID=${GPU_ID:-NVIDIA GeForce RTX 5090}
PORTS=${PORTS:-22/tcp,3000/http,10000/udp,10001/udp,10002/udp}
DISK=${DISK:-80}
INTERVAL=${INTERVAL:-60}
NAME_PREFIX=${NAME_PREFIX:-vj0-flux2-5090}

# Default: try 4 first, fall back to 3. Override on cmdline.
if [ $# -gt 0 ]; then
  COUNTS=("$@")
else
  COUNTS=(4 3)
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

notify() {
  # macOS-only notification + terminal bell. Silent no-op on non-macOS.
  echo -e '\a'  # terminal bell — works everywhere with audible bell on
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$1\" with title \"vj0 pod-sniper\" sound name \"Glass\"" 2>/dev/null || true
  fi
}

try_create() {
  local count=$1
  local cloud=$2
  log "  ↳ ${count}x 5090 / ${cloud} ..."
  local out
  out=$(runpodctl pod create \
    --name "${NAME_PREFIX}x${count}-snipe" \
    --image "$IMAGE" \
    --gpu-id "$GPU_ID" \
    --gpu-count "$count" \
    --container-disk-in-gb "$DISK" \
    --network-volume-id "$VOLUME_ID" \
    --ports "$PORTS" \
    --data-center-ids "$DC" \
    --cloud-type "$cloud" 2>&1)

  if echo "$out" | grep -q '"id"'; then
    local pod_id
    pod_id=$(echo "$out" | grep '"id"' | head -1 | sed 's/.*"id": *"\([^"]*\)".*/\1/')
    log ""
    log "🎯 SNIPED: ${count}x 5090 ${cloud} → pod $pod_id"
    log "   network volume $VOLUME_ID attached"
    log "   pod url: https://${pod_id}-3000.proxy.runpod.net"
    log "   ssh: runpodctl ssh info $pod_id"
    log ""
    echo "$pod_id" > /tmp/sniped-pod-id.txt
    notify "Got ${count}x 5090: ${pod_id}"
    return 0
  fi

  # Common case: stock unavailable. Print compact, keep looping.
  if echo "$out" | grep -qE "no longer any instances|requested specifications"; then
    return 1
  fi

  # Anything else is unexpected — show it so we can debug.
  log "    UNEXPECTED ERROR:"
  echo "$out" | tail -5 | sed 's/^/      /'
  return 2
}

log "================================================"
log "vj0 pod-sniper — looking for 5090 in $DC"
log "  GPU counts (in priority order): ${COUNTS[*]}"
log "  poll interval: ${INTERVAL}s"
log "  network volume: $VOLUME_ID"
log "================================================"

attempt=0
while true; do
  attempt=$((attempt + 1))
  log "── round $attempt ──"
  for count in "${COUNTS[@]}"; do
    for cloud in SECURE COMMUNITY; do
      if try_create "$count" "$cloud"; then
        log "Done. Pod $(cat /tmp/sniped-pod-id.txt) is creating."
        log "Suggested next steps:"
        log "  1) runpodctl ssh info \$(cat /tmp/sniped-pod-id.txt)   # get ssh + port"
        log "  2) ssh into the pod and run:"
        log "     bash /workspace/vj0-flux2klein/setup.sh --start"
        log "  3) update .env.local NEXT_PUBLIC_VJ0_WEBRTC_SIGNALING_URL"
        log "     to https://\$(cat /tmp/sniped-pod-id.txt)-3000.proxy.runpod.net/webrtc/offer"
        exit 0
      fi
    done
  done
  log "  none available, sleeping ${INTERVAL}s..."
  sleep "$INTERVAL"
done
