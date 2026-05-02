#!/usr/bin/env bash
# Container entrypoint for the FLUX.2 Klein worker.
#
# Order of operations:
#   1) Hydrate the torch-Inductor cache from /workspace -> /tmp (one big seq
#      copy, fast even on RunPod's mfs network volume; mfs is pathologically
#      slow for the thousands of small files Inductor reads on every compile,
#      so we keep the working copy on local container disk).
#   2) If PUBLIC_KEY env var is set (RunPod provides this when you add an SSH
#      key in the pod template), launch sshd so the web shell + runpodctl ssh
#      work. We do this BEFORE launching the server so debug access is
#      available even if the server crashes during warmup.
#   3) Pre-warm the HF model cache (Klein 4B + small decoder, ~23 GB). Done
#      synchronously so /healthz never reports "ready" before the model is
#      present on disk — avoids the misleading "connected but nothing
#      happens" state.
#   4) exec node /app/server.js (which spawns one Python worker per GPU and
#      starts the HTTP signaling endpoint on $PORT).
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# ── 1) Inductor cache hydration ───────────────────────────
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$INDUCTOR_PERSIST_DIR" "$HF_HOME"
if [ -d "$INDUCTOR_PERSIST_DIR" ] \
   && [ -n "$(ls -A "$INDUCTOR_PERSIST_DIR" 2>/dev/null)" ] \
   && [ ! -d "$TORCHINDUCTOR_CACHE_DIR/triton" ]; then
    log "hydrating inductor cache: $INDUCTOR_PERSIST_DIR -> $TORCHINDUCTOR_CACHE_DIR"
    t0=$(date +%s)
    rsync -a "$INDUCTOR_PERSIST_DIR/" "$TORCHINDUCTOR_CACHE_DIR/"
    log "hydrated in $(($(date +%s) - t0))s, size=$(du -sh "$TORCHINDUCTOR_CACHE_DIR" | cut -f1)"
elif [ ! -d "$INDUCTOR_PERSIST_DIR" ] || [ -z "$(ls -A "$INDUCTOR_PERSIST_DIR" 2>/dev/null)" ]; then
    log "no persisted inductor cache yet — first compile per shape will be cold (~150s)"
else
    log "inductor cache already populated, skipping hydrate"
fi

# ── 2) sshd (only if PUBLIC_KEY is set, RunPod convention) ─
if [ -n "${PUBLIC_KEY:-}" ]; then
    log "PUBLIC_KEY detected, configuring sshd"
    mkdir -p /root/.ssh
    # PUBLIC_KEY may contain multiple keys separated by literal "\n" or real
    # newlines; printf %b decodes the escape sequence form RunPod uses.
    printf '%b\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1
    /usr/sbin/sshd
    log "sshd listening on :22"
fi

# ── 3) Pre-fetch HF models (idempotent — skips if already cached) ──
if [ -d "$HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-4B" ] \
   && [ -d "$HF_HOME/hub/models--black-forest-labs--FLUX.2-small-decoder" ]; then
    log "Klein + small decoder already cached at $HF_HOME"
else
    log "fetching Klein 4B + small decoder to $HF_HOME (one-time, ~23 GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
for r in ['black-forest-labs/FLUX.2-klein-4B', 'black-forest-labs/FLUX.2-small-decoder']:
    print(f'  fetching {r}...', flush=True)
    snapshot_download(r)
print('  models cached', flush=True)
"
fi

# ── 4) Launch the server ──────────────────────────────────
log "starting node /app/server.js (PORT=$PORT, COMPILE_MODE=$COMPILE_MODE, WARMUP_SHAPES=$WARMUP_SHAPES)"
exec node /app/server.js
