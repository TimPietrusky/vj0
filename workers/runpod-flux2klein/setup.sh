#!/usr/bin/env bash
# One-shot bring-up for the FLUX.2 Klein worker on a fresh RunPod 5090 / Romania.
# Image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# Pod ports: 22/tcp, 3000/http, 10000/udp, 10001/udp, 10002/udp
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# RunPod's container filesystem is ephemeral on stop/start: every time you
# resume a stopped pod, /usr/lib/python3.12/dist-packages and the apt-installed
# binaries (node, etc.) are gone, even though /workspace (the network volume)
# survives. So you need to re-run pip + apt every boot. This script does that
# idempotently — fast-path skips reinstalls when something is already present.
#
# Persistent on /workspace (the network volume):
#   /workspace/hf-cache/             (~23 GB Klein + small decoder)
#   /workspace/torch-inductor-cache/ (~3 GB compiled kernels — saves 130 s/shape on cold compile)
#   /workspace/vj0-flux2klein/       (server.js, inference_server.py, package.json, node_modules)
#
# USAGE
#   bash setup.sh           # install + verify, then print the start command
#   bash setup.sh --start   # install + verify + auto-start server in background
#                            # with WARMUP_SHAPES from $WARMUP_SHAPES (default 512x288,288x512)
#
# TROUBLESHOOTING
#   - "torch CUDA driver too old": pod host has driver < 580. We install
#     torch 2.11.0+cu128 specifically to dodge that — driver 570 works fine
#     with cu128 wheels. (RunPod doesn't expose driver-version host pinning.)
#   - "diffusers can't find Flux2KleinKVPipeline": something pulled a newer
#     diffusers main where the class was renamed. requirements.txt pins to
#     commit 160852de which has the class.

set -euo pipefail

START_SERVER=0
SYNC_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --start) START_SERVER=1 ;;
    --sync-cache) SYNC_ONLY=1 ;;
    *) echo "unknown flag: $arg" ; exit 1 ;;
  esac
done

PIP="pip install --break-system-packages -q"
PROJECT_DIR=/workspace/vj0-flux2klein
HF_CACHE_DIR=${HF_HOME:-/workspace/hf-cache}
WARMUP_SHAPES=${WARMUP_SHAPES:-512x288,288x512}

export HF_HOME=$HF_CACHE_DIR
mkdir -p "$HF_CACHE_DIR"

# torch.compile / Inductor disk cache.
#
# Two-stage strategy:
#   1) Persistent store on the network volume:  /workspace/torch-inductor-cache
#   2) Working copy on local container disk:    /tmp/torchinductor_root
#
# Why split: the network volume's mfs backend is great for big sequential files
# (16 GB Klein weights) but pathologically slow for the thousands of tiny files
# Inductor's Triton bundler reads/writes on every compile call — empirically
# that turned a 20s cache-hit into a 10+ min hang.
#
# So at boot we rsync the persistent store -> local /tmp once (one big seq copy,
# fast even on mfs). torch.compile then runs entirely against /tmp at native
# speed. Before pod stop, run `bash setup.sh --sync-cache` to push /tmp back to
# /workspace so the next boot has fresh kernels. (--start mode does this
# automatically once warmup completes.)
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_root}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}
export TORCHINDUCTOR_AUTOGRAD_CACHE=${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}
INDUCTOR_PERSIST_DIR=${INDUCTOR_PERSIST_DIR:-/workspace/torch-inductor-cache}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$INDUCTOR_PERSIST_DIR"

# Hydrate /tmp from /workspace at startup. Skip if /tmp already populated to
# avoid redundant copies on repeated setup.sh runs in the same pod session.
if [ -d "$INDUCTOR_PERSIST_DIR" ] \
   && [ -n "$(ls -A "$INDUCTOR_PERSIST_DIR" 2>/dev/null)" ] \
   && [ ! -d "$TORCHINDUCTOR_CACHE_DIR/triton" ]; then
  echo "[cache] hydrating $TORCHINDUCTOR_CACHE_DIR from $INDUCTOR_PERSIST_DIR..."
  t0=$(date +%s)
  # rsync is way faster than cp -r for many-file workloads on slow filesystems
  # because it batches stat calls. -a preserves perms+times so cache hashes match.
  if command -v rsync >/dev/null; then
    rsync -a "$INDUCTOR_PERSIST_DIR/" "$TORCHINDUCTOR_CACHE_DIR/"
  else
    cp -a "$INDUCTOR_PERSIST_DIR/." "$TORCHINDUCTOR_CACHE_DIR/"
  fi
  echo "[cache] hydrated in $(($(date +%s) - t0))s, size=$(du -sh "$TORCHINDUCTOR_CACHE_DIR" | cut -f1)"
elif [ ! -d "$INDUCTOR_PERSIST_DIR" ] || [ -z "$(ls -A "$INDUCTOR_PERSIST_DIR" 2>/dev/null)" ]; then
  echo "[cache] no persisted cache yet — first compile will be cold (~150s/shape)"
else
  echo "[cache] $TORCHINDUCTOR_CACHE_DIR already populated, skipping hydrate"
fi

# Helper to push /tmp -> /workspace. Called by --sync-cache and end-of-start.
sync_cache_back() {
  if [ ! -d "$TORCHINDUCTOR_CACHE_DIR" ]; then
    echo "[cache] nothing to sync (no $TORCHINDUCTOR_CACHE_DIR)"
    return
  fi
  echo "[cache] syncing $TORCHINDUCTOR_CACHE_DIR -> $INDUCTOR_PERSIST_DIR..."
  t0=$(date +%s)
  if command -v rsync >/dev/null; then
    rsync -a --delete "$TORCHINDUCTOR_CACHE_DIR/" "$INDUCTOR_PERSIST_DIR/"
  else
    rm -rf "$INDUCTOR_PERSIST_DIR"
    cp -a "$TORCHINDUCTOR_CACHE_DIR" "$INDUCTOR_PERSIST_DIR"
  fi
  echo "[cache] synced in $(($(date +%s) - t0))s, size=$(du -sh "$INDUCTOR_PERSIST_DIR" | cut -f1)"
}

# ---------- helpers ---------- #

# True if a Python module imports cleanly. Use to skip slow reinstalls when
# the dist-packages dir survived (rare — usually it doesn't on RunPod).
pyhas() {
  python3 -c "import $1" >/dev/null 2>&1
}

# Print "x.y.z" for an installed PyPI package, or empty if missing.
pyver() {
  python3 -c "import $1; print($1.__version__)" 2>/dev/null
}

# ---------- node ---------- #

echo "[1/6] node.js (for WebRTC server)"
if command -v node >/dev/null && [[ "$(node --version 2>/dev/null)" == v20.* ]]; then
  echo "      node $(node --version) already installed, skipping"
else
  echo "      installing node 20 from NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1
  apt-get install -y nodejs >/dev/null 2>&1
  echo "      installed $(node --version)"
fi

# ---------- python: torch + cu128 wheel ---------- #

echo "[2/6] torch 2.11.0+cu128 (avoids cu130 driver-580 requirement)"
TORCH_VER=$(pyver torch || true)
if [[ "$TORCH_VER" == "2.11.0+cu128" ]]; then
  echo "      torch $TORCH_VER already installed, skipping"
else
  echo "      installing torch 2.11.0+cu128 (current: ${TORCH_VER:-none})..."
  # --ignore-installed cryptography handles the "no RECORD file" Debian wart.
  $PIP --ignore-installed cryptography \
    torch==2.11.0 torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu128
fi

# ---------- python: pinned diffusers + transformers + reqs ---------- #

echo "[3/6] python deps (pinned diffusers commit, transformers 5.5+, etc.)"
if pyhas diffusers && pyhas transformers && pyhas accelerate \
   && python3 -c "from diffusers import Flux2KleinKVPipeline" >/dev/null 2>&1; then
  echo "      diffusers $(pyver diffusers), transformers $(pyver transformers): klein KV imports OK, skipping pip install"
else
  $PIP -r "$(dirname "$0")/requirements.txt"
  # Force-reinstall the pinned diffusers commit if pip resolved a different one.
  $PIP --no-deps --force-reinstall \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@160852de680d36117e0a787f7f8b718232539abb" \
    "torch==2.11.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128
fi

# ---------- python: torchao (fp8 quantization) ---------- #

echo "[4/6] torchao for fp8 dynamic-act quantization"
if pyhas torchao; then
  echo "      torchao $(pyver torchao) already installed, skipping"
else
  $PIP torchao --extra-index-url https://download.pytorch.org/whl/cu128
fi

# ---------- node deps ---------- #

echo "[5/6] node deps in $PROJECT_DIR"
if [ -f "$PROJECT_DIR/package.json" ]; then
  if [ -d "$PROJECT_DIR/node_modules/@roamhq/wrtc" ]; then
    echo "      node_modules already populated, skipping npm install"
  else
    ( cd "$PROJECT_DIR" && npm install --silent )
  fi
else
  echo "      WARNING: $PROJECT_DIR/package.json missing — SCP project files first"
fi

# ---------- model cache ---------- #

echo "[6/6] models in HF_HOME=$HF_HOME"
if [ -d "$HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-4B" ] \
   && [ -d "$HF_HOME/hub/models--black-forest-labs--FLUX.2-small-decoder" ]; then
  echo "      Klein + small decoder already cached, skipping download"
else
  python3 -c "
from huggingface_hub import snapshot_download
for r in ['black-forest-labs/FLUX.2-klein-4B', 'black-forest-labs/FLUX.2-small-decoder']:
    print(f'fetching {r}...', flush=True)
    snapshot_download(r)
print('models cached')
"
fi

# ---------- verify ---------- #

echo
echo "Verify:"
python3 -c "
import torch, torchao
print(f'torch={torch.__version__} cuda={torch.version.cuda} cuda_avail={torch.cuda.is_available()} devs={torch.cuda.device_count()}')
print(f'torchao={torchao.__version__}')
"
python3 -c "from diffusers import Flux2KleinKVPipeline; print('klein KV imports OK')"

# ---------- maybe start ---------- #

if [ "$SYNC_ONLY" -eq 1 ]; then
  sync_cache_back
  exit 0
fi

if [ "$START_SERVER" -eq 1 ]; then
  echo
  echo "Starting server (WARMUP_SHAPES=$WARMUP_SHAPES)..."
  pkill -9 -f 'node server.js' 2>/dev/null || true
  pkill -9 -f inference_server.py 2>/dev/null || true
  sleep 2
  cd "$PROJECT_DIR"
  nohup env HF_HOME=$HF_HOME WARMUP_SHAPES=$WARMUP_SHAPES \
    TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    TORCHINDUCTOR_FX_GRAPH_CACHE=$TORCHINDUCTOR_FX_GRAPH_CACHE \
    TORCHINDUCTOR_AUTOGRAD_CACHE=$TORCHINDUCTOR_AUTOGRAD_CACHE \
    setsid node server.js > "$PROJECT_DIR/server.log" 2>&1 < /dev/null &
  disown
  sleep 2
  echo "      server pid=$(pgrep -f 'node server.js' | head -1)"
  echo "      logs: tail -F $PROJECT_DIR/server.log"
  echo "      health (from outside): curl https://\$(hostname)-3000.proxy.runpod.net/healthz"
  echo "      first warmup at each new shape: ~20-30 s on cache hit, ~150 s cold"
  echo
  echo "      Cache sync-back (run before stopping the pod to persist new compiled kernels):"
  echo "        bash $(dirname "$0")/setup.sh --sync-cache"
else
  echo
  echo "DONE. To start the server:"
  echo "  cd $PROJECT_DIR && \\"
  echo "    HF_HOME=$HF_HOME WARMUP_SHAPES='$WARMUP_SHAPES' \\"
  echo "    nohup node server.js > server.log 2>&1 < /dev/null & disown"
  echo
  echo "Or just re-run with --start to do it for you."
fi
