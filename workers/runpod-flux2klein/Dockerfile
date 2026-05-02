# vj0 FLUX.2 Klein img2img worker — production image.
#
# Base: nvidia/cuda:12.8.1-runtime-ubuntu24.04 (slim — torch wheels are
# prebuilt, no compilation needed at build time).
#
# Default: torch 2.11.0+cu128 — broadest Blackwell compat (driver ≥ 555).
# cu130 is 12% faster at 512×288 on driver ≥ 570 (bench 2026-05-02) and
# works on all current RunPod 5090 / PRO 6000 Blackwell hosts (driver 580).
#
# To build with the faster cu130 variant:
#   docker build --build-arg CU_TAG=cu130 -t vj0-flux2klein-worker:cu130 .
#
# What lives inside the image:
#   /app/server.js              — WebRTC dispatcher (Node 20)
#   /app/inference_server.py    — Python worker (one process per GPU)
#   /app/node_modules/          — express + @roamhq/wrtc
#   /usr/local/lib/python3.12/  — torch 2.11+cu130 (or cu128), diffusers
#                                 (pinned commit), transformers, accelerate,
#                                 torchao, …
#
# What stays on /workspace (network volume, mounted at runtime):
#   /workspace/hf-cache/                    — Klein 4B + small decoder (~23 GB)
#   /workspace/torch-inductor-cache/        — compiled Triton kernels (~3 GB)
#
# Auto-launches the server on container start (entrypoint.sh). Pod stop/
# start no longer needs a manual `setup.sh --start`.
#
# Build:   docker build -t vj0-flux2klein-worker:dev .
# Run:     docker run --gpus all -p 3000:3000 -v workspace:/workspace \
#               -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
#               vj0-flux2klein-worker:dev

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────
# - curl/ca-certs/git: fetching pip packages from VCS (pinned diffusers commit)
# - rsync: Inductor cache hydration (/workspace -> /tmp at boot)
# - libgomp1: torch's CPU OpenMP runtime
# - gcc/g++/python3-dev: REQUIRED at RUNTIME by Triton — torch.compile JITs
#   small CUDA helper modules via tritonbuild; without a C compiler in $PATH
#   the first compile call fails with:
#     "Failed to find C compiler. Please specify via CC environment variable"
#   The stock runpod/pytorch image bundles these; the cuda:runtime base does
#   not. Don't drop them to slim the image — Triton breaks at first frame.
# - openssh-server: optional, only used if PUBLIC_KEY env var is set at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git rsync \
        libgomp1 \
        gcc g++ \
        python3 python3-pip python3-venv python3-dev \
        openssh-server \
    && rm -f /usr/lib/python*/EXTERNALLY-MANAGED \
    && mkdir -p /var/run/sshd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Node 20 (for the WebRTC dispatcher) ───────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── PyTorch ───────────────────────────────────────────────
# Default cu128 — works on all Blackwell sm_120 hosts (driver ≥ 555).
# Override at build time:  docker build --build-arg CU_TAG=cu130 …
# Install order matters: torch + torchvision FIRST from the chosen index,
# then everything else resolves against the already-installed torch (no
# accidental pull-in of the wrong CUDA variant from a transitive dep).
ARG CU_TAG=cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-cache-dir \
        torch==2.11.0 torchvision \
        --extra-index-url https://download.pytorch.org/whl/${CU_TAG}

# ── Application Python deps ───────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt && \
    pip3 install --break-system-packages --no-cache-dir \
        torchao --extra-index-url https://download.pytorch.org/whl/${CU_TAG} && \
    rm /tmp/requirements.txt

# ── Sanity check at build time ────────────────────────────
# Fail the image build (instead of failing at first frame in production) if
# any of the load-bearing imports breaks under the resolved version set.
RUN python3 -c "\
import torch; \
import torchao; \
from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2; \
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig; \
from torchao.quantization.granularity import PerTensor; \
print(f'torch={torch.__version__} torchao={torchao.__version__} klein-imports=OK')"

# ── Application files ─────────────────────────────────────
WORKDIR /app

# Node deps first (better layer caching — these change rarely)
COPY package.json ./
RUN npm install --omit=dev --silent && \
    npm cache clean --force

# Source files (these change more often)
COPY server.js inference_server.py ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Production env defaults (Phase 9A winner) ─────────────
ENV PORT=3000 \
    HF_HOME=/workspace/hf-cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root \
    INDUCTOR_PERSIST_DIR=/workspace/torch-inductor-cache \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
    INFERENCE_SCRIPT=/app/inference_server.py \
    COMPILE_MODE=reduce-overhead \
    USE_VAE_FP8=1 \
    WARMUP_SHAPES=512x288,288x512

EXPOSE 3000 22

# Healthcheck — RunPod can use this to know when traffic is safe to route.
# start-period covers cold-cache compile (~3 min); after that it polls every
# 15 s and only marks healthy when both workers report `inferenceReady:true`.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/healthz | grep -q "\"inferenceReady\":true" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
