/**
 * WebRTC server with FLUX.2 Klein img2img inference.
 *
 * Spawns N Python `inference_server.py` workers (one per GPU). Frame requests
 * are round-robined; state changes (prompt/seed/resolution/etc.) are broadcast
 * to every worker so they all stay in sync. Each worker is pinned to its own
 * GPU via CUDA_VISIBLE_DEVICES.
 *
 * env vars:
 *   PORT              — HTTP signaling port (default 3000)
 *   INFERENCE_SCRIPT  — path to Python worker (default ./inference_server.py)
 *   WORKER_COUNT      — number of inference workers; defaults to torch's
 *                       cuda.device_count (auto-detected on startup; capped to 8).
 *                       Set to "1" to force the legacy single-GPU behavior.
 *   ICE_SERVERS_JSON  — STUN/TURN servers for WebRTC
 *   ICE_GATHER_TIMEOUT_MS
 */
const express = require("express");
const wrtc = require("@roamhq/wrtc");
const { spawn, execSync } = require("child_process");

const PORT = Number(process.env.PORT || 3000);
const INFERENCE_SCRIPT = process.env.INFERENCE_SCRIPT || "./inference_server.py";
const ICE_GATHER_TIMEOUT_MS = Number(process.env.ICE_GATHER_TIMEOUT_MS || 10000);

// Auto-detect GPU count if WORKER_COUNT not set
function detectGpuCount() {
  if (process.env.WORKER_COUNT) {
    const n = Number(process.env.WORKER_COUNT);
    if (Number.isFinite(n) && n >= 1) return Math.min(n, 8);
  }
  try {
    const out = execSync("nvidia-smi --query-gpu=index --format=csv,noheader", {
      encoding: "utf8",
      timeout: 5000,
    });
    const n = out.trim().split("\n").filter(Boolean).length;
    return Math.max(1, Math.min(n, 8));
  } catch {
    console.warn("nvidia-smi not available; defaulting to 1 worker");
    return 1;
  }
}

const WORKER_COUNT = detectGpuCount();

// State-changing fields a client message may set on a worker. If a message
// carries any of these, we broadcast that subset to ALL workers so they stay
// in sync. Frame data (`image_base64`) is the only field that gets routed
// round-robin to a single worker.
const STATE_FIELDS = ["prompt", "seed", "alpha", "n_steps", "width", "height",
                      "captureWidth", "captureHeight"];

// Drop outbound frames if the WebRTC DataChannel buffer exceeds this many bytes.
// Live VJ wants newest-wins; stale frames in flight clog the channel.
// 1MB ≈ 30 frames of buffered output at 256² JPEG.
const MAX_OUTBOUND_BUFFER = Number(process.env.MAX_OUTBOUND_BUFFER || 1024 * 1024);
let droppedOutbound = 0;

function getIceServers() {
  const raw = process.env.ICE_SERVERS_JSON;
  if (!raw) return [{ urls: "stun:stun.l.google.com:19302" }];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
  } catch {}
  return [{ urls: "stun:stun.l.google.com:19302" }];
}

// ---------------- worker pool ---------------- //

/** @type {Array<{proc: any, gpu: number, ready: boolean, stdoutBuf: string, framePending: number}>} */
const workers = [];
let roundRobinIdx = 0;

// Latest known state, accumulated from broadcasts. Replayed to each worker
// when it transitions to READY so workers that come up late don't end up
// stuck at default state while their peers are at the user's actual config.
// Without this, worker A and worker B can drift if the client sends a state
// change between A.READY and B.READY — round-robin dispatch then alternates
// good/bad frames.
const latestState = {};

function spawnWorker(gpu) {
  console.log(`[worker ${gpu}] spawning (CUDA_VISIBLE_DEVICES=${gpu})`);
  const env = { ...process.env, CUDA_VISIBLE_DEVICES: String(gpu), WORKER_ID: String(gpu) };
  const proc = spawn("python3", [INFERENCE_SCRIPT], {
    stdio: ["pipe", "pipe", "inherit"],
    env,
  });
  const w = { proc, gpu, ready: false, stdoutBuf: "", framePending: 0 };
  workers.push(w);

  proc.stdout.on("data", (chunk) => {
    w.stdoutBuf += chunk.toString();
    let nl;
    // Process complete lines only — frames are large base64 payloads,
    // chunks may split mid-line.
    while ((nl = w.stdoutBuf.indexOf("\n")) >= 0) {
      const line = w.stdoutBuf.slice(0, nl);
      w.stdoutBuf = w.stdoutBuf.slice(nl + 1);
      if (!line) continue;
      handleWorkerLine(w, line);
    }
  });

  proc.on("close", (code) => {
    console.log(`[worker ${gpu}] exited code=${code}`);
    w.ready = false;
  });

  proc.on("error", (err) => {
    console.error(`[worker ${gpu}] error:`, err);
    w.ready = false;
  });
}

function handleWorkerLine(w, line) {
  let msg;
  try { msg = JSON.parse(line); }
  catch {
    console.log(`[worker ${w.gpu} raw]`, line.slice(0, 200));
    return;
  }

  if (msg.log) {
    console.log(`[worker ${w.gpu}]`, msg.log);
    return;
  }
  if (msg.status === "ready") {
    console.log(`[worker ${w.gpu}] READY`);
    w.ready = true;
    // Replay the latest known state into this worker. Without this, a worker
    // that came up after its peer received a state change is stuck at default
    // settings (e.g. 256x256) while peers are at the user's actual config
    // (e.g. 288x512), and round-robin dispatch alternates good/bad frames.
    if (Object.keys(latestState).length > 0) {
      try {
        w.proc.stdin.write(JSON.stringify(latestState) + "\n");
        console.log(`[worker ${w.gpu}] replayed latest state: ${Object.keys(latestState).join(",")}`);
      } catch (e) {
        console.error(`[worker ${w.gpu}] state replay failed:`, e.message);
      }
    }
    flushPendingForWorker(w);
    return;
  }
  // Compile status messages — forward to WebRTC client so the UI can show
  // a "compiling 512x288..." overlay during the ~150s JIT cost on shape change.
  if (msg.status === "compiling" || msg.status === "compiling_progress" || msg.status === "warmed") {
    if (msg.status === "compiling") {
      console.log(`[worker ${w.gpu}] compiling ${msg.width}x${msg.height} (~${msg.est_seconds}s)`);
    } else if (msg.status === "warmed") {
      console.log(`[worker ${w.gpu}] warmed ${msg.width}x${msg.height} in ${msg.total_ms}ms`);
    }
    if (activeChannel?.readyState === "open") {
      activeChannel.send(JSON.stringify({
        type: "compile",
        status: msg.status,
        width: msg.width,
        height: msg.height,
        n_steps: msg.n_steps,
        iter: msg.iter,
        total_iters: msg.total_iters,
        elapsed_ms: msg.elapsed_ms,
        iter_ms: msg.iter_ms,
        total_ms: msg.total_ms,
        est_seconds: msg.est_seconds,
        worker: w.gpu,
      }));
    }
    return;
  }

  if (msg.status === "frame") {
    w.framePending = Math.max(0, w.framePending - 1);
    w.framesProduced = (w.framesProduced || 0) + 1;
    if (process.env.DEBUG_FRAMES) {
      console.log(`[worker ${w.gpu}] frame ${w.framesProduced} ${msg.gen_time_ms}ms ${msg.width}x${msg.height}`);
    }
    // Drop frame if WebRTC channel is congested. Live VJ wants the freshest
    // frame; stale frames in flight are useless and would wedge the channel.
    if (activeChannel?.readyState === "open") {
      const buffered = activeChannel.bufferedAmount || 0;
      if (buffered > MAX_OUTBOUND_BUFFER) {
        droppedOutbound++;
        if (droppedOutbound % 10 === 1) {
          console.log(`[dispatch] dropped frame from worker ${w.gpu} (channel buffer ${buffered} > ${MAX_OUTBOUND_BUFFER}; total dropped=${droppedOutbound})`);
        }
        return;
      }
      const imgBuffer = Buffer.from(msg.image_base64, "base64");
      activeChannel.send(imgBuffer);
      activeChannel.send(JSON.stringify({
        type: "stats",
        gen_time_ms: msg.gen_time_ms,
        width: msg.width,
        height: msg.height,
        worker: w.gpu,
      }));
    }
    return;
  }
  if (msg.status === "error") {
    console.error(`[worker ${w.gpu} ERROR]`, msg.message);
    return;
  }
  if (msg.status === "shutdown") {
    console.log(`[worker ${w.gpu}] shutdown`);
    w.ready = false;
    return;
  }
}

// Pending requests buffered until at least one worker is ready
const pendingBootstrap = [];

function flushPendingForWorker(w) {
  if (pendingBootstrap.length === 0) return;
  console.log(`[dispatch] flushing ${pendingBootstrap.length} pending requests`);
  // On first-ready-worker, replay buffered state changes to all ready workers.
  // Pending frames just get sent normally (we may lose a few that piled up
  // during startup — that's OK, frames are idempotent).
  const buf = pendingBootstrap.splice(0);
  for (const r of buf) sendToInference(r);
}

function nextReadyWorker() {
  if (workers.length === 0) return null;
  const start = roundRobinIdx;
  for (let i = 0; i < workers.length; i++) {
    const w = workers[(start + i) % workers.length];
    if (w.ready) {
      roundRobinIdx = (start + i + 1) % workers.length;
      return w;
    }
  }
  return null;
}

function broadcastState(stateOnly) {
  if (Object.keys(stateOnly).length === 0) return;
  // Remember every state field we've ever seen so a worker that comes up
  // late can be brought up to spec on its READY event (see handleWorkerLine).
  Object.assign(latestState, stateOnly);
  const line = JSON.stringify(stateOnly) + "\n";
  for (const w of workers) {
    if (w.ready) {
      try { w.proc.stdin.write(line); }
      catch (e) { console.error(`[worker ${w.gpu}] stdin write failed:`, e.message); }
    }
  }
}

function dispatchFrame(frameMsg) {
  const w = nextReadyWorker();
  if (!w) return false;
  try {
    const ok = w.proc.stdin.write(JSON.stringify(frameMsg) + "\n");
    w.framePending++;
    w.framesDispatched = (w.framesDispatched || 0) + 1;
    if (process.env.DEBUG_FRAMES) {
      console.log(`[dispatch] frame ${w.framesDispatched} → worker ${w.gpu} (pending=${w.framePending}${ok ? '' : ', backpressured'})`);
    }
    return true;
  } catch (e) {
    console.error(`[worker ${w.gpu}] stdin write failed:`, e.message);
    w.ready = false;
    return false;
  }
}

// Public: send a client request through the dispatcher. Splits state vs frame.
function sendToInference(req) {
  if (!req || typeof req !== "object") return;

  // Buffer until a worker is ready
  if (!workers.some(w => w.ready)) {
    pendingBootstrap.push(req);
    return;
  }

  // Extract state fields. Broadcast to all workers so each stays in sync.
  const stateOnly = {};
  for (const k of STATE_FIELDS) {
    if (k in req) stateOnly[k] = req[k];
  }
  if (Object.keys(stateOnly).length > 0) {
    broadcastState(stateOnly);
  }

  if (req.image_base64) {
    // Frame data — route to one worker round-robin.
    // The state was already broadcast above, so we send frame-only to avoid
    // redundant state updates eating stdin bandwidth.
    const frameMsg = { image_base64: req.image_base64 };
    dispatchFrame(frameMsg);
  }
}

// Drop pending frames buffered before any worker was ready (used when
// settings change to avoid stale frame replay).
function clearPendingFrames() {
  const before = pendingBootstrap.length;
  for (let i = pendingBootstrap.length - 1; i >= 0; i--) {
    if (pendingBootstrap[i].image_base64) pendingBootstrap.splice(i, 1);
  }
  const dropped = before - pendingBootstrap.length;
  if (dropped > 0) console.log(`[dispatch] dropped ${dropped} pending frames (settings changed)`);
}

// ---------------- WebRTC layer (unchanged) ---------------- //

let activePc = null;
let activeChannel = null;
let disconnectTimer = null;

function closeActivePc() {
  if (disconnectTimer) {
    clearTimeout(disconnectTimer);
    disconnectTimer = null;
  }
  if (activeChannel) {
    try { activeChannel.close(); } catch {}
    activeChannel = null;
  }
  if (activePc) {
    try { activePc.close(); } catch {}
    activePc = null;
  }
}

function waitForIceGatheringComplete(pc, timeoutMs) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener("icegatheringstatechange", onState);
      clearTimeout(timer);
      resolve();
    };
    const onState = () => { if (pc.iceGatheringState === "complete") finish(); };
    const timer = setTimeout(finish, timeoutMs);
    pc.addEventListener("icegatheringstatechange", onState);
  });
}

const app = express();
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});
app.use(express.json({ limit: "10mb" }));

app.get("/healthz", (_req, res) => {
  const readyCount = workers.filter(w => w.ready).length;
  res.json({
    ok: true,
    workerCount: workers.length,
    readyCount,
    inferenceReady: readyCount > 0,
    workers: workers.map(w => ({ gpu: w.gpu, ready: w.ready, framePending: w.framePending })),
  });
});

app.post("/webrtc/offer", async (req, res) => {
  const body = req.body || {};
  console.log("POST /webrtc/offer");
  if (body?.sdp?.sdp && typeof body.sdp.sdp === "string") {
    const offerSdp = body.sdp.sdp;
    const lines = offerSdp.split("\n");
    let host = 0, srflx = 0, relay = 0;
    for (const l of lines) {
      if (!l.startsWith("a=candidate:")) continue;
      if (l.includes(" typ host")) host++;
      else if (l.includes(" typ srflx")) srflx++;
      else if (l.includes(" typ relay")) relay++;
    }
    console.log(`Offer candidates host=${host} srflx=${srflx} relay=${relay}`);
  }
  if (!body.sdp || typeof body.sdp.type !== "string" || typeof body.sdp.sdp !== "string") {
    res.status(400).send("Expected body: { sdp: RTCSessionDescriptionInit }");
    return;
  }

  closeActivePc();

  const pc = new wrtc.RTCPeerConnection({ iceServers: getIceServers() });
  activePc = pc;

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    console.log("Connection state:", s);
    if (s === "connected" || s === "completed") {
      if (disconnectTimer) { clearTimeout(disconnectTimer); disconnectTimer = null; }
      return;
    }
    if (s === "disconnected") {
      if (disconnectTimer) clearTimeout(disconnectTimer);
      disconnectTimer = setTimeout(() => {
        if (activePc === pc && pc.connectionState === "disconnected") {
          console.log("Connection remained disconnected, closing peer");
          closeActivePc();
        }
        disconnectTimer = null;
      }, 5000);
      return;
    }
    if (s === "failed" || s === "closed") {
      if (activePc === pc) closeActivePc();
    }
  };

  pc.oniceconnectionstatechange = () => console.log("ICE connection state:", pc.iceConnectionState);
  pc.onicegatheringstatechange = () => console.log("ICE gathering state:", pc.iceGatheringState);

  pc.ondatachannel = (event) => {
    const channel = event.channel;
    channel.binaryType = "arraybuffer";
    activeChannel = channel;
    console.log("DataChannel opened:", channel.label);

    channel.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.prompt || msg.seed != null || msg.width || msg.height) clearPendingFrames();
          sendToInference(msg);
        } catch {
          console.log("Invalid JSON from client");
        }
      } else {
        const buffer = Buffer.from(ev.data);
        const base64 = buffer.toString("base64");
        sendToInference({ image_base64: base64 });
      }
    };
    channel.onclose = () => {
      console.log("DataChannel closed");
      if (activeChannel === channel) activeChannel = null;
    };
  };

  try {
    await pc.setRemoteDescription(body.sdp);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await waitForIceGatheringComplete(pc, ICE_GATHER_TIMEOUT_MS);
    if (!pc.localDescription) {
      res.status(500).send("Missing localDescription");
      return;
    }
    console.log("Sending SDP answer");
    res.json({ sdp: pc.localDescription });
  } catch (err) {
    if (activePc === pc) closeActivePc();
    res.status(500).send(err instanceof Error ? err.message : "WebRTC error");
  }
});

// ---------------- bootstrap ---------------- //

console.log(`VJ FLUX.2 Klein WebRTC server starting (workers=${WORKER_COUNT})`);
for (let i = 0; i < WORKER_COUNT; i++) spawnWorker(i);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Listening on 0.0.0.0:${PORT} (workers=${WORKER_COUNT})`);
});

// Graceful shutdown
function shutdown() {
  console.log("Shutting down...");
  for (const w of workers) {
    try { w.proc.stdin.write(JSON.stringify({ command: "shutdown" }) + "\n"); }
    catch {}
  }
  setTimeout(() => process.exit(0), 2000);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
