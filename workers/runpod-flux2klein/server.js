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

// Prevent node from crashing on unhandled errors — log and continue.
// WebRTC peer teardown and worker pipe errors are the usual culprits.
process.on("uncaughtException", (err) => {
  console.error("[UNCAUGHT]", err.message, err.stack);
});
process.on("unhandledRejection", (reason) => {
  console.error("[UNHANDLED_REJECTION]", reason);
});

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

// Latest known compile status per worker. Used to replay state to a client
// that connects mid-compile (so the overlay shows up immediately instead of
// staring at a frozen "connected" canvas with no idea why nothing is happening).
const latestCompileByWorker = new Map();

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
  const w = { proc, gpu, ready: false, stdoutBuf: "", framePending: 0, lastFrameAt: 0 };
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
    console.log(`[worker ${gpu}] exited code=${code}, respawning in 3s...`);
    w.ready = false;
    w.framePending = 0;
    // Auto-respawn: remove dead worker, spawn fresh one after a brief delay
    // so the GPU has time to release resources.
    setTimeout(() => {
      const idx = workers.indexOf(w);
      if (idx >= 0) workers.splice(idx, 1);
      console.log(`[worker ${gpu}] respawning now`);
      spawnWorker(gpu);
    }, 3000);
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
  // Boot phase events from inference_server.py (loading_weights, applying_fp8,
  // registering_compile_stubs, loaded). Forward to the WebRTC client so the
  // overlay can show "Loading the AI model" during the ~140s safetensors load
  // instead of a silent "preparing workers" placeholder.
  if (msg.status === "phase") {
    console.log(`[worker ${w.gpu}] phase=${msg.stage} est=${msg.est_seconds || "?"}s`);
    if (activeChannel?.readyState === "open") {
      activeChannel.send(JSON.stringify({
        type: "phase",
        stage: msg.stage,
        est_seconds: msg.est_seconds,
        elapsed_ms: msg.elapsed_ms,
        worker: w.gpu,
      }));
    }
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
  // Also remember the latest status per worker so a client that connects
  // mid-compile gets the overlay immediately (replayed in pc.ondatachannel).
  if (msg.status === "compiling" || msg.status === "compiling_progress" || msg.status === "warmed") {
    if (msg.status === "compiling") {
      console.log(`[worker ${w.gpu}] compiling ${msg.width}x${msg.height} (~${msg.est_seconds}s)`);
    } else if (msg.status === "warmed") {
      console.log(`[worker ${w.gpu}] warmed ${msg.width}x${msg.height} in ${msg.total_ms}ms`);
    }
    const payload = {
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
    };
    if (msg.status === "warmed") {
      // Worker is ready at this shape — clear the "currently compiling" memo.
      latestCompileByWorker.delete(w.gpu);
    } else {
      latestCompileByWorker.set(w.gpu, payload);
    }
    if (activeChannel?.readyState === "open") {
      activeChannel.send(JSON.stringify(payload));
    }
    return;
  }

  if (msg.status === "frame") {
    w.framePending = Math.max(0, w.framePending - 1);
    w.framesProduced = (w.framesProduced || 0) + 1;
    w.lastFrameAt = Date.now();
    diagStats.framesFromWorker++;
    diagStats.lastWorkerFrameAt = Date.now();
    if (process.env.DEBUG_FRAMES) {
      console.log(`[worker ${w.gpu}] frame ${w.framesProduced} ${msg.gen_time_ms}ms ${msg.width}x${msg.height}`);
    }
    // Drop frame if WebRTC channel is congested. Live VJ wants the freshest
    // frame; stale frames in flight are useless and would wedge the channel.
    if (activeChannel?.readyState === "open") {
      const buffered = activeChannel.bufferedAmount || 0;
      diagStats.channelBufferedAmount = buffered;
      diagStats.channelState = activeChannel.readyState;
      if (buffered > MAX_OUTBOUND_BUFFER) {
        droppedOutbound++;
        diagStats.droppedOutbound++;
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
        // Per-stage breakdown from inference_server.py — lets the UI show
        // where the latency budget is going (vae encode vs transformer vs jpeg).
        timing: msg.timing,
      }));
      diagStats.framesToClient++;
      diagStats.lastSentToClientAt = Date.now();
    } else {
      // Channel not open — log it so we can see if frames are being produced
      // but can't be delivered.
      if ((w.framesProduced % 50) === 1) {
        console.log(`[DIAG] frame from worker ${w.gpu} but channel=${activeChannel?.readyState || "null"}`);
      }
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

// Per-worker dispatch limit. Worker's internal request_queue is maxsize=2,
// so anything past 2 in flight gets silently dropped at the worker. Setting
// the dispatcher cap to 3 (one in flight + two queued) gives the worker
// just enough lookahead without wasting IPC bandwidth on frames that would
// be dropped anyway. Past this, dispatchFrame() rejects at the source.
const MAX_PENDING_PER_WORKER = 3;
let droppedInbound = 0;

// Load-aware "next worker" selector. Strict round-robin starves the dispatcher
// when one worker is briefly slower (recompile, GPU contention) — the slow
// worker's stdin queue grows while the other sits idle. Picking the worker
// with the FEWEST pending frames is robust to those transients. Tie-break
// preserves round-robin so equal-load workers still alternate predictably.
function nextReadyWorker() {
  if (workers.length === 0) return null;
  let best = null;
  let bestIdx = -1;
  for (let i = 0; i < workers.length; i++) {
    const idx = (roundRobinIdx + i) % workers.length;
    const w = workers[idx];
    if (!w.ready) continue;
    if (best === null || w.framePending < best.framePending) {
      best = w;
      bestIdx = idx;
    }
  }
  if (best !== null) {
    // Advance the round-robin head past the winner so equal-load ties
    // alternate next time we're called.
    roundRobinIdx = (bestIdx + 1) % workers.length;
  }
  return best;
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
  // Drop early if the chosen worker is already at saturation. Worker's
  // request_queue (maxsize=2) would silently drop this anyway — better to
  // skip the JSON encode + stdin round-trip + bytes over the pipe.
  if (w.framePending >= MAX_PENDING_PER_WORKER) {
    droppedInbound++;
    diagStats.droppedInbound++;
    if (droppedInbound % 10 === 1) {
      console.log(`[dispatch] dropped frame (worker ${w.gpu} saturated, pending=${w.framePending}, total dropped=${droppedInbound})`);
    }
    return false;
  }
  try {
    const ok = w.proc.stdin.write(JSON.stringify(frameMsg) + "\n");
    w.framePending++;
    w.framesDispatched = (w.framesDispatched || 0) + 1;
    diagStats.framesToWorker++;
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
  // Reset worker pending counts — in-flight frames from the dying connection
  // are stale. Results that trickle back will harmlessly clamp to 0 via
  // Math.max(0, framePending - 1). Without this reset, workers stay at
  // MAX_PENDING and dispatchFrame() drops every new frame → deadlock.
  for (const w of workers) {
    if (w.framePending > 0) {
      console.log(`[worker ${w.gpu}] reset framePending ${w.framePending} → 0 (connection replaced)`);
      w.framePending = 0;
    }
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
    console.log(`[DIAG] Connection state: ${s} (channel=${activeChannel?.readyState || "none"} buf=${activeChannel?.bufferedAmount || 0})`);
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
        diagStats.framesFromClient++;
        diagStats.lastClientFrameAt = Date.now();
        const buffer = Buffer.from(ev.data);
        const base64 = buffer.toString("base64");
        sendToInference({ image_base64: base64 });
      }
    };
    channel.onclose = () => {
      console.log(`[DIAG] DataChannel CLOSED (was ${diagStats.channelState})`);
      diagStats.channelState = "closed";
      if (activeChannel === channel) activeChannel = null;
    };
    channel.onerror = (err) => {
      console.log(`[DIAG] DataChannel ERROR: ${err?.error?.message || err?.message || err}`);
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

// ---------------- diagnostics ---------------- //
// Tracks frame flow, event loop health, memory, and DataChannel state.
// All data exposed via /debug endpoint and logged every DIAG_INTERVAL_MS.

const DIAG_INTERVAL_MS = 5000; // dump stats every 5s
const diagStats = {
  framesFromClient: 0,       // binary frames received from WebRTC client
  framesToWorker: 0,         // frames dispatched to worker stdin
  framesFromWorker: 0,       // frame results from worker stdout
  framesToClient: 0,         // frames sent back over WebRTC channel
  droppedInbound: 0,         // dropped because worker saturated
  droppedOutbound: 0,        // dropped because channel congested
  lastClientFrameAt: 0,      // when we last got a frame from the client
  lastWorkerFrameAt: 0,      // when a worker last produced a frame
  lastSentToClientAt: 0,     // when we last sent a frame to the client
  channelBufferedAmount: 0,  // latest DataChannel bufferedAmount
  channelState: "none",      // latest DataChannel readyState
  eventLoopLagMs: 0,         // max event loop lag since last dump
  eventLoopLagTotal: 0,      // accumulated lag for average
  eventLoopChecks: 0,
  startedAt: Date.now(),
};

// Event loop lag detector — fires every 500ms, measures actual vs expected delay.
// If the event loop is blocked (GC, large JSON parse, etc.), the delay exceeds 500ms.
let _lastLoopCheck = Date.now();
setInterval(() => {
  const now = Date.now();
  const expected = 500;
  const actual = now - _lastLoopCheck;
  const lag = Math.max(0, actual - expected);
  if (lag > diagStats.eventLoopLagMs) diagStats.eventLoopLagMs = lag;
  diagStats.eventLoopLagTotal += lag;
  diagStats.eventLoopChecks++;
  _lastLoopCheck = now;
  // Alert on severe lag (>500ms = half-second event loop stall)
  if (lag > 500) {
    console.log(`[DIAG] ⚠️  EVENT LOOP LAG ${lag}ms`);
  }
}, 500);

// Periodic state dump
setInterval(() => {
  const now = Date.now();
  const uptimeS = ((now - diagStats.startedAt) / 1000).toFixed(0);
  const mem = process.memoryUsage();
  const heapMB = (mem.heapUsed / 1024 / 1024).toFixed(1);
  const rssMB = (mem.rss / 1024 / 1024).toFixed(1);
  const avgLag = diagStats.eventLoopChecks > 0
    ? (diagStats.eventLoopLagTotal / diagStats.eventLoopChecks).toFixed(1)
    : "0";

  const workerSummary = workers.map(w =>
    `gpu${w.gpu}:${w.ready ? "R" : "x"} pend=${w.framePending} frames=${w.framesProduced || 0} buf=${w.stdoutBuf.length}`
  ).join(" | ");

  const sinceClient = diagStats.lastClientFrameAt ? ((now - diagStats.lastClientFrameAt) / 1000).toFixed(1) : "never";
  const sinceWorker = diagStats.lastWorkerFrameAt ? ((now - diagStats.lastWorkerFrameAt) / 1000).toFixed(1) : "never";
  const sinceSent = diagStats.lastSentToClientAt ? ((now - diagStats.lastSentToClientAt) / 1000).toFixed(1) : "never";

  console.log(
    `[DIAG] t=${uptimeS}s ` +
    `in=${diagStats.framesFromClient} →wk=${diagStats.framesToWorker} ←wk=${diagStats.framesFromWorker} →cl=${diagStats.framesToClient} ` +
    `drop_in=${diagStats.droppedInbound} drop_out=${diagStats.droppedOutbound} | ` +
    `since: client=${sinceClient}s worker=${sinceWorker}s sent=${sinceSent}s | ` +
    `ch=${diagStats.channelState} buf=${diagStats.channelBufferedAmount} | ` +
    `lag: max=${diagStats.eventLoopLagMs}ms avg=${avgLag}ms | ` +
    `heap=${heapMB}MB rss=${rssMB}MB | ` +
    `${workerSummary}`
  );

  // Reset per-interval peaks
  diagStats.eventLoopLagMs = 0;
}, DIAG_INTERVAL_MS);

// /debug endpoint — live JSON snapshot for quick checks
app.get("/debug", (_req, res) => {
  const now = Date.now();
  res.json({
    uptime_s: ((now - diagStats.startedAt) / 1000).toFixed(0),
    stats: { ...diagStats },
    memory: process.memoryUsage(),
    workers: workers.map(w => ({
      gpu: w.gpu,
      ready: w.ready,
      framePending: w.framePending,
      framesProduced: w.framesProduced || 0,
      framesDispatched: w.framesDispatched || 0,
      lastFrameAt: w.lastFrameAt,
      stdoutBufLen: w.stdoutBuf.length,
    })),
    channel: activeChannel ? {
      readyState: activeChannel.readyState,
      bufferedAmount: activeChannel.bufferedAmount,
      label: activeChannel.label,
    } : null,
    connection: activePc ? {
      connectionState: activePc.connectionState,
      iceConnectionState: activePc.iceConnectionState,
    } : null,
  });
});

// ---------------- bootstrap ---------------- //

console.log(`VJ FLUX.2 Klein WebRTC server starting (workers=${WORKER_COUNT})`);
for (let i = 0; i < WORKER_COUNT; i++) spawnWorker(i);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Listening on 0.0.0.0:${PORT} (workers=${WORKER_COUNT})`);
});

// ---------------- watchdog ---------------- //
// If a worker has pending frames but hasn't produced output in 30s, it's hung
// (typically a CUDA graph stall). Kill the process — the "close" handler
// auto-respawns it on the same GPU. The other worker keeps serving while the
// dead one reboots (~30-60s warm, ~3 min cold).
const WATCHDOG_INTERVAL_MS = 5000;
const WATCHDOG_STALL_MS = 30000;

setInterval(() => {
  const now = Date.now();
  for (const w of workers) {
    if (!w.ready || w.framePending === 0) continue;
    // lastFrameAt is 0 before the first frame — skip (still compiling).
    if (w.lastFrameAt === 0) continue;
    const stalled = now - w.lastFrameAt;
    if (stalled > WATCHDOG_STALL_MS) {
      console.log(`[watchdog] worker ${w.gpu} stalled ${(stalled / 1000).toFixed(1)}s with pending=${w.framePending} stdoutBuf=${w.stdoutBuf.length}, killing for respawn`);
      w.ready = false;
      w.framePending = 0;
      try { w.proc.kill("SIGKILL"); } catch {}
    }
    // Warn if stdout buffer is growing — means worker is writing faster than
    // we parse, or a partial JSON line is stuck (incomplete newline = pipe stall)
    if (w.stdoutBuf.length > 512 * 1024) {
      console.log(`[DIAG] ⚠️  worker ${w.gpu} stdoutBuf=${(w.stdoutBuf.length / 1024).toFixed(0)}KB — possible pipe stall`);
    }
  }
}, WATCHDOG_INTERVAL_MS);

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
