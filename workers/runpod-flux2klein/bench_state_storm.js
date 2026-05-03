#!/usr/bin/env node
/**
 * State-storm stress test — hammers prompt+seed changes over WebRTC
 * while continuously sending frames, exactly like the browser does
 * when switching presets rapidly.
 *
 * This reproduces the crash the user sees: both workers die when
 * prompt/seed change rapidly. The test logs exactly what kills them.
 *
 * Usage (on the pod, while server.js is running):
 *   node bench_state_storm.js
 *   node bench_state_storm.js --settings-interval 500    # change every 500ms
 *   node bench_state_storm.js --settings-interval 100    # aggressive
 */
const wrtc = require("@roamhq/wrtc");

const args = parseArgs();
const WIDTH = args.width || 448;
const HEIGHT = args.height || 768;
const FRAMES = args.frames || 10000;
const SERVER_URL = args.server || "http://localhost:3000";
const SEND_INTERVAL_MS = args.interval || 50;        // frame send rate ~20fps
const SETTINGS_INTERVAL_MS = args.settingsInterval || 1000;  // change prompt+seed every N ms
const HANG_TIMEOUT_MS = args.hangTimeout || 30000;
const ALPHA = args.alpha || 0.10;
const N_STEPS = args.nSteps || 4;

const PROMPTS = [
  "vibrant neon cyberpunk city street at night, rain, reflections",
  "abstract flowing liquid metal, chrome, iridescent rainbow",
  "deep ocean underwater scene, bioluminescent creatures, dark blue",
  "burning fire tornado in desert, orange sparks, smoke",
  "frozen ice cave, crystal formations, blue light",
  "psychedelic fractal patterns, vivid colors, kaleidoscope",
  "dense jungle canopy, sunbeams, mist, tropical birds",
  "electric lightning storm, purple sky, dark clouds",
  "aurora borealis over snowy mountains, green glow",
  "molten lava flowing into ocean, steam, red glow",
];

function parseArgs() {
  const result = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, "").replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    result[key] = isNaN(Number(argv[i + 1])) ? argv[i + 1] : Number(argv[i + 1]);
  }
  return result;
}

function makeSyntheticFrame(w, h) {
  const { execSync } = require("child_process");
  const jpegBuf = execSync(
    `python3 -c "
import sys, io, numpy as np
from PIL import Image
arr = np.random.randint(0, 255, (${h}, ${w}, 3), dtype=np.uint8)
img = Image.fromarray(arr)
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=85)
sys.stdout.buffer.write(buf.getvalue())
"`,
    { maxBuffer: 10 * 1024 * 1024 }
  );
  return jpegBuf;
}

async function main() {
  console.log(`=== State-storm stress test: ${WIDTH}x${HEIGHT}, ${FRAMES} frames ===`);
  console.log(`    server: ${SERVER_URL}`);
  console.log(`    frame interval: ${SEND_INTERVAL_MS}ms (~${(1000 / SEND_INTERVAL_MS).toFixed(0)} fps send)`);
  console.log(`    settings change interval: ${SETTINGS_INTERVAL_MS}ms`);
  console.log(`    prompts pool: ${PROMPTS.length}`);
  console.log(`    hang timeout: ${HANG_TIMEOUT_MS}ms`);
  console.log();

  const pc = new wrtc.RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  const channel = pc.createDataChannel("vj0");
  channel.binaryType = "arraybuffer";

  let framesReceived = 0;
  let framesSent = 0;
  let settingsChanges = 0;
  let lastRecvAt = 0;
  let hungAt = null;
  let currentPromptIdx = 0;
  let currentSeed = 42;
  const timings = [];
  const workerCrashes = [];

  channel.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      framesReceived++;
      lastRecvAt = Date.now();
      if (framesReceived <= 5 || framesReceived % 200 === 0) {
        console.log(`  recv frame ${framesReceived}: ${(ev.data.byteLength / 1024).toFixed(1)} KB`);
      }
    } else if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "stats") {
          timings.push(msg.timing || {});
          if (framesReceived <= 5 || framesReceived % 200 === 0) {
            const t = msg.timing || {};
            console.log(`    stats: total=${t.total_ms?.toFixed(1)}ms worker=${msg.worker}`);
          }
        } else if (msg.type === "phase") {
          console.log(`  [server] phase: ${msg.stage} worker=${msg.worker}`);
          workerCrashes.push({ time: Date.now(), event: "phase", stage: msg.stage, worker: msg.worker });
        } else if (msg.type === "compile") {
          console.log(`  [server] compile: ${msg.status} ${msg.width}x${msg.height} worker=${msg.worker}`);
          workerCrashes.push({ time: Date.now(), event: "compile", status: msg.status, worker: msg.worker });
        }
      } catch {}
    }
  };

  channel.onopen = () => console.log("DataChannel open");
  channel.onclose = () => console.log("DataChannel closed");
  pc.onconnectionstatechange = () => console.log(`Connection: ${pc.connectionState}`);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const check = () => {
      if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", check); resolve(); }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 10000);
  });

  console.log("Sending offer...");
  const resp = await fetch(`${SERVER_URL}/webrtc/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription }),
  });
  if (!resp.ok) { console.error(`Signaling failed: ${resp.status}`); process.exit(1); }
  const { sdp: answerSdp } = await resp.json();
  await pc.setRemoteDescription(answerSdp);

  await new Promise((resolve) => {
    if (channel.readyState === "open") return resolve();
    const orig = channel.onopen;
    channel.onopen = (...a) => { if (orig) orig(...a); resolve(); };
    setTimeout(() => { if (channel.readyState !== "open") process.exit(1); resolve(); }, 30000);
  });

  console.log("DataChannel ready, sending initial settings...");
  channel.send(JSON.stringify({
    prompt: PROMPTS[0],
    width: WIDTH, height: HEIGHT,
    captureWidth: WIDTH, captureHeight: HEIGHT,
    alpha: ALPHA, n_steps: N_STEPS, seed: currentSeed,
  }));

  await new Promise((r) => setTimeout(r, 500));

  console.log(`Generating synthetic frame (${WIDTH}x${HEIGHT})...`);
  const frameBuffer = makeSyntheticFrame(WIDTH, HEIGHT);
  console.log(`  JPEG: ${(frameBuffer.length / 1024).toFixed(1)} KB\n`);

  console.log(`Starting storm: frames every ${SEND_INTERVAL_MS}ms + settings every ${SETTINGS_INTERVAL_MS}ms\n`);
  const startTime = Date.now();
  lastRecvAt = Date.now();

  // Settings storm — change prompt + seed at fixed interval
  const settingsTimer = setInterval(() => {
    if (framesSent >= FRAMES || hungAt !== null) return;
    currentPromptIdx = (currentPromptIdx + 1) % PROMPTS.length;
    currentSeed = Math.floor(Math.random() * 999999);
    settingsChanges++;

    if (channel.readyState === "open") {
      channel.send(JSON.stringify({
        prompt: PROMPTS[currentPromptIdx],
        seed: currentSeed,
        // SAME resolution — no change
        width: WIDTH, height: HEIGHT,
        captureWidth: WIDTH, captureHeight: HEIGHT,
        alpha: ALPHA, n_steps: N_STEPS,
      }));
      if (settingsChanges <= 5 || settingsChanges % 50 === 0) {
        console.log(`  [settings #${settingsChanges}] prompt="${PROMPTS[currentPromptIdx].slice(0, 40)}..." seed=${currentSeed}`);
      }
    }
  }, SETTINGS_INTERVAL_MS);

  // Frame pump
  const sendFrame = () => {
    if (framesSent >= FRAMES || hungAt !== null) return;

    const sinceLast = Date.now() - lastRecvAt;
    if (framesReceived > 0 && sinceLast > HANG_TIMEOUT_MS) {
      hungAt = framesSent;
      console.log(`\n!!! HANG at frame ${framesSent} — no response in ${(sinceLast / 1000).toFixed(1)}s !!!`);
      console.log(`  settings changes: ${settingsChanges}, frames recv: ${framesReceived}`);
      finish();
      return;
    }

    if (channel.readyState === "open") {
      const buffered = channel.bufferedAmount || 0;
      if (buffered < 2 * 1024 * 1024) {
        channel.send(frameBuffer);
        framesSent++;
        if (framesSent <= 5 || framesSent % 1000 === 0) {
          console.log(`  sent frame ${framesSent} (buf=${(buffered / 1024).toFixed(0)}KB recv=${framesReceived} settings=${settingsChanges})`);
        }
      }
    }
    setTimeout(sendFrame, SEND_INTERVAL_MS);
  };
  sendFrame();

  // Also poll /debug every 10s
  const debugPoller = setInterval(async () => {
    try {
      const r = await fetch(`${SERVER_URL}/debug`);
      const d = await r.json();
      const ws = d.workers.map(w => `gpu${w.gpu}:${w.ready ? "R" : "X"} prod=${w.framesProduced} pend=${w.framePending}`).join(" | ");
      console.log(`  [debug] ${ws} | ch=${d.channel?.readyState || "null"} | lag=${d.stats.eventLoopLagMs}ms`);
      // Detect worker crash
      for (const w of d.workers) {
        if (!w.ready && w.framesProduced === 0 && framesSent > 100) {
          console.log(`  [debug] ⚠️  worker gpu${w.gpu} is DOWN (not ready, 0 frames produced)`);
        }
      }
    } catch {}
  }, 10000);

  const hangChecker = setInterval(() => {
    if (hungAt !== null) { clearInterval(hangChecker); return; }
    if (framesReceived > 0) {
      const sinceLast = Date.now() - lastRecvAt;
      if (sinceLast > HANG_TIMEOUT_MS) {
        hungAt = framesSent;
        console.log(`\n!!! HANG — no response in ${(sinceLast / 1000).toFixed(1)}s !!!`);
        clearInterval(hangChecker);
        finish();
      }
    }
  }, 5000);

  function finish() {
    clearInterval(hangChecker);
    clearInterval(settingsTimer);
    clearInterval(debugPoller);
    const elapsed = (Date.now() - startTime) / 1000;

    console.log(`\n${"=".repeat(60)}`);
    if (hungAt) {
      console.log(`RESULT: HUNG at sent frame ${hungAt}/${FRAMES}`);
    } else {
      console.log(`RESULT: All ${FRAMES} frames sent, ${framesReceived} received`);
    }
    console.log(`  Elapsed: ${elapsed.toFixed(1)}s`);
    console.log(`  Frames sent: ${framesSent}`);
    console.log(`  Frames received: ${framesReceived}`);
    console.log(`  Settings changes: ${settingsChanges}`);
    console.log(`  Drop rate: ${((1 - framesReceived / Math.max(1, framesSent)) * 100).toFixed(1)}%`);

    if (workerCrashes.length > 0) {
      console.log(`\n  Worker events (${workerCrashes.length}):`);
      for (const c of workerCrashes.slice(-20)) {
        const t = ((c.time - startTime) / 1000).toFixed(1);
        console.log(`    t=${t}s ${c.event}: ${c.stage || c.status || "?"} worker=${c.worker}`);
      }
    }

    if (timings.length > 0) {
      const totals = timings.map(t => t.total_ms || 0).filter(v => v > 0);
      if (totals.length > 0) {
        console.log(`\n  Inference timing (ms):`);
        console.log(`    total: mean=${(totals.reduce((a, b) => a + b, 0) / totals.length).toFixed(1)}  min=${Math.min(...totals).toFixed(1)}  max=${Math.max(...totals).toFixed(1)}`);
        console.log(`  Throughput: ${(framesReceived / elapsed).toFixed(1)} fps`);
      }
    }

    try { channel.close(); } catch {}
    try { pc.close(); } catch {}
    process.exit(hungAt ? 1 : 0);
  }

  const waitForDone = setInterval(() => {
    if (hungAt !== null) { clearInterval(waitForDone); return; }
    if (framesSent >= FRAMES) {
      setTimeout(() => { clearInterval(waitForDone); finish(); }, 10000);
      clearInterval(waitForDone);
    }
  }, 1000);
}

main().catch((err) => { console.error("FATAL:", err); process.exit(1); });
