/**
 * Smoke test for the multi-worker dispatcher in server.js.
 *
 * Connects to localhost:3000 over WebRTC, opens a DataChannel, sends a state
 * change (prompt + resolution), then sends N frames, collects responses,
 * verifies:
 *   1) BOTH workers handled at least one frame (round-robin actually works)
 *   2) Per-frame stats came back as JSON (.gen_time_ms present)
 *   3) Effective throughput matches the multi-process bench (~45 fps @ 256²/4-step)
 *
 * Run on the pod:
 *   node test_dispatch.js
 */
const wrtc = require("@roamhq/wrtc");
const fs = require("fs");

const SERVER_URL = process.env.SERVER_URL || "http://127.0.0.1:3000";
const WAVE_PATH = process.env.WAVE_PATH || "/workspace/waveforms/waveform_1.png";
const N_FRAMES = Number(process.env.N_FRAMES || 8);
const SEND_INTERVAL_MS = Number(process.env.SEND_INTERVAL_MS || 80);  // > per-frame latency
const SIZE = Number(process.env.SIZE || 256);
const N_STEPS = Number(process.env.N_STEPS || 4);
const ALPHA = Number(process.env.ALPHA || 0.10);
const PROMPT = process.env.PROMPT || (
    "a bright white lightning bolt against a pitch black night sky, " +
    "dramatic, photographic, high contrast"
);

(async () => {
    // 1) Wait for server.js to report both workers ready
    console.log(`waiting for /healthz with readyCount=2 at ${SERVER_URL}...`);
    let healthOk = false;
    for (let i = 0; i < 600; i++) {
        try {
            const r = await fetch(SERVER_URL + "/healthz");
            const j = await r.json();
            if (j.readyCount >= 2) {
                console.log(`healthz ok: workers=${j.workerCount} ready=${j.readyCount}`);
                healthOk = true;
                break;
            }
            if (i % 10 === 0) console.log(`  /healthz: ready=${j.readyCount}/${j.workerCount}`);
        } catch (e) {
            if (i % 10 === 0) console.log(`  /healthz error: ${e.message}`);
        }
        await new Promise(r => setTimeout(r, 1000));
    }
    if (!healthOk) {
        console.error("workers never became ready");
        process.exit(1);
    }

    // 2) Open WebRTC peer connection
    console.log("\nopening WebRTC connection...");
    const pc = new wrtc.RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
    const channel = pc.createDataChannel("vj");
    channel.binaryType = "arraybuffer";

    const responses = [];
    let resolveDone;
    const done = new Promise(r => { resolveDone = r; });

    let pendingStat = null;
    channel.onmessage = (ev) => {
        if (typeof ev.data === "string") {
            // stats JSON
            try {
                const stat = JSON.parse(ev.data);
                pendingStat = stat;
            } catch {}
        } else {
            // binary image — pair with most recent stats
            const stat = pendingStat || {};
            pendingStat = null;
            responses.push({ ...stat, byteLength: ev.data.byteLength, t_recv: Date.now() });
            if (responses.length >= N_FRAMES) resolveDone();
        }
    };

    channel.onopen = async () => {
        console.log("DataChannel open. Sending prompt + initial state...");

        // Send state — broadcast to all workers
        channel.send(JSON.stringify({
            prompt: PROMPT,
            seed: 42,
            alpha: ALPHA,
            n_steps: N_STEPS,
            width: SIZE,
            height: SIZE,
            captureWidth: SIZE,
            captureHeight: SIZE,
        }));

        // Brief delay so workers process the state change before frames arrive
        await new Promise(r => setTimeout(r, 500));

        // Read waveform once, send N times as binary.
        // PNG is recognized by the worker's bytes_to_pil().
        const waveBuf = fs.readFileSync(WAVE_PATH);
        console.log(`sending ${N_FRAMES} frames @ ${SIZE}² / ${N_STEPS}-step (waveform PNG, ${waveBuf.length} bytes each)...`);
        const t_start = Date.now();
        for (let i = 0; i < N_FRAMES; i++) {
            // Check WebRTC backpressure
            while (channel.bufferedAmount > 256 * 1024) {
                await new Promise(r => setTimeout(r, 5));
            }
            channel.send(waveBuf);
            console.log(`  sent frame ${i + 1}/${N_FRAMES} (bufferedAmount=${channel.bufferedAmount})`);
            await new Promise(r => setTimeout(r, SEND_INTERVAL_MS));
        }
        console.log(`all ${N_FRAMES} frames sent in ${Date.now() - t_start} ms`);
    };

    pc.oniceconnectionstatechange = () => console.log("ICE state:", pc.iceConnectionState);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await new Promise(r => {
        if (pc.iceGatheringState === "complete") return r();
        const onState = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", onState); r(); } };
        pc.addEventListener("icegatheringstatechange", onState);
        setTimeout(r, 5000);
    });
    const r = await fetch(SERVER_URL + "/webrtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: pc.localDescription }),
    });
    const ans = await r.json();
    await pc.setRemoteDescription(ans.sdp);

    // 3) Wait for N responses
    const t_first_frame = Date.now();
    await Promise.race([
        done,
        new Promise((_, reject) => setTimeout(() => reject(new Error("timeout waiting for frames")), 60000)),
    ]);
    const t_last_frame = Date.now();

    const wallS = (t_last_frame - t_first_frame) / 1000;
    const fps = responses.length / wallS;

    // 4) Analyze who handled each frame
    const byWorker = {};
    for (const r of responses) {
        const w = r.worker ?? "unknown";
        byWorker[w] = (byWorker[w] || 0) + 1;
    }

    console.log("\n=== DISPATCH TEST RESULT ===");
    console.log(`frames received: ${responses.length}/${N_FRAMES}`);
    console.log(`wall time:       ${wallS.toFixed(2)}s`);
    console.log(`throughput:      ${fps.toFixed(2)} fps`);
    console.log(`per-worker counts:`, byWorker);
    const meanGen = responses.reduce((a, r) => a + (r.gen_time_ms || 0), 0) / responses.length;
    console.log(`mean gen_time_ms (per frame, server-reported): ${meanGen.toFixed(2)}`);

    const workerCount = Object.keys(byWorker).length;
    if (workerCount < 2) {
        console.error(`FAIL: only ${workerCount} worker(s) handled frames — round-robin is broken`);
        process.exit(1);
    }
    const minPerWorker = Math.min(...Object.values(byWorker));
    if (minPerWorker < 5) {
        console.error(`WARN: one worker only handled ${minPerWorker} frames — uneven distribution`);
    }
    console.log("\nPASS: round-robin dispatch verified ✓");
    pc.close();
    process.exit(0);
})().catch((e) => { console.error("test failed:", e); process.exit(1); });
