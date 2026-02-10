const express = require("express");
const wrtc = require("wrtc");

const PORT = Number(process.env.PORT || 3000);

/**
 * PoC assumptions:
 * - single client at a time
 * - non-trickle ICE (offer/answer contains candidates)
 * - STUN by default; TURN supported via env
 */

function getIceServers() {
  // Provide as JSON: [{"urls":"turns:turn.example.com:5349?transport=tcp","username":"...","credential":"..."}]
  const raw = process.env.ICE_SERVERS_JSON;
  if (!raw) return [{ urls: "stun:stun.l.google.com:19302" }];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
  } catch {
    // ignore
  }
  return [{ urls: "stun:stun.l.google.com:19302" }];
}

function getIceTransportPolicy() {
  // "relay" forces TURN; "all" allows STUN/direct candidates
  const raw = process.env.ICE_TRANSPORT_POLICY;
  if (raw === "relay" || raw === "all") return raw;
  return undefined;
}

/** @type {import("wrtc").RTCPeerConnection | null} */
let activePc = null;

function closeActivePc() {
  if (!activePc) return;
  try {
    activePc.close();
  } catch {
    // ignore
  }
  activePc = null;
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

    const onState = () => {
      if (pc.iceGatheringState === "complete") finish();
    };

    const timer = setTimeout(finish, timeoutMs);
    pc.addEventListener("icegatheringstatechange", onState);
  });
}

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/healthz", (_req, res) => {
  res.json({ ok: true });
});

app.post("/webrtc/offer", async (req, res) => {
  /** @type {{ sdp?: import("wrtc").RTCSessionDescriptionInit }} */
  const body = req.body || {};

  if (
    !body.sdp ||
    typeof body.sdp.type !== "string" ||
    typeof body.sdp.sdp !== "string"
  ) {
    res.status(400).send("Expected body: { sdp: RTCSessionDescriptionInit }");
    return;
  }

  // single-client PoC: replace any existing connection
  closeActivePc();

  const pc = new wrtc.RTCPeerConnection({
    iceServers: getIceServers(),
    iceTransportPolicy: getIceTransportPolicy(),
  });
  activePc = pc;

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    if (s === "failed" || s === "disconnected" || s === "closed") {
      if (activePc === pc) closeActivePc();
    }
  };

  pc.ondatachannel = (event) => {
    const channel = event.channel;
    channel.binaryType = "arraybuffer";

    channel.onmessage = (ev) => {
      // Echo exactly what we received (string JSON or binary)
      try {
        channel.send(ev.data);
      } catch {
        // ignore
      }
    };
  };

  try {
    await pc.setRemoteDescription(body.sdp);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    await waitForIceGatheringComplete(pc, 1500);

    if (!pc.localDescription) {
      res.status(500).send("Missing localDescription after createAnswer()");
      return;
    }

    res.json({ sdp: pc.localDescription });
  } catch (err) {
    if (activePc === pc) closeActivePc();
    res.status(500).send(err instanceof Error ? err.message : "WebRTC error");
  }
});

app.listen(PORT, "0.0.0.0", () => {
  // eslint-disable-next-line no-console
  console.log(`vj0 webrtc echo worker listening on 0.0.0.0:${PORT}`);
});

