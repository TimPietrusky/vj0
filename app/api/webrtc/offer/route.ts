import { NextRequest } from "next/server";

const workerEndpoint = process.env.VJ0_WEBRTC_SIGNALING_ENDPOINT;
const workerAuthBearerToken = process.env.VJ0_WEBRTC_WORKER_BEARER_TOKEN;

export async function POST(req: NextRequest) {
  if (!workerEndpoint) {
    return new Response("Signaling endpoint not configured", { status: 500 });
  }

  let offer: unknown;
  try {
    offer = await req.json();
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (workerAuthBearerToken) {
    headers.Authorization = `Bearer ${workerAuthBearerToken}`;
  }

  const res = await fetch(workerEndpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(offer),
    // never cache signaling
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    return new Response(
      `Remote signaling failed (${res.status}): ${text || res.statusText}`,
      { status: 502 }
    );
  }

  const answer = (await res.json()) as unknown;
  return Response.json(answer);
}

