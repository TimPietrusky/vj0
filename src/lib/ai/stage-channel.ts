/**
 * BroadcastChannel protocol between the VJ control page (/vj) and the audience
 * stage page (/vj/stage). Same-origin only — both tabs must be on the same host.
 *
 * The control page owns the WebRTC connection to the inference backend and
 * publishes each incoming JPEG frame + the active prompt to this channel. The
 * stage page subscribes and renders only the output at fullscreen, with a
 * fade-in overlay for prompt changes. No controls, no config UI — just visuals.
 */

export const STAGE_CHANNEL_NAME = "vj0-stage";

export type StageFrameMsg = {
  type: "frame";
  bytes: ArrayBuffer;
  width: number;
  height: number;
  genTimeMs?: number;
  // Incrementing counter so stage can detect dropped frames if it cares.
  seq: number;
};

export type StagePromptMsg = {
  type: "prompt";
  prompt: string;
};

export type StageConnectionMsg = {
  type: "connection";
  status: "idle" | "connecting" | "connected" | "disconnected" | "error";
};

export type StageHelloMsg = {
  type: "hello";
};

export type StageMsg =
  | StageFrameMsg
  | StagePromptMsg
  | StageConnectionMsg
  | StageHelloMsg;

export function openStageChannel(): BroadcastChannel | null {
  if (typeof window === "undefined") return null;
  if (typeof BroadcastChannel === "undefined") return null;
  return new BroadcastChannel(STAGE_CHANNEL_NAME);
}
