"use client";

import { useEffect, useRef, useImperativeHandle, forwardRef } from "react";

/**
 * WebGL2 canvas that draws an incoming ImageBitmap upscaled with FSR 1
 * (AMD FidelityFX Super Resolution 1) — a two-pass spatial upscaler:
 *
 *   1. EASU (Edge-Adaptive Spatial Upsampling): edge-aware reconstruction.
 *      Detects local gradient orientation from a 12-tap kernel, then fits a
 *      Lanczos-2-shaped reconstruction kernel along the detected edge. Hard
 *      edges stay sharp instead of being smeared by bilinear, with explicit
 *      de-ringing via a clamp to the inner 2×2 quadrant range.
 *
 *   2. RCAS (Robust Contrast Adaptive Sharpening): final sharpening pass with
 *      explicit halo limits. Sharpens crisply without the ringing that naive
 *      unsharp mask creates on hard edges. Includes noise attenuation that
 *      reduces sharpening on already-noisy regions.
 *
 * Why FSR over the previous bilinear+unsharp pass: FSR is purely spatial (no
 * temporal data, so no ghosting on fast-changing audio-reactive content),
 * MIT-licensed, and is the standard solution for "render small, display big"
 * with deterministic output. EASU's edge-aware reconstruction beats
 * bilinear-then-sharpen on hard edges (logos, faces, lettering), and RCAS's
 * built-in halo clamps avoid the ringing that aggressive unsharp mask
 * produced at high sharpen values.
 *
 * Designed for the stage page where a small generated frame (e.g. 256×144)
 * is upscaled to a projector (e.g. 2048×1152). Same component is mounted in
 * the control-page preview and the stage projector page so both surfaces
 * show identical output.
 *
 * Pixelate mode bypasses both passes — sharpening pixelated content amplifies
 * block-edge steps. The pixelate path snaps UVs to a source-pixel grid and
 * renders directly to the canvas in a single pass.
 *
 * Set `sharpen` to 0 to bypass RCAS (acts as EASU-only upscale, still much
 * sharper than plain bilinear).
 *
 * Imperative API via ref:
 *   - drawBitmap(bitmap): upload + render one frame
 */

// Vertex shader for passes that sample a source ImageBitmap-uploaded texture
// (here: the EASU pass, also used by the single-pass pixelate path). Applies
// a Y-flip in UV so the top-left-origin ImageBitmap lines up with WebGL's
// bottom-left-origin texture sampling.
const VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 1.0 - (a_pos.y * 0.5 + 0.5));
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

// Vertex shader for passes that sample an FBO texture (here: RCAS, which
// reads the EASU intermediate). No Y-flip — the FBO was rendered to with
// the flipping vertex shader above, which already put the image right-side
// up in OpenGL convention (gl_FragCoord.y=0 at the bottom). Applying the
// flip a second time inverts the image, which is the bug this exists to
// avoid; we hit it the moment EASU+RCAS replaced the previous single-pass
// pipeline. Pixelate mode survives because it skips the FBO entirely.
const VERT_FBO_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, a_pos.y * 0.5 + 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Pass 1: EASU upscale (or pixelate bypass).
// Renders into the offscreen FBO when in EASU mode; renders straight to the
// canvas when in pixelate mode (no second pass needed).
// ---------------------------------------------------------------------------
const EASU_FRAG_SRC = `#version 300 es
precision highp float;

uniform sampler2D u_src;     // source ImageBitmap, LINEAR-filtered
uniform vec2 u_srcSize;      // source dimensions in pixels
uniform float u_pixelate;    // 0 = EASU; >=1 = pixelate block size in source pixels

in vec2 v_uv;
out vec4 o;

// --- EASU helpers ----------------------------------------------------------

// Accumulate one Lanczos-2-shaped weighted tap. 'off' is the offset (in
// source-pixel units) from the resolve position to this tap, 'dir' is the
// detected edge direction (unit vector), 'len' is the per-axis anisotropy,
// 'lob' shapes the negative lobe, 'clp' is the squared-distance clip.
void easuTap(
  inout vec3 aC, inout float aW,
  vec2 off, vec2 dir, vec2 len, float lob, float clp, vec3 c
) {
  // Rotate the offset into the gradient frame, then apply anisotropy.
  vec2 v = vec2(off.x * dir.x + off.y * dir.y,
                off.x * -dir.y + off.y * dir.x);
  v *= len;
  // Distance squared, clipped to the lobe window.
  float d2 = min(v.x * v.x + v.y * v.y, clp);
  // Lanczos-2 approximation: base * window, no sin/sqrt needed.
  //   (25/16 * (2/5 * d2 - 1)^2 - (25/16 - 1)) * (lob * d2 - 1)^2
  float wB = 2.0 / 5.0 * d2 - 1.0;
  float wA = lob * d2 - 1.0;
  wB *= wB;
  wA *= wA;
  wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
  float w = wB * wA;
  aC += c * w;
  aW += w;
}

// Accumulate gradient direction + length estimate from one bilinear corner
// of the inner 2x2 quadrant (f, g, j, k). Each call contributes the corner's
// bilinear weight to the running totals.
void easuSet(
  inout vec2 dir, inout float len,
  vec2 pp,
  bool biS, bool biT, bool biU, bool biV,
  float lA, float lB, float lC, float lD, float lE
) {
  float w = 0.0;
  if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
  if (biT) w =        pp.x  * (1.0 - pp.y);
  if (biU) w = (1.0 - pp.x) *        pp.y;
  if (biV) w =        pp.x  *        pp.y;

  // X-axis gradient ('+' pattern around lC: lB on the left, lD on the right).
  float dc = lD - lC;
  float cb = lC - lB;
  float lenX = max(abs(dc), abs(cb));
  lenX = lenX > 0.0 ? 1.0 / lenX : 0.0;
  float dirX = lD - lB;
  dir.x += dirX * w;
  lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
  len += lenX * lenX * w;

  // Y-axis gradient (lA above lC, lE below).
  float ec = lE - lC;
  float ca = lC - lA;
  float lenY = max(abs(ec), abs(ca));
  lenY = lenY > 0.0 ? 1.0 / lenY : 0.0;
  float dirY = lE - lA;
  dir.y += dirY * w;
  lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
  len += lenY * lenY * w;
}

void main() {
  // Pixelate bypass: snap UV to an NxN source-pixel grid, single texture
  // read, done. RCAS is skipped at the call site when pixelate is on.
  if (u_pixelate >= 1.0) {
    vec2 blockUv = u_pixelate / u_srcSize;
    vec2 uv = (floor(v_uv / blockUv) + 0.5) * blockUv;
    o = vec4(texture(u_src, uv).rgb, 1.0);
    return;
  }

  // Resolve position in source-pixel space, centred on the output pixel.
  // pp = (output uv) * srcSize - 0.5 — the -0.5 puts us at pixel centres.
  vec2 pp = v_uv * u_srcSize - vec2(0.5);
  vec2 fp = floor(pp);
  pp -= fp;

  // 12-tap kernel layout (centred on f at fp+(0,0)):
  //          b c
  //        e f g h
  //        i j k l
  //          n o
  vec2 inv = 1.0 / u_srcSize;
  vec3 b = texture(u_src, (fp + vec2( 0.5, -0.5)) * inv).rgb;
  vec3 c = texture(u_src, (fp + vec2( 1.5, -0.5)) * inv).rgb;
  vec3 e = texture(u_src, (fp + vec2(-0.5,  0.5)) * inv).rgb;
  vec3 f = texture(u_src, (fp + vec2( 0.5,  0.5)) * inv).rgb;
  vec3 g = texture(u_src, (fp + vec2( 1.5,  0.5)) * inv).rgb;
  vec3 h = texture(u_src, (fp + vec2( 2.5,  0.5)) * inv).rgb;
  vec3 i = texture(u_src, (fp + vec2(-0.5,  1.5)) * inv).rgb;
  vec3 j = texture(u_src, (fp + vec2( 0.5,  1.5)) * inv).rgb;
  vec3 k = texture(u_src, (fp + vec2( 1.5,  1.5)) * inv).rgb;
  vec3 l = texture(u_src, (fp + vec2( 2.5,  1.5)) * inv).rgb;
  vec3 n = texture(u_src, (fp + vec2( 0.5,  2.5)) * inv).rgb;
  vec3 oo = texture(u_src, (fp + vec2( 1.5,  2.5)) * inv).rgb;

  // Twice-luma approx (FSR uses ratios so the 2x scale doesn't matter).
  float bL = b.g + 0.5 * (b.b + b.r);
  float cL = c.g + 0.5 * (c.b + c.r);
  float eL = e.g + 0.5 * (e.b + e.r);
  float fL = f.g + 0.5 * (f.b + f.r);
  float gL = g.g + 0.5 * (g.b + g.r);
  float hL = h.g + 0.5 * (h.b + h.r);
  float iL = i.g + 0.5 * (i.b + i.r);
  float jL = j.g + 0.5 * (j.b + j.r);
  float kL = k.g + 0.5 * (k.b + k.r);
  float lL = l.g + 0.5 * (l.b + l.r);
  float nL = n.g + 0.5 * (n.b + n.r);
  float oL = oo.g + 0.5 * (oo.b + oo.r);

  // Direction & length accumulation across the four bilinear corners of the
  // inner 2x2 quadrant: top-left=f, top-right=g, bottom-left=j, bottom-right=k.
  vec2 dir = vec2(0.0);
  float len = 0.0;
  easuSet(dir, len, pp, true,  false, false, false, bL, eL, fL, gL, jL);
  easuSet(dir, len, pp, false, true,  false, false, cL, fL, gL, hL, kL);
  easuSet(dir, len, pp, false, false, true,  false, fL, iL, jL, kL, nL);
  easuSet(dir, len, pp, false, false, false, true,  gL, jL, kL, lL, oL);

  // Normalise direction. If gradient is essentially zero (flat region), fall
  // back to (1, 0) — the kernel becomes isotropic Lanczos-2.
  float dirSq = dir.x * dir.x + dir.y * dir.y;
  bool zero = dirSq < (1.0 / 32768.0);
  float invLen = inversesqrt(max(dirSq, 1.0 / 32768.0));
  if (zero) { dir = vec2(1.0, 0.0); invLen = 1.0; }
  dir *= invLen;

  // Length → anisotropy (squared and halved).
  len = len * 0.5;
  len *= len;

  // Stretch the kernel along the detected edge: 1.0 vert/horz, ~sqrt(2) on
  // diagonals. The minor-axis length lerps from 1.0 toward ~0.5 on edges.
  float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
  vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);

  // Lobe shape: shifts toward the negative lobe on edges (sharpening effect).
  float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
  float clp = 1.0 / lob;

  // 12-tap weighted accumulation.
  vec3 aC = vec3(0.0);
  float aW = 0.0;
  easuTap(aC, aW, vec2( 0.0, -1.0) - pp, dir, len2, lob, clp, b);
  easuTap(aC, aW, vec2( 1.0, -1.0) - pp, dir, len2, lob, clp, c);
  easuTap(aC, aW, vec2(-1.0,  1.0) - pp, dir, len2, lob, clp, i);
  easuTap(aC, aW, vec2( 0.0,  1.0) - pp, dir, len2, lob, clp, j);
  easuTap(aC, aW, vec2( 0.0,  0.0) - pp, dir, len2, lob, clp, f);
  easuTap(aC, aW, vec2(-1.0,  0.0) - pp, dir, len2, lob, clp, e);
  easuTap(aC, aW, vec2( 1.0,  1.0) - pp, dir, len2, lob, clp, k);
  easuTap(aC, aW, vec2( 2.0,  1.0) - pp, dir, len2, lob, clp, l);
  easuTap(aC, aW, vec2( 2.0,  0.0) - pp, dir, len2, lob, clp, h);
  easuTap(aC, aW, vec2( 1.0,  0.0) - pp, dir, len2, lob, clp, g);
  easuTap(aC, aW, vec2( 1.0,  2.0) - pp, dir, len2, lob, clp, oo);
  easuTap(aC, aW, vec2( 0.0,  2.0) - pp, dir, len2, lob, clp, n);

  // De-ring: clamp to the inner 2×2 quadrant range. Kills any overshoot from
  // the negative lobe — the source of halos on hard edges.
  vec3 mn4 = min(min(f, g), min(j, k));
  vec3 mx4 = max(max(f, g), max(j, k));
  vec3 result = clamp(aC / aW, mn4, mx4);
  o = vec4(result, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Pass 2: RCAS sharpen. EASU output → canvas.
// Skipped entirely when sharpen is 0 (caller can also force it off).
// ---------------------------------------------------------------------------
const RCAS_FRAG_SRC = `#version 300 es
precision highp float;

uniform sampler2D u_src;     // EASU output texture
uniform vec2 u_srcSize;      // EASU output dimensions = canvas dimensions
uniform float u_sharpen;     // 0 = bypass; ~0.3 mild; ~0.6 default; ~1.5 max

in vec2 v_uv;
out vec4 o;

// FSR_RCAS_LIMIT = 0.25 - (1.0 / 16.0). Hard ceiling on lobe magnitude — if
// the kernel ever asked for more sharpening than this it'd flip the resolve
// denominator's sign and produce garbage. The real limit is ~0.1875.
const float FSR_RCAS_LIMIT = 0.1875;

void main() {
  if (u_sharpen <= 0.001) {
    o = vec4(texture(u_src, v_uv).rgb, 1.0);
    return;
  }

  // 5-tap cross around the centre 'e':
  //         b
  //       d e f
  //         h
  vec2 t = 1.0 / u_srcSize;
  vec3 b = texture(u_src, v_uv + vec2( 0.0, -t.y)).rgb;
  vec3 d = texture(u_src, v_uv + vec2(-t.x,  0.0)).rgb;
  vec3 e = texture(u_src, v_uv).rgb;
  vec3 f = texture(u_src, v_uv + vec2( t.x,  0.0)).rgb;
  vec3 h = texture(u_src, v_uv + vec2( 0.0,  t.y)).rgb;

  // Twice-luma approx.
  float bL = b.g + 0.5 * (b.b + b.r);
  float dL = d.g + 0.5 * (d.b + d.r);
  float eL = e.g + 0.5 * (e.b + e.r);
  float fL = f.g + 0.5 * (f.b + f.r);
  float hL = h.g + 0.5 * (h.b + h.r);

  // Noise attenuation: high-frequency noise gets less sharpening. nz is the
  // ratio of "centre vs neighbour average" to "local luma range" — high on
  // single-pixel speckle, low on smooth gradients.
  float lumaMax = max(max(max(bL, dL), max(eL, fL)), hL);
  float lumaMin = min(min(min(bL, dL), min(eL, fL)), hL);
  float nz = 0.25 * (bL + dL + fL + hL) - eL;
  nz = clamp(abs(nz) / max(lumaMax - lumaMin, 1e-5), 0.0, 1.0);
  nz = -0.5 * nz + 1.0;

  // Lobe limit: bound how aggressively neighbours can pull the centre, so
  // we don't overshoot the ring's own min/max range (= no halos).
  vec3 mn4 = min(min(b, d), min(f, h));
  vec3 mx4 = max(max(b, d), max(f, h));
  vec3 hitMin = mn4 / (4.0 * mx4 + 1e-5);
  vec3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-5);
  vec3 lobeRGB = max(-hitMin, hitMax);
  float lobe = max(-FSR_RCAS_LIMIT,
                   min(max(max(lobeRGB.r, lobeRGB.g), lobeRGB.b), 0.0)) * u_sharpen;
  lobe *= nz;

  // Resolve: sharpened = (lobe * sum(neighbours) + centre) / (4*lobe + 1).
  // Lobe is negative, so this subtracts the neighbours and renormalises.
  float rcpL = 1.0 / (4.0 * lobe + 1.0);
  vec3 result = (lobe * (b + d + f + h) + e) * rcpL;
  o = vec4(clamp(result, 0.0, 1.0), 1.0);
}
`;

export interface StageRendererHandle {
  drawBitmap: (bitmap: ImageBitmap) => void;
  getCanvas: () => HTMLCanvasElement | null;
}

interface StageRendererProps {
  /** RCAS sharpen strength. 0 = bypass (EASU-only), ~0.3 mild, ~0.6 default,
   *  ~1.5 max useful. Higher values stay halo-free thanks to RCAS's built-in
   *  lobe limiter. */
  sharpen: number;
  /** Pixelate block size in source pixels. 0 = bypass, 1 = unchanged,
   *  N>1 = each NxN source-pixel block becomes a single solid colour.
   *  When on, EASU+RCAS are skipped (sharpening pixelated content amplifies
   *  block-edge steps). */
  pixelate?: number;
  /** Target minimum long side for the canvas's *internal* pixel grid.
   *  Default 1920 → "Full HD or above on the long side", aligned with
   *  standard display resolutions (1080p, 1440p, 4K — long sides 1920, 2560,
   *  3840). The canvas pixel size is computed as `source × ceil(target/sourceLong)`,
   *  which keeps the upscale at an INTEGER multiplier of the source —
   *  required by pixelate's UV-snap math. EASU is ratio-agnostic and
   *  tolerates any scale, but integer scale costs nothing and keeps
   *  pixelate consistent.
   *
   *  Pass 3840 for 4K-equivalent canvases (downloads ≥ ~4K resolution).
   *  CSS still controls the *displayed* size; this only changes the byte size
   *  of the rendered image. */
  targetLongSide?: number;
  className?: string;
  style?: React.CSSProperties;
}

export const StageRenderer = forwardRef<
  StageRendererHandle,
  StageRendererProps
>(function StageRenderer({ sharpen, pixelate, targetLongSide, className, style }, ref) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);

  // Two programs: pass 1 (EASU/pixelate) and pass 2 (RCAS).
  const easuProgRef = useRef<WebGLProgram | null>(null);
  const rcasProgRef = useRef<WebGLProgram | null>(null);

  // Source bitmap texture (input to pass 1).
  const srcTexRef = useRef<WebGLTexture | null>(null);

  // Offscreen FBO + texture for the EASU intermediate (input to pass 2).
  const fboRef = useRef<WebGLFramebuffer | null>(null);
  const fboTexRef = useRef<WebGLTexture | null>(null);
  const fboSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });

  // Uniforms, cached at link time.
  const easuUniformsRef = useRef<{
    srcSize: WebGLUniformLocation | null;
    pixelate: WebGLUniformLocation | null;
  } | null>(null);
  const rcasUniformsRef = useRef<{
    srcSize: WebGLUniformLocation | null;
    sharpen: WebGLUniformLocation | null;
  } | null>(null);

  const sharpenRef = useRef(sharpen);
  const pixelateRef = useRef(pixelate ?? 0);
  const targetLongSideRef = useRef(Math.max(1, Math.round(targetLongSide ?? 1920)));

  useEffect(() => {
    sharpenRef.current = sharpen;
  }, [sharpen]);

  useEffect(() => {
    pixelateRef.current = pixelate ?? 0;
  }, [pixelate]);

  useEffect(() => {
    targetLongSideRef.current = Math.max(1, Math.round(targetLongSide ?? 1920));
  }, [targetLongSide]);

  // One-shot WebGL setup. Compile both programs, allocate the source texture,
  // create the FBO + intermediate texture, set up the full-screen quad. We
  // render in `drawBitmap`, not on a RAF loop — there's nothing to animate
  // without a new frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      premultipliedAlpha: false,
      // Keep the backbuffer alive between renders. AI frames arrive at
      // 15–30 fps but the recording engine reads this canvas at 30 fps via
      // `drawImage` (not the WebGL-aware `captureStream`), so without this
      // flag the backbuffer is cleared after each compositor present and
      // every recording tick that lands between AI frames captures a blank
      // canvas — recordings show bursts of frames interleaved with black.
      // Cost: ~one extra framebuffer of GPU memory and possibly fewer swap
      // optimisations. drawBitmap always renders a full-screen quad, so
      // stale content is never visible.
      preserveDrawingBuffer: true,
    });
    if (!gl) return;
    glRef.current = gl;

    const compile = (type: number, src: string): WebGLShader | null => {
      const sh = gl.createShader(type);
      if (!sh) return null;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.warn("[stage shader]", gl.getShaderInfoLog(sh));
        gl.deleteShader(sh);
        return null;
      }
      return sh;
    };

    const link = (vs: WebGLShader, fs: WebGLShader): WebGLProgram | null => {
      const prog = gl.createProgram();
      if (!prog) return null;
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      // Pin a_pos to attribute location 0 so the same vertex buffer + attrib
      // setup works for both programs without re-binding.
      gl.bindAttribLocation(prog, 0, "a_pos");
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.warn("[stage program]", gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        return null;
      }
      return prog;
    };

    const vsFlip = compile(gl.VERTEX_SHADER, VERT_SRC);
    const vsFbo = compile(gl.VERTEX_SHADER, VERT_FBO_SRC);
    const easuFs = compile(gl.FRAGMENT_SHADER, EASU_FRAG_SRC);
    const rcasFs = compile(gl.FRAGMENT_SHADER, RCAS_FRAG_SRC);
    if (!vsFlip || !vsFbo || !easuFs || !rcasFs) return;

    // EASU samples the source bitmap → use the Y-flipping VS.
    // RCAS samples the FBO (already in WebGL/OpenGL convention) → use the
    // pass-through VS. Mixing them is what fixes the upside-down output.
    const easuProg = link(vsFlip, easuFs);
    const rcasProg = link(vsFbo, rcasFs);
    if (!easuProg || !rcasProg) return;

    easuProgRef.current = easuProg;
    rcasProgRef.current = rcasProg;

    // Two-triangle quad covering [-1, 1] in clip space.
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Source texture (pass-1 input). LINEAR filter is fine because EASU
    // samples at exact pixel centres — bilinear at a centre returns the
    // texel value with no interpolation.
    const srcTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    srcTexRef.current = srcTex;

    // Offscreen FBO + texture for the EASU intermediate.
    const fbo = gl.createFramebuffer();
    const fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, fboTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // 1×1 placeholder; resized to canvas size on first drawBitmap.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      fboTex,
      0
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fboRef.current = fbo;
    fboTexRef.current = fboTex;
    fboSizeRef.current = { w: 1, h: 1 };

    // Cache uniform locations.
    easuUniformsRef.current = {
      srcSize: gl.getUniformLocation(easuProg, "u_srcSize"),
      pixelate: gl.getUniformLocation(easuProg, "u_pixelate"),
    };
    rcasUniformsRef.current = {
      srcSize: gl.getUniformLocation(rcasProg, "u_srcSize"),
      sharpen: gl.getUniformLocation(rcasProg, "u_sharpen"),
    };

    // Pin both samplers to texture unit 0 (we swap which texture is bound
    // there per pass).
    gl.useProgram(easuProg);
    gl.uniform1i(gl.getUniformLocation(easuProg, "u_src"), 0);
    gl.useProgram(rcasProg);
    gl.uniform1i(gl.getUniformLocation(rcasProg, "u_src"), 0);

    return () => {
      gl.deleteProgram(easuProg);
      gl.deleteProgram(rcasProg);
      gl.deleteShader(vsFlip);
      gl.deleteShader(vsFbo);
      gl.deleteShader(easuFs);
      gl.deleteShader(rcasFs);
      if (srcTex) gl.deleteTexture(srcTex);
      if (fboTex) gl.deleteTexture(fboTex);
      if (fbo) gl.deleteFramebuffer(fbo);
      if (buf) gl.deleteBuffer(buf);
    };
  }, []);

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    drawBitmap: (bitmap: ImageBitmap) => {
      const gl = glRef.current;
      const easuProg = easuProgRef.current;
      const rcasProg = rcasProgRef.current;
      const srcTex = srcTexRef.current;
      const fbo = fboRef.current;
      const fboTex = fboTexRef.current;
      const easuU = easuUniformsRef.current;
      const rcasU = rcasUniformsRef.current;
      const canvas = canvasRef.current;
      if (!gl || !easuProg || !rcasProg || !srcTex || !fbo || !fboTex || !easuU || !rcasU || !canvas) {
        return;
      }

      // Canvas internal pixel size = source × N where N is the smallest
      // INTEGER multiplier that gets the long side ≥ targetLongSide (default
      // 1920 → Full HD-equivalent). CSS still controls the *displayed* size
      // (small in the panel, full-screen on stage). The shaders' u_srcSize
      // uniform references SOURCE pixel size (pass 1) and EASU-output size
      // (pass 2), so the visual is identical regardless of N.
      //
      // Examples (default targetLongSide=1920):
      //   512×288 → ×4 → 2048×1152
      //   384×224 → ×5 → 1920×1120 (long side exactly 1920)
      //   256×144 → ×8 → 2048×1152
      //   192×112 → ×10 → 1920×1120
      //   288×512 → ×4 → 1152×2048 (vertical 9:16)
      //
      // Pass targetLongSide={3840} for 4K-equivalent canvases.
      const sourceLongSide = Math.max(bitmap.width, bitmap.height);
      const scale = Math.max(1, Math.ceil(targetLongSideRef.current / sourceLongSide));
      const targetW = bitmap.width * scale;
      const targetH = bitmap.height * scale;
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      // Upload bitmap to source texture.
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        bitmap
      );

      const pix = pixelateRef.current;

      if (pix >= 1) {
        // Pixelate path: single pass, source → canvas, no FBO involved.
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(easuProg);
        if (easuU.srcSize) gl.uniform2f(easuU.srcSize, bitmap.width, bitmap.height);
        if (easuU.pixelate) gl.uniform1f(easuU.pixelate, pix);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        return;
      }

      // EASU path: pass 1 source → FBO, pass 2 FBO → canvas.

      // Resize the FBO texture if the canvas size changed.
      const fboSize = fboSizeRef.current;
      if (fboSize.w !== canvas.width || fboSize.h !== canvas.height) {
        gl.bindTexture(gl.TEXTURE_2D, fboTex);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          canvas.width,
          canvas.height,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          null
        );
        fboSize.w = canvas.width;
        fboSize.h = canvas.height;
      }

      // Pass 1: EASU. Source texture → FBO at canvas resolution.
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.useProgram(easuProg);
      if (easuU.srcSize) gl.uniform2f(easuU.srcSize, bitmap.width, bitmap.height);
      if (easuU.pixelate) gl.uniform1f(easuU.pixelate, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // Pass 2: RCAS. FBO texture → canvas (default framebuffer).
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fboTex);
      gl.useProgram(rcasProg);
      if (rcasU.srcSize) gl.uniform2f(rcasU.srcSize, canvas.width, canvas.height);
      if (rcasU.sharpen) gl.uniform1f(rcasU.sharpen, sharpenRef.current);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
  }));

  // When pixelate is on, the source image is intentionally low-frequency
  // (each NxN block is one solid colour). Browser bilinear upscaling — the
  // default — would blur block edges, defeating the look. `image-rendering:
  // pixelated` switches the upscale to nearest-neighbour, giving crisp blocks
  // at any display size. With pixelate off we keep the default LINEAR upscale
  // because the canvas is already at a generous internal resolution and the
  // browser's bilinear blit to the displayed size is fine.
  const imageRendering = (pixelate ?? 0) >= 1 ? "pixelated" : undefined;
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={imageRendering ? { ...style, imageRendering } : style}
    />
  );
});
