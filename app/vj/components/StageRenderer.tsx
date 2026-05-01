"use client";

import { useEffect, useRef, useImperativeHandle, forwardRef } from "react";

/**
 * WebGL2 canvas that draws an incoming ImageBitmap with a configurable
 * unsharp-mask sharpen pass. Designed for the stage page where a small
 * generated frame (e.g. 256×144) is upscaled to a projector (e.g. 1920×1080)
 * — bilinear alone looks soft, the sharpen pass restores edge contrast.
 *
 * The pass samples the source texture at its own texel grid (1/sourceSize),
 * so the kernel works in source-pixel space. At display resolution the
 * sampler does the upscale (LINEAR), then sharpen recovers the high-freq
 * detail that bilinear smeared.
 *
 * Set `sharpen` to 0 to bypass entirely (acts as a plain bilinear blit).
 *
 * Imperative API via ref:
 *   - drawBitmap(bitmap): upload + render one frame
 */

const VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  // Full-screen triangle pair from a unit quad in clip space.
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 1.0 - (a_pos.y * 0.5 + 0.5));
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

uniform sampler2D u_tex;
uniform vec2 u_texel;       // 1.0 / source-texture size, in UV units
uniform float u_sharpen;    // 0 = bypass, ~0.5 mild, ~1.5 aggressive
uniform float u_pixelate;   // 0 = bypass, N>=1 = block size in source pixels

in vec2 v_uv;
out vec4 o;

void main() {
  // Pixelate pre-pass: snap UV to a grid in source-pixel space so each NxN
  // block samples a single source pixel, then everything downstream sees that
  // chunky-pixel image. Sharpen is forced off when pixelate is on — sharpening
  // pixelated content amplifies the block-edge step, which looks bad.
  vec2 uv = v_uv;
  if (u_pixelate >= 1.0) {
    vec2 blockUv = u_texel * u_pixelate;
    uv = (floor(v_uv / blockUv) + 0.5) * blockUv;
    o = vec4(texture(u_tex, uv).rgb, 1.0);
    return;
  }

  vec3 c = texture(u_tex, uv).rgb;
  if (u_sharpen <= 0.001) {
    o = vec4(c, 1.0);
    return;
  }
  // 4-tap cross unsharp mask: sample N/S/E/W in source-pixel units, average
  // them, subtract from the centre and add the difference back scaled by the
  // sharpen amount. Cheap (5 texture reads), no ringing artifacts.
  vec3 n = texture(u_tex, uv + vec2(0.0, -u_texel.y)).rgb;
  vec3 s = texture(u_tex, uv + vec2(0.0,  u_texel.y)).rgb;
  vec3 e = texture(u_tex, uv + vec2(u_texel.x, 0.0)).rgb;
  vec3 w = texture(u_tex, uv + vec2(-u_texel.x, 0.0)).rgb;
  vec3 blur = (n + s + e + w) * 0.25;
  vec3 sharp = c + (c - blur) * u_sharpen;
  o = vec4(clamp(sharp, 0.0, 1.0), 1.0);
}
`;

export interface StageRendererHandle {
  drawBitmap: (bitmap: ImageBitmap) => void;
  getCanvas: () => HTMLCanvasElement | null;
}

interface StageRendererProps {
  sharpen: number;
  /** Pixelate block size in source pixels. 0 = bypass, 1 = unchanged,
   *  N>1 = each NxN source-pixel block becomes a single solid colour. */
  pixelate?: number;
  /** Target minimum long side for the canvas's *internal* pixel grid.
   *  Default 1920 → "Full HD or above on the long side", aligned with
   *  standard display resolutions (1080p, 1440p, 4K — long sides 1920, 2560,
   *  3840). The canvas pixel size is computed as `source × ceil(target/sourceLong)`,
   *  which keeps the upscale at an INTEGER multiplier of the source — important
   *  because pixelate's UV-snap math and sharpen's source-pixel kernel both
   *  expect integer-aligned grids; non-integer scaling would introduce LINEAR
   *  blur between source pixels.
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
  const programRef = useRef<WebGLProgram | null>(null);
  const texRef = useRef<WebGLTexture | null>(null);
  const uTexelRef = useRef<WebGLUniformLocation | null>(null);
  const uSharpenRef = useRef<WebGLUniformLocation | null>(null);
  const uPixelateRef = useRef<WebGLUniformLocation | null>(null);
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

  // One-shot WebGL setup. Compiles the program, allocates the texture, sets
  // up the full-screen quad. We render in `drawBitmap`, not on a RAF loop —
  // there's nothing to animate without a new frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
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

    const vs = compile(gl.VERTEX_SHADER, VERT_SRC);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG_SRC);
    if (!vs || !fs) return;
    const prog = gl.createProgram();
    if (!prog) return;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.warn("[stage program]", gl.getProgramInfoLog(prog));
      return;
    }
    programRef.current = prog;

    // Two-triangle quad covering [-1, 1] in clip space.
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    const aPos = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    texRef.current = tex;

    uTexelRef.current = gl.getUniformLocation(prog, "u_texel");
    uSharpenRef.current = gl.getUniformLocation(prog, "u_sharpen");
    uPixelateRef.current = gl.getUniformLocation(prog, "u_pixelate");

    return () => {
      gl.deleteProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      if (tex) gl.deleteTexture(tex);
      if (buf) gl.deleteBuffer(buf);
    };
  }, []);

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    drawBitmap: (bitmap: ImageBitmap) => {
      const gl = glRef.current;
      const prog = programRef.current;
      const tex = texRef.current;
      const canvas = canvasRef.current;
      if (!gl || !prog || !tex || !canvas) return;

      // Canvas internal pixel size = source × N where N is the smallest
      // INTEGER multiplier that gets the long side ≥ targetLongSide (default
      // 1920 → Full HD-equivalent). CSS still controls the *displayed* size
      // (small in the panel, full-screen on stage). The shader's u_texel
      // uniform references SOURCE pixel size, not canvas size, so the visual
      // is identical regardless of N — pixelate blocks stay the same NxN
      // source pixels, sharpen samples N/S/E/W in source space.
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
        gl.viewport(0, 0, canvas.width, canvas.height);
      }

      gl.useProgram(prog);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        bitmap
      );
      if (uTexelRef.current) {
        gl.uniform2f(uTexelRef.current, 1 / bitmap.width, 1 / bitmap.height);
      }
      if (uSharpenRef.current) {
        gl.uniform1f(uSharpenRef.current, sharpenRef.current);
      }
      if (uPixelateRef.current) {
        gl.uniform1f(uPixelateRef.current, pixelateRef.current);
      }
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
  }));

  // When pixelate is on, the source image is intentionally low-frequency
  // (each NxN block is one solid colour). Browser bilinear upscaling — the
  // default — would blur block edges, defeating the look. `image-rendering:
  // pixelated` switches the upscale to nearest-neighbour, giving crisp blocks
  // at any display size. Lossless full-HD: WebGL renders at source res,
  // browser does the cheap nearest upscale. With pixelate off we keep the
  // default LINEAR upscale so the unsharp-mask sharpen still has soft edges
  // to work with.
  const imageRendering = (pixelate ?? 0) >= 1 ? "pixelated" : undefined;
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={imageRendering ? { ...style, imageRendering } : style}
    />
  );
});
