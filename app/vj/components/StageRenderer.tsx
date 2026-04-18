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
uniform vec2 u_texel;     // 1.0 / source-texture size, in UV units
uniform float u_sharpen;  // 0 = bypass, ~0.5 mild, ~1.5 aggressive

in vec2 v_uv;
out vec4 o;

void main() {
  vec3 c = texture(u_tex, v_uv).rgb;
  if (u_sharpen <= 0.001) {
    o = vec4(c, 1.0);
    return;
  }
  // 4-tap cross unsharp mask: sample N/S/E/W in source-pixel units, average
  // them, subtract from the centre and add the difference back scaled by the
  // sharpen amount. Cheap (5 texture reads), no ringing artifacts.
  vec3 n = texture(u_tex, v_uv + vec2(0.0, -u_texel.y)).rgb;
  vec3 s = texture(u_tex, v_uv + vec2(0.0,  u_texel.y)).rgb;
  vec3 e = texture(u_tex, v_uv + vec2(u_texel.x, 0.0)).rgb;
  vec3 w = texture(u_tex, v_uv + vec2(-u_texel.x, 0.0)).rgb;
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
  className?: string;
  style?: React.CSSProperties;
}

export const StageRenderer = forwardRef<
  StageRendererHandle,
  StageRendererProps
>(function StageRenderer({ sharpen, className, style }, ref) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const texRef = useRef<WebGLTexture | null>(null);
  const uTexelRef = useRef<WebGLUniformLocation | null>(null);
  const uSharpenRef = useRef<WebGLUniformLocation | null>(null);
  const sharpenRef = useRef(sharpen);

  useEffect(() => {
    sharpenRef.current = sharpen;
  }, [sharpen]);

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

      // Match canvas pixel size to source. CSS scales the canvas up to fill
      // the viewport, so the sharpen kernel always operates on source pixels.
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
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
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    },
  }));

  return <canvas ref={canvasRef} className={className} style={style} />;
});
