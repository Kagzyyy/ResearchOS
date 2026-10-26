"use client"

import * as React from "react"

/**
 * Halftone Nebula — a pixel-art night sky printed in dots.
 *
 * One full-screen fragment pass. The nebula is domain-warped fbm sampled once
 * per pixel cell, quantised with an ordered Bayer dither, then printed as a
 * halftone: every cell is a flat square of the dim ramp with a round dot of the
 * bright ramp on top, the dot's radius driven by density. Where the gas is thin
 * the dots shrink to a faint mesh; where it is thick they grow past the cell
 * and merge into solid pixel blotches — the same trick a risograph uses.
 *
 * On top: three parallax layers of hashed pixel stars, a halftone planet, and
 * cross-shaped sparkle stars whose spikes are drawn at one CSS pixel so they
 * stay needle-thin against the chunky grid. The pointer is a lamp that parts
 * the gas and swells the dots under it; a click drops a new sparkle and sends a
 * shock ring through the cloud.
 *
 * Self-contained: raw WebGL2, React is the only import. No textures, no image
 * assets, no CSS file. The canvas sizes itself from its own box, never the
 * window, and releases every GL object on unmount.
 *
 * Local changes from the original: the GLSL declaration join uses a real "\n"
 * (the pasted source had a raw line break inside the string literal), the
 * reduced-motion preference is read with useSyncExternalStore, the params ref is
 * updated in an effect rather than during render, and elements marked
 * `data-nebula-ignore` swallow pointer presses so panels laid over the sky do not
 * drop sparkles when clicked.
 */

export type NebulaParams = {
  // the grid
  /** CSS px per pixel cell. Everything snaps to this. */
  pixel: number
  /** Dot radius, in cells, where the gas is empty / densest. >0.5 merges dots. */
  dotMin: number
  dotMax: number
  /** Quantisation steps before dithering. Fewer reads more like pixel art. */
  levels: number
  // the gas
  scale: number
  warp: number
  drift: number
  density: number
  threshold: number
  softness: number
  /** The luminous river of gas that sweeps across the frame. */
  band: number
  bandAngle: number
  bandOffset: number
  bandWidth: number
  haze: number
  // stars
  stars: number
  twinkle: number
  starDrift: number
  sparkles: number
  seed: number
  spikeWidth: number
  // planet
  planet: boolean
  planetX: number
  planetY: number
  planetRadius: number
  // the pointer
  parallax: number
  lens: number
  lensRadius: number
  lensPush: number
  rippleSpeed: number
  // post
  speed: number
  vignette: number
  grain: number
  // palette, dark to bright
  voidColor: string
  hazeColor: string
  duskColor: string
  wineColor: string
  crimsonColor: string
  hotColor: string
  starColor: string
}

export const NEBULA_DEFAULTS: NebulaParams = {
  pixel: 6,
  dotMin: 0.12,
  dotMax: 0.56,
  levels: 7,

  scale: 1.7,
  warp: 1.5,
  drift: 0.035,
  density: 0.5,
  threshold: 0.54,
  softness: 0.5,
  band: 0.5,
  bandAngle: 1.05,
  bandOffset: 0.42,
  bandWidth: 0.34,
  haze: 0.8,

  stars: 1,
  twinkle: 1.4,
  starDrift: 1.2,
  sparkles: 9,
  seed: 11,
  spikeWidth: 1,

  planet: true,
  planetX: 0.26,
  planetY: 0.36,
  planetRadius: 0.075,

  parallax: 1,
  lens: 0.55,
  lensRadius: 170,
  lensPush: 0.3,
  rippleSpeed: 420,

  speed: 1,
  vignette: 0.5,
  grain: 0.035,

  voidColor: "#050309",
  hazeColor: "#161a38",
  duskColor: "#3b1646",
  wineColor: "#5c0d31",
  crimsonColor: "#c01245",
  hotColor: "#ff1f5a",
  starColor: "#f6e2e8",
}

/** Overlays on the defaults. Named for the sky, not the numbers. */
export const NEBULA_PRESETS: Record<string, Partial<NebulaParams>> = {
  crimson: {},
  ultraviolet: {
    hazeColor: "#10183f", duskColor: "#2a1a5e", wineColor: "#3d1478",
    crimsonColor: "#7b2cf0", hotColor: "#c77dff", starColor: "#eef0ff",
    bandAngle: 2.2, bandOffset: 0.3, planetX: 0.74, planetY: 0.3, seed: 4,
  },
  abyssal: {
    voidColor: "#02060a", hazeColor: "#0a1f2e", duskColor: "#0d2f3f", wineColor: "#0a4453",
    crimsonColor: "#0f8f9f", hotColor: "#3ff2e0", starColor: "#e4fffb",
    band: 0.5, bandAngle: -0.4, bandOffset: -0.2, planetX: 0.7, planetY: 0.66, seed: 23,
  },
  solar: {
    voidColor: "#070403", hazeColor: "#231208", duskColor: "#3d1a07", wineColor: "#6e2406",
    crimsonColor: "#d8520c", hotColor: "#ffb020", starColor: "#fff3d6",
    warp: 1.9, band: 0.72, bandAngle: 0.4, planetRadius: 0.1, seed: 5,
  },
  phosphor: {
    voidColor: "#020502", hazeColor: "#08170c", duskColor: "#0c2412", wineColor: "#0e3d18",
    crimsonColor: "#1f9d3a", hotColor: "#6dff7a", starColor: "#e6ffe8",
    pixel: 5, levels: 5, planet: false, sparkles: 6, seed: 31,
  },
}

const MAX_SPARKS = 16
const MAX_RIPPLES = 4

type Kind = "f" | "i" | "b" | "c"
type Slot = [keyof NebulaParams, Kind]

/** Which parameters reach the shader, and as what. `c` is a hex colour. */
const UNIFORMS: Slot[] = [
  ["pixel", "f"], ["dotMin", "f"], ["dotMax", "f"], ["levels", "f"],
  ["scale", "f"], ["warp", "f"], ["drift", "f"], ["density", "f"],
  ["threshold", "f"], ["softness", "f"],
  ["band", "f"], ["bandAngle", "f"], ["bandOffset", "f"], ["bandWidth", "f"], ["haze", "f"],
  ["stars", "f"], ["twinkle", "f"], ["starDrift", "f"], ["spikeWidth", "f"],
  ["parallax", "f"], ["lens", "f"], ["lensRadius", "f"], ["lensPush", "f"], ["rippleSpeed", "f"],
  ["vignette", "f"], ["grain", "f"],
  ["voidColor", "c"], ["hazeColor", "c"], ["duskColor", "c"], ["wineColor", "c"],
  ["crimsonColor", "c"], ["hotColor", "c"], ["starColor", "c"],
]

const uName = (k: string) => "u" + k[0].toUpperCase() + k.slice(1)
const glslType = (kind: Kind) => (kind === "c" ? "vec3" : kind === "i" || kind === "b" ? "int" : "float")
const declare = (slots: Slot[]) =>
  slots.map(([k, kind]) => "uniform " + glslType(kind) + " " + uName(k) + ";").join("\n")

const VERT = `#version 300 es
void main(){
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uDpr;
uniform float uTime;
uniform vec2 uPointer;
uniform float uPointerOn;
uniform vec2 uLook;
uniform int uSparkCount;
uniform vec4 uSpark[${MAX_SPARKS}];
uniform vec4 uSparkB[${MAX_SPARKS}];
uniform vec4 uRipple[${MAX_RIPPLES}];
uniform vec4 uPlanet;
${declare(UNIFORMS)}
out vec4 frag;

float hash12(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash12(i), b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0)), d = hash12(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p){
  float s = 0.0, a = 0.5;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = r * p * 2.03 + 17.1; a *= 0.5; }
  return s / 0.96875;
}
float bayer4(vec2 c){
  vec2 m = mod(c, 4.0);
  int i = int(m.x) + int(m.y) * 4;
  int b[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
  return (float(b[i]) + 0.5) / 16.0;
}
vec3 ramp(float d, vec3 haze){
  vec3 c = mix(uVoidColor, haze, smoothstep(0.0, 0.22, d));
  c = mix(c, uWineColor, smoothstep(0.18, 0.44, d));
  c = mix(c, uCrimsonColor, smoothstep(0.42, 0.72, d));
  return mix(c, uHotColor, smoothstep(0.7, 0.96, d));
}
vec2 shift(float depth){
  return floor(uLook * uParallax * depth * 28.0);
}

// Density of the gas at a CSS-pixel position: x = density, y = haze, z = tone.
vec3 field(vec2 pos, vec2 resCss){
  float minSide = min(resCss.x, resCss.y);
  vec2 uv = (pos - 0.5 * resCss) / minSide;
  vec2 p = uv * uScale + uLook * uParallax * 0.05;

  vec2 toP = pos - uPointer;
  float lamp = exp(-dot(toP, toP) / (uLensRadius * uLensRadius)) * uPointerOn;
  p += toP / minSide * lamp * uLensPush * uScale;

  float ring = 0.0;
  for (int i = 0; i < ${MAX_RIPPLES}; i++) {
    vec4 r = uRipple[i];
    float age = uTime - r.z;
    if (r.w <= 0.0 || age < 0.0 || age > 2.4) continue;
    vec2 dv = pos - r.xy;
    float dist = length(dv);
    float w = 14.0 + age * 26.0;
    float k = exp(-pow((dist - age * uRippleSpeed) / w, 2.0)) * (1.0 - age / 2.4) * r.w;
    ring += k;
    p += dv / max(dist, 1.0) * k * 0.12;
  }

  float t = uTime * uDrift;
  vec2 q = vec2(fbm(p + vec2(0.0, t * 0.7)), fbm(p + vec2(5.2, 1.3) - t * 0.5));
  float n = fbm(p + uWarp * q + vec2(t * 0.4, -t * 0.25));

  vec2 dir = vec2(cos(uBandAngle), sin(uBandAngle));
  float along = dot(uv, dir);
  float across = dot(uv, vec2(-dir.y, dir.x)) - uBandOffset - 0.22 * sin(along * 2.3 + t * 3.0) - (q.x - 0.5) * 0.35;
  float river = exp(-across * across / (uBandWidth * uBandWidth));

  float raw = n + river * uBand * 0.5 + (uDensity - 0.5) * 0.5;
  float d = smoothstep(uThreshold, uThreshold + uSoftness, raw);
  d = clamp(d + lamp * uLens * 0.45 + ring * 0.55, 0.0, 1.0);

  for (int i = 0; i < ${MAX_SPARKS}; i++) {
    if (i >= uSparkCount) break;
    vec4 s = uSpark[i];
    vec4 b = uSparkB[i];
    float grow = smoothstep(0.0, 0.6, uTime - s.w);
    float dist = length(pos - s.xy - shift(1.2));
    d += exp(-dist / max(b.w * 2.2 * (1.0 + b.z * 0.6), 1.0)) * 0.55 * grow;
  }

  float hz = smoothstep(0.25, 0.75, n + river * 0.25) * uHaze;
  float tone = fbm(p * 0.45 + vec2(11.0, -3.0) + t * 0.2);
  return vec3(clamp(d, 0.0, 1.0), hz, tone);
}

void main(){
  vec2 resCss = uRes / uDpr;
  vec2 css = gl_FragCoord.xy / uDpr;
  float px = max(uPixel, 1.0);
  vec2 cell = floor(css / px);
  vec2 cellC = (cell + 0.5) * px;
  vec2 f = fract(css / px) - 0.5;
  float L = max(uLevels, 2.0);
  float dith = bayer4(cell);

  // ---- the gas, printed as a halftone ---------------------------------------
  vec3 g = field(cellC, resCss);
  vec3 hazeCol = mix(uHazeColor, uDuskColor, smoothstep(0.38, 0.62, g.z));
  float dq = clamp(floor(g.x * L + dith) / L, 0.0, 1.0);
  float hq = floor(g.y * 3.0 + dith) / 3.0;
  vec3 bg = mix(uVoidColor, hazeCol, hq * 0.7);
  bg = mix(bg, ramp(dq * 0.7, hazeCol) * 0.42, smoothstep(0.0, 0.3, dq));
  float r = mix(uDotMin, uDotMax, sqrt(dq));
  float dotMask = step(length(f), r);
  vec3 dotCol = dq < 0.01 ? mix(uVoidColor, hazeCol, 0.35 + hq * 0.5) : ramp(min(dq + 0.1, 1.0), hazeCol);
  vec3 col = mix(bg, dotCol, dotMask);

  // ---- pixel stars, three parallax depths ------------------------------------
  vec3 starAcc = vec3(0.0);
  float starA = 0.0;
  for (int l = 0; l < 3; l++) {
    float fl = float(l);
    float depth = 0.3 + fl * 0.4;
    float gpx = px * (l == 2 ? 2.0 : 1.0);
    vec2 sp = css + shift(depth) + vec2(0.0, floor(uTime * uStarDrift * depth));
    vec2 id = floor(sp / gpx);
    vec2 fr = fract(sp / gpx) - 0.5;
    float prob = uStars * (l == 0 ? 0.009 : l == 1 ? 0.014 : 0.01);
    float h = hash12(id + fl * 71.3);
    if (h > 1.0 - prob) {
      float h2 = hash12(id * 1.7 + 3.1 + fl);
      float tw = 0.45 + 0.55 * (0.5 + 0.5 * sin(uTime * uTwinkle * (0.6 + h2 * 2.5) + h2 * 40.0));
      float rad = l == 2 ? 0.42 : 0.26 + h2 * 0.22;
      float m = step(length(fr), rad);
      vec3 c = (l == 0 || h2 > 0.82) ? uStarColor : uHotColor;
      starAcc = max(starAcc, c * m * tw);
      starA = max(starA, m * tw);
    }
  }
  col = mix(col, starAcc / max(starA, 1e-3), starA * (1.0 - g.x * 0.55));

  // ---- the planet, same halftone, lit from the upper left --------------------
  if (uPlanet.w > 0.5) {
    vec2 pc = uPlanet.xy + shift(0.8);
    float R = uPlanet.z;
    vec2 dc = cellC - pc;
    if (length(dc) < R) {
      vec2 n2 = dc / R;
      float z = sqrt(max(1.0 - dot(n2, n2), 0.0));
      float lit = clamp(dot(vec3(n2, z), normalize(vec3(-0.45, 0.55, 0.7))), 0.0, 1.0);
      float surf = fbm(vec2(n2.x * 1.4 + uTime * 0.015, n2.y * 4.2) * 1.6 + 9.0);
      float pd = clamp(lit * 1.05 - smoothstep(0.55, 0.75, surf) * 0.45 * (1.0 - lit * 0.5) + 0.05, 0.0, 1.0);
      float pq = floor(pd * L + dith) / L;
      vec3 pbg = mix(mix(uVoidColor, uWineColor, 0.35), uWineColor, pq);
      float pr = mix(0.2, 0.62, sqrt(pq));
      vec3 pdot = ramp(min(pq * 0.85 + 0.25, 1.0), hazeCol);
      col = mix(pbg, pdot, step(length(f), pr));
    }
  }

  // ---- sparkle stars: chunky core, needle spikes -----------------------------
  vec2 sh = shift(1.2);
  for (int i = 0; i < ${MAX_SPARKS}; i++) {
    if (i >= uSparkCount) break;
    vec4 s = uSpark[i];
    vec4 b = uSparkB[i];
    float age = uTime - s.w;
    if (age < 0.0) continue;
    float grow = smoothstep(0.0, 0.55, age) * (1.0 + 0.3 * exp(-age * 3.0) * sin(age * 13.0));
    float tw = 0.84 + 0.16 * sin(uTime * uTwinkle * 1.7 + b.y);
    float flare = 1.0 + b.z * 0.6;
    float reach = s.z * grow * tw * flare;
    float core = b.w * grow * flare;
    vec3 tint = mix(uHotColor, uStarColor, b.x);
    vec2 c = s.xy + sh;
    vec2 d = css - c;
    float th = uSpikeWidth * 0.5 + 0.25;
    float hx = step(abs(d.y), th) * pow(max(1.0 - abs(d.x) / max(reach, 1.0), 0.0), 1.1);
    float vy = step(abs(d.x), th) * pow(max(1.0 - abs(d.y) / max(reach, 1.0), 0.0), 1.1);
    // A short second pair of spikes, turned 45deg, only on the big ones.
    vec2 rd = vec2(d.x + d.y, d.x - d.y) * 0.7071;
    float diag = step(0.5, core / px - 1.5) * 0.45 * max(
      step(abs(rd.y), th) * pow(max(1.0 - abs(rd.x) / max(reach * 0.22, 1.0), 0.0), 2.0),
      step(abs(rd.x), th) * pow(max(1.0 - abs(rd.y) / max(reach * 0.22, 1.0), 0.0), 2.0));
    col = mix(col, tint, clamp(max(max(hx, vy), diag), 0.0, 1.0));
    // The core snaps to the cell grid, so it reads as a pixel-art disc.
    float hp = px * 0.5;
    vec2 dcell = (floor(css / hp) + 0.5) * hp - (floor(c / hp) + 0.5) * hp;
    float disc = step(length(dcell), core);
    float glow = exp(-length(d) / max(core * 1.4, 1.0)) * (1.0 - disc);
    col += tint * glow * 0.35;
    col = mix(col, tint, disc);
    float cross = max(step(abs(d.y), th) * step(abs(d.x), core * 0.85),
                      step(abs(d.x), th) * step(abs(d.y), core * 0.85));
    col = mix(col, uStarColor, cross * disc * 0.9);
  }

  // ---- post -----------------------------------------------------------------
  vec2 vu = gl_FragCoord.xy / uRes - 0.5;
  col *= 1.0 - uVignette * smoothstep(0.35, 0.95, length(vu * vec2(uRes.x / uRes.y, 1.0) * 1.1));
  col += (hash12(floor(css) + fract(uTime) * 91.0) - 0.5) * uGrain;
  frag = vec4(max(col, vec3(0.0)), 1.0);
}`

/**
 * One sparkle star. Position is 0..1 of the box (y up); `reach` and `core` are
 * fractions of the box's shorter side, so the sky scales with its container.
 */
export type Sparkle = {
  x: number
  y: number
  reach: number
  core: number
  /** 0 = the accent colour, 1 = the pale star colour. */
  tint: number
  phase: number
  /** Seconds on the component clock when it was born; it grows in from there. */
  born: number
  user: boolean
}

// #region sky
/** A tiny seeded PRNG, so the same `seed` always hangs the same sky. */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}


/**
 * Hang `count` sparkles. The first is the hero — long spikes, fat core, upper
 * half — and the second is the pale one; the rest are small and scattered.
 * Placement rejects anything too close to a star already hung, so the spikes
 * never tangle into a single knot.
 */
export function layoutSparkles(seed: number, count: number): Sparkle[] {
  const rand = mulberry32(seed)
  const out: Sparkle[] = []
  const n = Math.max(0, Math.min(Math.floor(count), MAX_SPARKS))
  for (let i = 0; i < n; i++) {
    const hero = i === 0
    const pale = i === 1
    let x = 0.5
    let y = 0.5
    for (let tries = 0; tries < 40; tries++) {
      x = 0.08 + 0.84 * rand()
      y = hero ? 0.58 + 0.28 * rand() : 0.08 + 0.84 * rand()
      const clear = out.every((s) => Math.hypot(s.x - x, s.y - y) > (hero ? 0.2 : 0.13))
      if (clear) break
    }
    out.push({
      x,
      y,
      reach: hero ? 0.2 + 0.05 * rand() : pale ? 0.08 + 0.03 * rand() : 0.025 + 0.055 * rand(),
      core: hero ? 0.024 : pale ? 0.011 : 0.004 + 0.006 * rand(),
      tint: pale ? 1 : 0,
      phase: rand() * 6.283,
      born: -10,
      user: false,
    })
  }
  return out
}

/**
 * Add a sparkle someone clicked into being. The seeded sky is never evicted;
 * once the list is full the oldest *user* sparkle makes room.
 */
export function pushSparkle(list: Sparkle[], next: Sparkle, max = MAX_SPARKS): Sparkle[] {
  if (list.length < max) return [...list, next]
  const oldest = list.findIndex((s) => s.user)
  if (oldest === -1) return list
  return [...list.slice(0, oldest), ...list.slice(oldest + 1), next]
}

/** "#f0a" / "#ff00aa" → [r, g, b] in 0..1. Anything unparsable is black. */
export function hexToRgb(hex: string): [number, number, number] {
  let h = hex.trim().replace(/^#/, "")
  if (h.length === 3) h = h.split("").map((c) => c + c).join("")
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return [0, 0, 0]
  const v = parseInt(h, 16)
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]
}

/** The idle lamp: wanders the sky on incommensurate sines, never closing a loop. */
export function driftPos(t: number) {
  const x = 0.5 + 0.32 * Math.sin(t * 0.21) + 0.1 * Math.sin(t * 0.077 + 1.3)
  const y = 0.5 + 0.26 * Math.cos(t * 0.17) + 0.12 * Math.cos(t * 0.053 + 4.2)
  return [Math.min(Math.max(x, 0.05), 0.95), Math.min(Math.max(y, 0.05), 0.95)]
}
// #endregion

const REDUCED_QUERY = "(prefers-reduced-motion: reduce)"
const subscribeReduced = (onChange: () => void) => {
  const mq = window.matchMedia(REDUCED_QUERY)
  mq.addEventListener("change", onChange)
  return () => mq.removeEventListener("change", onChange)
}
const readReduced = () => window.matchMedia(REDUCED_QUERY).matches
const serverReduced = () => false

export type HalftoneNebulaProps = {
  /**
   * Explicit height. The canvas fills this box, so it must be a definite
   * length — "100%" only works if every ancestor has one too.
   */
  height?: string
  /** A named palette + layout, layered over the defaults. */
  preset?: keyof typeof NEBULA_PRESETS
  /** Overrides layered over the preset. */
  params?: Partial<NebulaParams>
  /** Pointer lights the gas and parallaxes the sky; click drops a sparkle. */
  interactive?: boolean
  /** "scroll" keeps page scrolling on touch; "draw" takes the gesture. */
  touch?: "scroll" | "draw"
  /** Device-pixel-ratio cap. The shader is per-pixel, so 2 is plenty. */
  maxDpr?: number
  /** Content laid over the sky. Pointer events pass through unless opted in. */
  children?: React.ReactNode
  className?: string
}

export default function HalftoneNebula({
  height = "100svh",
  preset = "crimson",
  params,
  interactive = true,
  touch = "scroll",
  maxDpr = 2,
  children,
  className = "",
}: HalftoneNebulaProps) {
  const rootRef = React.useRef<HTMLElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [failed, setFailed] = React.useState(false)
  const [generation, setGeneration] = React.useState(0)
  const reduced = React.useSyncExternalStore(subscribeReduced, readReduced, serverReduced)

  // Defaults < preset < explicit params.
  const P = React.useMemo<NebulaParams>(
    () => ({ ...NEBULA_DEFAULTS, ...(NEBULA_PRESETS[preset] ?? {}), ...(params ?? {}) }),
    [preset, params],
  )
  // The loop reads through a ref, so tuning a value never restarts WebGL.
  const paramsRef = React.useRef(P)
  React.useEffect(() => {
    paramsRef.current = P
  }, [P])

  React.useEffect(() => {
    const root = rootRef.current
    const canvas = canvasRef.current
    if (!root || !canvas) return

    const gl = canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      depth: false,
      powerPreference: "high-performance",
    })
    if (!gl) {
      setFailed(true)
      return
    }

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)
      if (!s) return null
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("halftone-nebula:", gl.getShaderInfoLog(s))
        gl.deleteShader(s)
        return null
      }
      return s
    }
    const vs = compile(gl.VERTEX_SHADER, VERT)
    const fs = compile(gl.FRAGMENT_SHADER, FRAG)
    const program = vs && fs ? gl.createProgram() : null
    if (!program || !vs || !fs) {
      setFailed(true)
      return
    }
    gl.attachShader(program, vs)
    gl.attachShader(program, fs)
    gl.linkProgram(program)
    gl.deleteShader(vs)
    gl.deleteShader(fs)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("halftone-nebula:", gl.getProgramInfoLog(program))
      gl.deleteProgram(program)
      setFailed(true)
      return
    }

    const loc = (n: string) => gl.getUniformLocation(program, n)
    const tuned = UNIFORMS.map(([k, kind]) => [loc(uName(k)), k, kind] as const)
    const U = {
      res: loc("uRes"), dpr: loc("uDpr"), time: loc("uTime"),
      pointer: loc("uPointer"), pointerOn: loc("uPointerOn"), look: loc("uLook"),
      sparkCount: loc("uSparkCount"), spark: loc("uSpark"), sparkB: loc("uSparkB"),
      ripple: loc("uRipple"), planet: loc("uPlanet"),
    }
    const vao = gl.createVertexArray()

    // ---- sizing, from the element rather than the window --------------------
    let cssW = 1
    let cssH = 1
    let dpr = 1
    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, Math.max(maxDpr, 0.5))
      cssW = Math.max(canvas.clientWidth, 1)
      cssH = Math.max(canvas.clientHeight, 1)
      const w = Math.floor(cssW * dpr)
      const h = Math.floor(cssH * dpr)
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      if (reduced) paint()
    }

    // ---- the sky ------------------------------------------------------------
    let sparks = layoutSparkles(paramsRef.current.seed, paramsRef.current.sparkles)
    let laidOut = paramsRef.current.seed + ":" + paramsRef.current.sparkles
    const flare = new Float32Array(MAX_SPARKS)
    const sparkA = new Float32Array(MAX_SPARKS * 4)
    const sparkB = new Float32Array(MAX_SPARKS * 4)
    const ripples = new Float32Array(MAX_RIPPLES * 4)
    let rippleNext = 0

    // Reduced motion freezes the clock on a frame that is already lit.
    const FROZEN = 14
    const t0 = performance.now()
    const clock = () => (reduced ? FROZEN : ((performance.now() - t0) / 1000) * paramsRef.current.speed)
    let lastClock = 0

    // ---- the pointer --------------------------------------------------------
    let targetX = 0.5
    let targetY = 0.5
    let lampX = 0.5
    let lampY = 0.5
    let lookX = 0
    let lookY = 0
    let lampOn = 0
    let inside = false
    let lastTouched = -1e9

    const toUv = (e: PointerEvent): [number, number] => {
      const r = canvas.getBoundingClientRect()
      return [
        Math.min(Math.max((e.clientX - r.left) / Math.max(r.width, 1), 0), 1),
        Math.min(Math.max(1 - (e.clientY - r.top) / Math.max(r.height, 1), 0), 1),
      ]
    }
    const onMove = (e: PointerEvent) => {
      if (!interactive) return
      ;[targetX, targetY] = toUv(e)
      inside = true
      lastTouched = performance.now() / 1000
    }
    const onLeave = () => {
      inside = false
    }
    const onDown = (e: PointerEvent) => {
      if (!interactive || e.button > 0) return
      // Let buttons and links in the overlay keep their clicks, and let anything
      // marked data-nebula-ignore (a results panel, say) be clicked without
      // hanging a star behind it.
      if (
        (e.target as HTMLElement | null)?.closest(
          "a,button,input,textarea,select,label,[role=button],[data-nebula-ignore]",
        )
      )
        return
      const [x, y] = toUv(e)
      targetX = x
      targetY = y
      inside = true
      lastTouched = performance.now() / 1000
      const now = clock()
      const rand = Math.random()
      sparks = pushSparkle(sparks, {
        x, y,
        reach: 0.045 + 0.08 * rand,
        core: 0.006 + 0.008 * rand,
        tint: Math.random() < 0.2 ? 1 : 0,
        phase: Math.random() * 6.283,
        born: reduced ? now - 10 : now,
        user: true,
      })
      ripples.set([x * cssW, y * cssH, now, reduced ? 0 : 1], rippleNext * 4)
      rippleNext = (rippleNext + 1) % MAX_RIPPLES
      if (reduced) paint()
    }

    root.addEventListener("pointermove", onMove)
    root.addEventListener("pointerdown", onDown)
    root.addEventListener("pointerleave", onLeave)
    root.addEventListener("pointercancel", onLeave)

    const onLost = (e: Event) => {
      e.preventDefault()
      cancelAnimationFrame(raf)
      raf = 0
    }
    const onRestored = () => setGeneration((g) => g + 1)
    canvas.addEventListener("webglcontextlost", onLost)
    canvas.addEventListener("webglcontextrestored", onRestored)

    // ---- one frame ----------------------------------------------------------
    const paint = () => {
      const p = paramsRef.current
      const key = p.seed + ":" + p.sparkles
      if (key !== laidOut) {
        sparks = [...layoutSparkles(p.seed, p.sparkles), ...sparks.filter((s) => s.user)].slice(0, MAX_SPARKS)
        laidOut = key
      }
      const t = clock()
      const dt = Math.min(Math.max(t - lastClock, 0), 0.1)
      lastClock = t

      // With no hand on it, the lamp wanders on its own so the sky never sits
      // dead — and a still capture of it still shows the lens.
      const idle = !inside || performance.now() / 1000 - lastTouched > 6
      const [gx, gy] = driftPos(t * 0.6)
      const wantX = idle ? gx : targetX
      const wantY = idle ? gy : targetY
      const k = reduced ? 1 : 1 - Math.exp(-dt * (idle ? 1.5 : 7))
      lampX += (wantX - lampX) * k
      lampY += (wantY - lampY) * k
      lookX += ((wantX - 0.5) * 2 - lookX) * k
      lookY += ((wantY - 0.5) * 2 - lookY) * k
      const onTarget = interactive && !reduced ? (idle ? 0.55 : 1) : 0
      lampOn += (onTarget - lampOn) * (reduced ? 1 : 1 - Math.exp(-dt * 3))

      const minSide = Math.min(cssW, cssH)
      const n = Math.min(sparks.length, MAX_SPARKS)
      for (let i = 0; i < n; i++) {
        const s = sparks[i]
        const reach = s.reach * minSide
        const d = Math.hypot(s.x * cssW - lampX * cssW, s.y * cssH - lampY * cssH)
        const want = interactive && !reduced ? Math.exp(-(d * d) / Math.max(reach * reach * 0.6, 400)) : 0
        flare[i] += (want - flare[i]) * (reduced ? 1 : 1 - Math.exp(-dt * 6))
        sparkA.set([s.x * cssW, s.y * cssH, reach, s.born], i * 4)
        sparkB.set([s.tint, s.phase, flare[i], Math.max(s.core * minSide, 1.5)], i * 4)
      }

      gl.useProgram(program)
      gl.bindVertexArray(vao)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      for (const [l, key, kind] of tuned) {
        const v = p[key]
        if (kind === "c") gl.uniform3fv(l, hexToRgb(v as string))
        else if (kind === "i" || kind === "b") gl.uniform1i(l, Number(v) | 0)
        else gl.uniform1f(l, v as number)
      }
      gl.uniform2f(U.res, canvas.width, canvas.height)
      gl.uniform1f(U.dpr, canvas.width / cssW)
      gl.uniform1f(U.time, t)
      gl.uniform2f(U.pointer, lampX * cssW, lampY * cssH)
      gl.uniform1f(U.pointerOn, lampOn)
      gl.uniform2f(U.look, lookX, lookY)
      gl.uniform1i(U.sparkCount, n)
      gl.uniform4fv(U.spark, sparkA)
      gl.uniform4fv(U.sparkB, sparkB)
      gl.uniform4fv(U.ripple, ripples)
      gl.uniform4f(U.planet, p.planetX * cssW, p.planetY * cssH, p.planetRadius * minSide, p.planet ? 1 : 0)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
    }

    // ---- loop, paused whenever nobody can see it -----------------------------
    let raf = 0
    let visible = true
    const frame = () => {
      raf = 0
      paint()
      if (visible && !document.hidden) raf = requestAnimationFrame(frame)
    }
    const wake = () => {
      if (!reduced && !raf && visible && !document.hidden) raf = requestAnimationFrame(frame)
    }
    const io = new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting)
      wake()
    })
    io.observe(root)
    document.addEventListener("visibilitychange", wake)

    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)

    // Paint before the first rAF so nothing ever flashes an empty canvas.
    paint()
    if (reduced) {
      // Nothing moves; one frame is the whole piece.
    } else {
      wake()
    }

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
      io.disconnect()
      document.removeEventListener("visibilitychange", wake)
      root.removeEventListener("pointermove", onMove)
      root.removeEventListener("pointerdown", onDown)
      root.removeEventListener("pointerleave", onLeave)
      root.removeEventListener("pointercancel", onLeave)
      canvas.removeEventListener("webglcontextlost", onLost)
      canvas.removeEventListener("webglcontextrestored", onRestored)
      gl.deleteVertexArray(vao)
      gl.deleteProgram(program)
    }
  }, [interactive, reduced, generation, maxDpr])

  return (
    <section
      ref={rootRef}
      className={
        "relative w-full overflow-hidden bg-[#050309] " +
        (interactive ? "cursor-crosshair " : "") +
        (touch === "draw" ? "touch-none " : "touch-pan-y ") +
        className
      }
      style={{ height }}
      aria-label="A pixel-art nebula printed in halftone dots"
    >
      {failed ? (
        // No WebGL2: a still picture of the same sky beats a black box.
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(60% 45% at 12% 88%, #ff1f5a 0%, #c01245 22%, #5c0d31 50%, transparent 75%)," +
              "radial-gradient(45% 30% at 22% 22%, #5c0d31 0%, transparent 70%)," +
              "radial-gradient(40% 30% at 80% 30%, #3b1646 0%, transparent 70%)," +
              "#050309",
          }}
        />
      ) : (
        <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 block h-full w-full" />
      )}
      {children ? (
        <div className="pointer-events-none relative z-10 flex h-full w-full flex-col">{children}</div>
      ) : null}
    </section>
  )
}
