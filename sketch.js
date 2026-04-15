// Wingspan — MediaPipe Pose + Face Landmarker: wings, plumage, face shield; eagle eyes on iris mesh (pose fallback).
// Default 2D canvas (WEBGL broke visibility for line/image-style drawing without extra lighting).
// I = still reference image, V = webcam (mirrored). Needs local or https server.

let poseLandmarker;
/** MediaPipe Face Landmarker (478 pts) — iris centers for exact eagle-eye placement. */
let faceLandmarker;
let capture;
let refImg;
/** @type {'video' | 'image'} */
let inputMode = "video";

/** Previous-frame screen positions for motion-reactive feathers. */
let prevProj = null;

/** Per-side smoothed plumage (left / right arms drive their own feathers; chest stays mirrored). */
let smoothPlumage = {
  left: { rest: 0.45, expand: 0.55 },
  right: { rest: 0.45, expand: 0.55 },
};

/** Extra smoothing for symmetric back wings (slower, more fluid pose response). */
let smoothWingPlumage = { expand: 0.55, rest: 0.45 };

/** Wing feathers use slightly lagged combined react so motion feels heavier / more supple. */
let smoothWingReact = { vx: 0, vy: 0, speed: 0 };

/** Low-pass filtered velocities for feather react (less jitter, more elastic follow). */
let smoothMotionReact = {
  left: { vx: 0, vy: 0, speed: 0 },
  right: { vx: 0, vy: 0, speed: 0 },
  chest: { vx: 0, vy: 0, speed: 0 },
  both: { vx: 0, vy: 0, speed: 0 },
};

/** Space toggles wind / wing-flap ambience (Web Audio). */
let wingWind = {
  enabled: false,
  ctx: null,
  src: null,
  gain: null,
  hp: null,
  lp: null,
  graphReady: false,
  env: 0,
  prevDrive: 0,
  /** Smoothed 0 = start of day (low, dark wind) → 1 = end of day (brighter, higher). */
  smoothDayPitch: 0,
};

const SILENT_ARM_MOTION = {
  left: { vx: 0, vy: 0, speed: 0 },
  right: { vx: 0, vy: 0, speed: 0 },
};

/** Smoothed 0–1 from arm speed: tints plumage + drives sparks. */
let smoothSpeedGlow = 0;

/** Smoothed screen slots { x, y, span } from face mesh (iris-accurate). */
let smoothFaceEagleL = null;
let smoothFaceEagleR = null;

function tintPlumageWithSpeed(rgb, glow, warmth) {
  const g = constrain(glow, 0, 1) * warmth;
  return [
    min(255, rgb[0] + g * 78 + g * g * 38),
    min(255, rgb[1] + g * 58 + g * g * 42),
    min(255, rgb[2] + g * 34 + g * g * 62),
  ];
}

const MODEL_URI =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
/** Full face mesh + iris (468 / 473 = iris centers). */
const FACE_MODEL_URI =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const TASKS_VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const WASM_PATH = `${TASKS_VISION}/wasm`;

// Warm raptor / large-bird palette (high contrast, reads clearly on video)
const RACHIS = [38, 26, 18];
const RACHIS_HIGHLIGHT = [85, 62, 42];
const VANE_FACE = [248, 228, 195];
const VANE_BODY = [218, 178, 118];
const VANE_SHADOW = [155, 118, 72];
const BAR_TIP = [52, 38, 28];
const ACCENT_RUST = [160, 78, 38];

const TORSO = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
];
const ARMS_LEFT = [
  [11, 13],
  [13, 15],
];
const ARMS_RIGHT = [
  [12, 14],
  [14, 16],
];
const HANDS_LEFT = [
  [15, 19],
  [15, 21],
  [15, 17],
];
const HANDS_RIGHT = [
  [16, 20],
  [16, 22],
  [16, 18],
];

/** BlazePose: L 17 pinky, 19 index, 21 thumb · R 18, 20, 22 */
const HAND_LEFT_TIPS = [17, 19, 21];
const HAND_RIGHT_TIPS = [18, 20, 22];

/** Smoothed 0–1+ finger splay per hand. */
let smoothHandSpread = { left: 0.35, right: 0.35 };
/** Smoothed mean pairwise tip distance (px); ring gaps ∝ this. */
let smoothTipSpreadPx = { left: 36, right: 36 };

/** Pose face mesh (nose, eyes, ears, mouth) — used to keep feathers from drawing over the face. */
const POSE_FACE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

function preload() {
  refImg = loadImage("assets/reference-angel.png");
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function keyPressed() {
  if (key === " " || key === "Spacebar" || keyCode === 32) {
    toggleWingWindSound();
    return false;
  }
  if (key === "i" || key === "I") inputMode = "image";
  if (key === "v" || key === "V") inputMode = "video";
}

function makeBrownNoiseBuffer(ctx, seconds = 2.5) {
  const rate = ctx.sampleRate;
  const n = Math.floor(rate * seconds);
  const buf = ctx.createBuffer(1, n, rate);
  const data = buf.getChannelData(0);
  let last = 0;
  for (let i = 0; i < n; i++) {
    const white = Math.random() * 2 - 1;
    last = (last + 0.024 * white) * 0.983;
    data[i] = constrain(last * 4.2, -1, 1);
  }
  return buf;
}

function ensureWingWindGraph() {
  if (wingWind.graphReady) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  wingWind.ctx = new AC();
  const ctx = wingWind.ctx;
  const noiseBuf = makeBrownNoiseBuffer(ctx);
  const src = ctx.createBufferSource();
  src.buffer = noiseBuf;
  src.loop = true;
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 200;
  hp.Q.value = 0.65;
  const lp = ctx.createBiquadFilter();
  lp.type = "lowpass";
  lp.frequency.value = 1400;
  lp.Q.value = 0.82;
  const gain = ctx.createGain();
  gain.gain.value = 0;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -32;
  comp.knee.value = 24;
  comp.ratio.value = 3.2;
  comp.attack.value = 0.004;
  comp.release.value = 0.28;
  src.connect(hp);
  hp.connect(lp);
  lp.connect(gain);
  gain.connect(comp);
  comp.connect(ctx.destination);
  src.start(0);
  wingWind.src = src;
  wingWind.gain = gain;
  wingWind.hp = hp;
  wingWind.lp = lp;
  wingWind.graphReady = true;
}

/**
 * Real clock: morning (from ~6:00) = 0 very low / dark; late evening (~22:30+) = 1 brighter, higher.
 * Before 6:00 stays at 0; after ~22:30 at 1 so a new day can feel like a fresh low start again overnight.
 */
function clockDayPitchPhase() {
  const d = new Date();
  const h = d.getHours() + d.getMinutes() / 60 + d.getSeconds() / 3600;
  const dayStart = 6;
  const dayEnd = 22.5;
  if (h < dayStart) return 0;
  if (h >= dayEnd) return 1;
  return (h - dayStart) / (dayEnd - dayStart);
}

function toggleWingWindSound() {
  wingWind.enabled = !wingWind.enabled;
  if (wingWind.enabled) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) {
      wingWind.enabled = false;
      return;
    }
    ensureWingWindGraph();
    if (!wingWind.graphReady) {
      wingWind.enabled = false;
      return;
    }
    if (wingWind.ctx && wingWind.ctx.state === "suspended") wingWind.ctx.resume();
    wingWind.smoothDayPitch = clockDayPitchPhase();
  } else if (wingWind.gain && wingWind.ctx) {
    const t = wingWind.ctx.currentTime;
    wingWind.gain.gain.cancelScheduledValues(t);
    wingWind.gain.gain.setValueAtTime(wingWind.gain.gain.value, t);
    wingWind.gain.gain.setTargetAtTime(0, t, 0.035);
    wingWind.env = 0;
    wingWind.prevDrive = 0;
  }
}

/**
 * Arm speed → filtered noise (breeze); spikes on acceleration read as flaps.
 * Time of day slowly raises spectral “pitch” (brighter highs) from morning to evening.
 */
function updateWingWindSound(motion) {
  if (!wingWind.enabled || !wingWind.graphReady || !wingWind.ctx || !wingWind.gain || !wingWind.lp)
    return;
  if (wingWind.ctx.state === "suspended") wingWind.ctx.resume();

  const dayTarget = clockDayPitchPhase();
  wingWind.smoothDayPitch = lerp(wingWind.smoothDayPitch, dayTarget, 0.012);
  const day = constrain(wingWind.smoothDayPitch, 0, 1);
  const dayEase = pow(day, 0.82);

  const left = motion?.left?.speed ?? 0;
  const right = motion?.right?.speed ?? 0;
  const raw = (left + right) * 0.5 + Math.abs(left - right) * 0.22;
  const drive = constrain(raw * 0.031, 0, 1.35);

  const kUp = 0.26;
  const kDn = 0.055;
  wingWind.env += (drive - wingWind.env) * (drive > wingWind.env ? kUp : kDn);

  const delta = max(0, drive - wingWind.prevDrive);
  wingWind.prevDrive = lerp(wingWind.prevDrive, drive, 0.22);
  const flap = min(1, delta * 3.2);
  const level = constrain(wingWind.env * 0.68 + flap * 0.52, 0, 1);

  const t = wingWind.ctx.currentTime;
  const g = level * 0.36;
  wingWind.gain.gain.setTargetAtTime(g, t, 0.038);

  const lpFloor = lerp(340, 1950, dayEase);
  const lpMotion = level * lerp(520, 3400, dayEase);
  const lpTarget = lpFloor + lpMotion;
  wingWind.lp.frequency.setTargetAtTime(lpTarget, t, 0.055);

  if (wingWind.hp) {
    const hpTarget = lerp(72, 340, dayEase);
    wingWind.hp.frequency.setTargetAtTime(hpTarget, t, 0.07);
    wingWind.hp.Q.value = lerp(0.52, 0.78, dayEase);
  }
}

function setup() {
  createCanvas(windowWidth, windowHeight);

  capture = createCapture(VIDEO);
  capture.size(640, 480);
  capture.hide();

  initPoseLandmarker().catch((err) => {
    console.error(err);
  });
}

async function initPoseLandmarker() {
  await new Promise((resolve, reject) => {
    const el = capture.elt;
    const done = () => resolve();
    el.onloadeddata = done;
    el.onerror = () => reject(new Error("Webcam failed to start"));
    if (el.readyState >= 2) done();
  });

  const { PoseLandmarker, FaceLandmarker, FilesetResolver } = await import(`${TASKS_VISION}/+esm`);
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  const makePoseOptions = (delegate) => ({
    baseOptions: {
      modelAssetPath: MODEL_URI,
      delegate,
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, makePoseOptions("GPU"));
  } catch {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, makePoseOptions("CPU"));
  }

  const makeFaceOptions = (delegate) => ({
    baseOptions: {
      modelAssetPath: FACE_MODEL_URI,
      delegate,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.42,
    minFacePresenceConfidence: 0.42,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, makeFaceOptions("GPU"));
  } catch (e) {
    console.warn("Face Landmarker GPU failed, trying CPU", e);
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, makeFaceOptions("CPU"));
    } catch (e2) {
      console.warn("Face Landmarker unavailable — eagle eyes use pose face points only", e2);
      faceLandmarker = null;
    }
  }
}

function inputDims() {
  if (inputMode === "image" && refImg && refImg.width > 0) {
    return { vw: refImg.width, vh: refImg.height };
  }
  return { vw: capture.width, vh: capture.height };
}

function videoCoverLayout() {
  const { vw, vh } = inputDims();
  const s = Math.max(width / vw, height / vh);
  const dw = vw * s;
  const dh = vh * s;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;
  return { dw, dh, ox, oy };
}

function project(lm) {
  const { dw, dh, ox, oy } = videoCoverLayout();
  const xNorm = inputMode === "image" ? lm.x : 1 - lm.x;
  return { x: ox + xNorm * dw, y: oy + lm.y * dh };
}

/** Face mesh landmarks use the same normalized image space as pose — identical screen mapping. */
function projectFacePoint(pt) {
  if (!pt || pt.x === undefined || pt.y === undefined) return null;
  return project({ x: pt.x, y: pt.y, visibility: 1 });
}

/**
 * Iris centers (468 L / 473 R) + eye-corner span in screen px (MediaPipe 478 topology).
 */
function faceMeshEyeSlots(faceResult) {
  const fl = faceResult?.faceLandmarks?.[0];
  if (!fl || fl.length < 474) return null;

  const span2d = (i, j) => {
    if (!fl[i] || !fl[j]) return 0;
    const a = projectFacePoint(fl[i]);
    const b = projectFacePoint(fl[j]);
    if (!a || !b) return 0;
    return Math.hypot(b.x - a.x, b.y - a.y);
  };

  const PL = projectFacePoint(fl[468]);
  const PR = projectFacePoint(fl[473]);
  if (!PL || !PR) return null;

  const leftSpan = span2d(362, 263) || span2d(374, 263);
  const rightSpan = span2d(398, 133) || span2d(398, 33) || span2d(385, 263);

  return {
    L: { x: PL.x, y: PL.y, span: leftSpan },
    R: { x: PR.x, y: PR.y, span: rightSpan },
  };
}

function smoothFaceEagleSlot(prev, next, kPos, kSpan) {
  if (!next) return null;
  if (!prev) return { x: next.x, y: next.y, span: next.span };
  return {
    x: lerp(prev.x, next.x, kPos),
    y: lerp(prev.y, next.y, kPos),
    span: lerp(prev.span, next.span, kSpan),
  };
}

function drawInputCover() {
  const { dw, dh, ox, oy } = videoCoverLayout();
  if (inputMode === "image" && refImg && refImg.width > 0) {
    image(refImg, ox, oy, dw, dh);
  } else {
    push();
    translate(ox + dw, oy);
    scale(-1, 1);
    image(capture, 0, 0, dw, dh);
    pop();
  }
}

function poseSourceElt() {
  if (inputMode === "image" && refImg && refImg.width > 0) return refImg.elt;
  return capture.elt;
}

function visibleEnough(lm, thresh = 0.25) {
  return lm && (lm.visibility === undefined || lm.visibility >= thresh);
}

function torsoCenterProj(lm) {
  const idx = [11, 12, 23, 24];
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const i of idx) {
    if (!visibleEnough(lm[i], 0.15)) continue;
    const p = project(lm[i]);
    sx += p.x;
    sy += p.y;
    n++;
  }
  if (n === 0) return null;
  return { x: sx / n, y: sy / n };
}

function shoulderWidthPx(lm) {
  if (!visibleEnough(lm[11]) || !visibleEnough(lm[12])) return 120;
  const a = project(lm[11]);
  const b = project(lm[12]);
  return Math.hypot(b.x - a.x, b.y - a.y) || 1;
}

/** Mean screen distance between each pair of fingertips (index, pinky, thumb). */
function meanPairwiseTipSpreadPx(lm, tipIndices) {
  const pts = [];
  for (const i of tipIndices) {
    if (!visibleEnough(lm[i], 0.12)) continue;
    pts.push(project(lm[i]));
  }
  if (pts.length < 2) return null;
  let sum = 0;
  let c = 0;
  for (let a = 0; a < pts.length; a++) {
    for (let b = a + 1; b < pts.length; b++) {
      sum += Math.hypot(pts[b].x - pts[a].x, pts[b].y - pts[a].y);
      c++;
    }
  }
  return c ? sum / c : null;
}

/**
 * How open the hand is: fingertip spread relative to forearm length (roughly 0 closed, 1+ wide).
 * Returns null if wrist/forearm/tips not visible enough.
 */
function handSpreadNormalized(lm, isRight) {
  const wrist = isRight ? 16 : 15;
  const el = isRight ? 14 : 13;
  const tips = isRight ? HAND_RIGHT_TIPS : HAND_LEFT_TIPS;
  if (!visibleEnough(lm[wrist], 0.12) || !visibleEnough(lm[el], 0.12)) return null;
  const spreadPx = meanPairwiseTipSpreadPx(lm, tips);
  if (spreadPx === null) return null;
  const Pw = project(lm[wrist]);
  const Pe = project(lm[el]);
  const forearm = Math.hypot(Pw.x - Pe.x, Pw.y - Pe.y);
  const denom = max(forearm * 0.4, 26);
  const raw = spreadPx / denom;
  return constrain((raw - 0.2) / 0.58, 0, 1.45);
}

function updateSmoothHandSpread(lm) {
  if (!lm) {
    smoothHandSpread.left = lerp(smoothHandSpread.left, 0.32, 0.06);
    smoothHandSpread.right = lerp(smoothHandSpread.right, 0.32, 0.06);
    smoothTipSpreadPx.left = lerp(smoothTipSpreadPx.left, 32, 0.05);
    smoothTipSpreadPx.right = lerp(smoothTipSpreadPx.right, 32, 0.05);
    return;
  }
  const lN = handSpreadNormalized(lm, false);
  const rN = handSpreadNormalized(lm, true);
  const kUp = 0.22;
  const kDn = 0.12;
  if (lN !== null) {
    const k = lN > smoothHandSpread.left ? kUp : kDn;
    smoothHandSpread.left = lerp(smoothHandSpread.left, lN, k);
  } else smoothHandSpread.left = lerp(smoothHandSpread.left, 0.32, 0.08);
  if (rN !== null) {
    const k = rN > smoothHandSpread.right ? kUp : kDn;
    smoothHandSpread.right = lerp(smoothHandSpread.right, rN, k);
  } else smoothHandSpread.right = lerp(smoothHandSpread.right, 0.32, 0.08);

  const spL = meanPairwiseTipSpreadPx(lm, HAND_LEFT_TIPS);
  const spR = meanPairwiseTipSpreadPx(lm, HAND_RIGHT_TIPS);
  const kt = 0.24;
  if (spL !== null) smoothTipSpreadPx.left = lerp(smoothTipSpreadPx.left, spL, kt);
  else {
    const el = visibleEnough(lm[13], 0.12) ? project(lm[13]) : null;
    const wr = visibleEnough(lm[15], 0.12) ? project(lm[15]) : null;
    const fa = el && wr ? Math.hypot(wr.x - el.x, wr.y - el.y) : shoulderWidthPx(lm) * 0.32;
    const guess = fa * (0.2 + 0.42 * smoothHandSpread.left);
    smoothTipSpreadPx.left = lerp(smoothTipSpreadPx.left, guess, 0.1);
  }
  if (spR !== null) smoothTipSpreadPx.right = lerp(smoothTipSpreadPx.right, spR, kt);
  else {
    const el = visibleEnough(lm[14], 0.12) ? project(lm[14]) : null;
    const wr = visibleEnough(lm[16], 0.12) ? project(lm[16]) : null;
    const fa = el && wr ? Math.hypot(wr.x - el.x, wr.y - el.y) : shoulderWidthPx(lm) * 0.32;
    const guess = fa * (0.2 + 0.42 * smoothHandSpread.right);
    smoothTipSpreadPx.right = lerp(smoothTipSpreadPx.right, guess, 0.1);
  }
}

/** Concentric rings at wrist; gap between rings scales with fingertip spread (same proportion). */
function drawHandFingerRings(landmarks) {
  if (!landmarks) return;

  function oneHand(isRight) {
    const wrist = isRight ? 16 : 15;
    if (!visibleEnough(landmarks[wrist], 0.15)) return;
    const w = project(landmarks[wrist]);
    const spreadPx = isRight ? smoothTipSpreadPx.right : smoothTipSpreadPx.left;
    const stepR = max(10, spreadPx * 0.29);
    const nRings = 6;
    push();
    noFill();
    strokeCap(ROUND);
    for (let i = 1; i <= nRings; i++) {
      const diam = 2 * stepR * i;
      const a = inputMode === "image" ? 38 + i * 14 : 28 + i * 12;
      stroke(160, 215, 255, constrain(a, 32, 120));
      strokeWeight(1.15 + i * 0.06);
      circle(w.x, w.y, diam);
    }
    pop();
  }

  oneHand(false);
  oneHand(true);
}

/**
 * Axis-aligned region covering the head/face in screen space (from pose face landmarks).
 * Feathers that intersect this box are not drawn so the face stays unobstructed.
 */
function computeFaceShieldBounds(lm) {
  if (!lm) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let n = 0;
  for (const i of POSE_FACE_INDICES) {
    if (!lm[i] || !visibleEnough(lm[i], 0.12)) continue;
    const p = project(lm[i]);
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
    n++;
  }
  if (n >= 2) {
    const w = maxX - minX;
    const h = maxY - minY;
    const padX = Math.max(w * 0.55, h * 0.5, 20);
    const padY = Math.max(h * 0.72, w * 0.55, 24);
    const upLift = Math.max(h * 0.95, w * 0.4);
    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY - upLift,
      maxY: maxY + padY * 0.55,
    };
  }
  if (visibleEnough(lm[0], 0.08)) {
    const p = project(lm[0]);
    const sw = shoulderWidthPx(lm);
    const r = Math.max(sw * 0.26, 44);
    return {
      minX: p.x - r * 1.08,
      maxX: p.x + r * 1.08,
      minY: p.y - r * 1.4,
      maxY: p.y + r,
    };
  }
  return null;
}

function pointInFaceRect(x, y, b) {
  return x >= b.minX && x <= b.maxX && y >= b.minY && y <= b.maxY;
}

function snapshotProj(lm) {
  const snap = {};
  for (let i = 0; i < 33; i++) {
    if (lm[i] && visibleEnough(lm[i], 0.08)) snap[i] = { ...project(lm[i]) };
  }
  return snap;
}

/** Average screen velocity (px/frame) for a set of landmark indices. */
function velocityForIndices(lm, indices) {
  if (!prevProj) return { vx: 0, vy: 0, speed: 0 };
  let vx = 0;
  let vy = 0;
  let n = 0;
  for (const i of indices) {
    if (!visibleEnough(lm[i], 0.12) || !prevProj[i]) continue;
    const p = project(lm[i]);
    vx += p.x - prevProj[i].x;
    vy += p.y - prevProj[i].y;
    n++;
  }
  if (!n) return { vx: 0, vy: 0, speed: 0 };
  vx /= n;
  vy /= n;
  const cap = 42;
  vx = constrain(vx, -cap, cap);
  vy = constrain(vy, -cap, cap);
  return { vx, vy, speed: Math.hypot(vx, vy) };
}

function smoothReactCell(dst, src, k) {
  dst.vx = lerp(dst.vx, src.vx, k);
  dst.vy = lerp(dst.vy, src.vy, k);
  dst.speed = lerp(dst.speed, src.speed, k * 0.78);
}

/** Feather react uses this; raw frame-delta velocities are jittery from pose + video. */
function smoothMotionBundle(raw) {
  const k = 0.125;
  smoothReactCell(smoothMotionReact.left, raw.left, k);
  smoothReactCell(smoothMotionReact.right, raw.right, k);
  smoothReactCell(smoothMotionReact.chest, raw.chest, k);
  smoothReactCell(smoothMotionReact.both, raw.both, k * 0.92);
  return smoothMotionReact;
}

function armMotionBundle(lm) {
  const left = velocityForIndices(lm, [11, 13, 15]);
  const right = velocityForIndices(lm, [12, 14, 16]);
  const chest = {
    vx: (left.vx + right.vx) * 0.5,
    vy: (left.vy + right.vy) * 0.5,
    speed: (left.speed + right.speed) * 0.5,
  };
  const both = {
    vx: chest.vx,
    vy: chest.vy,
    speed: Math.max(left.speed, right.speed),
  };
  return { left, right, chest, both };
}

/** Inward normal (toward torso) perpendicular to upper arm / forearm segment. */
function armSegmentFrame(lm, iA, iB) {
  if (!visibleEnough(lm[iA]) || !visibleEnough(lm[iB])) return null;
  const a = project(lm[iA]);
  const b = project(lm[iB]);
  const cx = (a.x + b.x) * 0.5;
  const cy = (a.y + b.y) * 0.5;
  let tx = b.x - a.x;
  let ty = b.y - a.y;
  const tl = Math.hypot(tx, ty) || 1e-6;
  tx /= tl;
  ty /= tl;
  let nx = -ty;
  let ny = tx;
  const tc = torsoCenterProj(lm);
  if (!tc) return null;
  if (nx * (tc.x - cx) + ny * (tc.y - cy) < 0) {
    nx = -nx;
    ny = -ny;
  }
  return { a, b, tx, ty, nx, ny, segLen: tl };
}

/** Shared drive: wrists wide → bigger fan; elbows higher → longer plumes. */
function globalArmDrive(lm) {
  const sw = shoulderWidthPx(lm);
  let wristSpread = sw;
  if (visibleEnough(lm[15], 0.2) && visibleEnough(lm[16], 0.2)) {
    const wL = project(lm[15]);
    const wR = project(lm[16]);
    wristSpread = Math.hypot(wR.x - wL.x, wR.y - wL.y);
  }
  const spreadFactor = constrain(wristSpread / sw, 0.75, 2.6);

  let lift = 0;
  let n = 0;
  for (const [sh, el] of [
    [11, 13],
    [12, 14],
  ]) {
    if (!visibleEnough(lm[sh]) || !visibleEnough(lm[el])) continue;
    lift += (project(lm[sh]).y - project(lm[el]).y) / sw;
    n++;
  }
  lift = n ? lift / n : 0;
  const openFactor = constrain(0.55 + lift * 0.42 + (spreadFactor - 1) * 0.22, 0.42, 1.65);

  return { spreadFactor, openFactor, sw };
}

/**
 * One arm only: hang / lift / reach away from opposite shoulder + that side’s velocity.
 */
function computeArmSideTargets(lm, motion, isRight) {
  const sh = isRight ? 12 : 11;
  const el = isRight ? 14 : 13;
  const wr = isRight ? 16 : 15;
  const osh = isRight ? 11 : 12;
  const sw = shoulderWidthPx(lm) || 1;
  const vel = isRight ? motion?.right : motion?.left;

  if (!visibleEnough(lm[sh]) || !visibleEnough(lm[el])) {
    return { restDroop: 0.45, expandActive: 0.55 };
  }
  const ps = project(lm[sh]);
  const pe = project(lm[el]);
  const hang = constrain((pe.y - ps.y) / sw, -0.35, 1.35);
  const lift = constrain((ps.y - pe.y) / sw, -0.25, 1.15);

  let reachOut = 0;
  if (visibleEnough(lm[wr], 0.12) && visibleEnough(lm[osh], 0.12)) {
    const pw = project(lm[wr]);
    const po = project(lm[osh]);
    reachOut = constrain(Math.hypot(pw.x - po.x, pw.y - po.y) / sw, 0, 2.3);
  }
  const reachNorm = constrain((reachOut - 0.5) / 1.18, 0, 1);

  const g = globalArmDrive(lm);
  const globalSpread = constrain((g.spreadFactor - 0.82) / 1.45, 0, 1);
  const spreadBlend = constrain(reachNorm * 0.58 + globalSpread * 0.42, 0, 1);

  const move = vel ? constrain(vel.speed * 0.013, 0, 1.2) : 0;
  const restDroop = constrain(
    0.28 + hang * 0.62 - lift * 0.52 - spreadBlend * 0.4,
    0,
    1
  );
  const openFromPose = constrain(
    0.15 + lift * 0.54 + spreadBlend * 0.7 - hang * 0.2,
    0,
    1
  );
  const expandActive = constrain(openFromPose + move * 0.62, 0, 1.35);
  return { restDroop, expandActive };
}

function updateSmoothedPlumageSides(lm, motion) {
  const tL = computeArmSideTargets(lm, motion, false);
  const tR = computeArmSideTargets(lm, motion, true);
  smoothPlumage.left.rest = lerp(smoothPlumage.left.rest, tL.restDroop, 0.068);
  smoothPlumage.left.expand = lerp(smoothPlumage.left.expand, tL.expandActive, 0.086);
  smoothPlumage.right.rest = lerp(smoothPlumage.right.rest, tR.restDroop, 0.068);
  smoothPlumage.right.expand = lerp(smoothPlumage.right.expand, tR.expandActive, 0.086);
  const wTarget = blendWingPlumage(smoothPlumage.left, smoothPlumage.right);
  smoothWingPlumage.expand = lerp(smoothWingPlumage.expand, wTarget.expand, 0.042);
  smoothWingPlumage.rest = lerp(smoothWingPlumage.rest, wTarget.rest, 0.038);
  return smoothPlumage;
}

function stepSmoothWingReact(combined) {
  if (!combined) {
    smoothWingReact.vx *= 0.87;
    smoothWingReact.vy *= 0.87;
    smoothWingReact.speed *= 0.87;
    return smoothWingReact;
  }
  const k = 0.11;
  smoothWingReact.vx = lerp(smoothWingReact.vx, combined.vx, k);
  smoothWingReact.vy = lerp(smoothWingReact.vy, combined.vy, k);
  smoothWingReact.speed = lerp(smoothWingReact.speed, combined.speed, k * 0.88);
  return smoothWingReact;
}

function normalize2(x, y) {
  const L = Math.hypot(x, y) || 1;
  return { x: x / L, y: y / L };
}

/** Vertical mid-sagittal line (screen X) for bilateral mirror. */
function mirrorAxisX(lm) {
  const Pc = torsoCenterProj(lm);
  if (Pc) return Pc.x;
  const p11 = lm[11] && visibleEnough(lm[11], 0.1) ? project(lm[11]) : null;
  const p12 = lm[12] && visibleEnough(lm[12], 0.1) ? project(lm[12]) : null;
  if (p11 && p12) return (p11.x + p12.x) * 0.5;
  if (p12) return p12.x;
  if (p11) return p11.x;
  return width * 0.5;
}

/** Mirror feather base + direction across x = mx; react is set by caller. */
function mirrorFeatherAcrossX(f, mx) {
  return {
    ...f,
    bx: 2 * mx - f.bx,
    by: f.by,
    dirx: -f.dirx,
    diry: f.diry,
  };
}

function mirroredHorizontalReact(r) {
  if (!r) return null;
  return { vx: -r.vx, vy: r.vy, speed: r.speed };
}

/** Back wings: one silhouette for both sides; blend L/R arm posture. */
function blendWingPlumage(plL, plR) {
  const L = plL || { expand: 0.55, rest: 0.45 };
  const R = plR || { expand: 0.55, rest: 0.45 };
  return {
    expand: constrain((L.expand + R.expand) * 0.5, 0, 1),
    rest: constrain((L.rest + R.rest) * 0.5, 0, 1),
  };
}

function combineWingReact(ml, mr) {
  if (!ml && !mr) return null;
  if (!ml) return mr;
  if (!mr) return ml;
  return {
    vx: (ml.vx + mr.vx) * 0.5,
    vy: (ml.vy + mr.vy) * 0.5,
    speed: (ml.speed + mr.speed) * 0.5,
  };
}

/**
 * One side: roots sit medial-up from shoulder (“upper back”); plumes fan up & out past the arm line.
 * That side’s elbow biases outward direction so raising/swinging the arm pulls the wing.
 */
function buildSideWingFeathers(lm, isLeft, globalDrive, wingReact, plumage) {
  const TOP_WING_LEN_SCALE = 1.48;
  const TOP_WING_SPAN_SCALE = 1.14;
  const pl = plumage || { expand: 0.55, rest: 0.45 };
  const r = constrain(pl.rest, 0, 1);
  const e = constrain(pl.expand, 0, 1.35);
  const rEase = pow(r, 0.76);
  const eEase = pow(e, 0.88);
  const armOpenBoost =
    lerp(0.72, 1.26, eEase) * lerp(1.0, 0.56, rEase);
  const restLenMul = lerp(1.0, 0.5, rEase);
  /** Arms down: pack span & fold; arms out/up: wider, longer, showier primaries. */
  const spanMul =
    lerp(0.74, 1.12, eEase * (1 - 0.78 * rEase)) * lerp(0.86, 1.04, 1 - rEase * 0.32);

  const sh = isLeft ? 11 : 12;
  const el = isLeft ? 13 : 14;
  const wr = isLeft ? 15 : 16;
  const Pc = torsoCenterProj(lm);
  if (!Pc || !visibleEnough(lm[sh])) return [];

  const Ps = project(lm[sh]);
  const sw = globalDrive.sw;
  const inVec = normalize2(Pc.x - Ps.x, Pc.y - Ps.y);
  const root = {
    x: Ps.x + inVec.x * sw * 0.14 + 0,
    y: Ps.y + inVec.y * sw * 0.09 - sw * 0.13,
  };

  let out = normalize2(Ps.x - Pc.x, Ps.y - Pc.y);
  if (visibleEnough(lm[el], 0.2)) {
    const Pe = project(lm[el]);
    const toEl = normalize2(Pe.x - root.x, Pe.y - root.y);
    out = normalize2(out.x * 0.62 + toEl.x * 0.38, out.y * 0.62 + toEl.y * 0.38);
  }
  if (visibleEnough(lm[wr], 0.2)) {
    const Pw = project(lm[wr]);
    const toWr = normalize2(Pw.x - root.x, Pw.y - root.y);
    out = normalize2(out.x * 0.78 + toWr.x * 0.22, out.y * 0.78 + toWr.y * 0.22);
  }

  const tang = { x: -out.y, y: out.x };
  let towardOut = constrain(0.36 + globalDrive.spreadFactor * 0.24, 0.3, 0.96);
  towardOut *= lerp(1.0, 0.45, rEase);
  towardOut *= lerp(0.88, 1.06, eEase);

  let openSide = 0.68;
  if (visibleEnough(lm[el], 0.2)) {
    const Pe = project(lm[el]);
    openSide = constrain(0.54 + (Ps.y - Pe.y) / sw * 0.44, 0.46, 1.58);
  }
  openSide *= constrain(0.8 + (globalDrive.spreadFactor - 1) * 0.3, 0.62, 1.42);
  openSide *= lerp(1.0, 0.58, rEase);
  openSide *= lerp(0.9, 1.12, eEase);

  const refBoost = inputMode === "image" ? 1.06 : 1;
  const nMain = 24;
  const list = [];

  for (let i = 0; i < nMain; i++) {
    const u = i / Math.max(1, nMain - 1);
    const along =
      (i - (nMain - 1) * 0.5) * sw * 0.066 * TOP_WING_SPAN_SCALE * spanMul;
    const bx = root.x + tang.x * along;
    const by = root.y + tang.y * along;
    const mix = pow(u, 0.5) * min(1.05, towardOut * 1.08);
    let dx = lerp(0, out.x, mix);
    let dy = lerp(-1, out.y, mix);
    const droop = r * (0.46 + 0.26 * u);
    dx = lerp(dx, 0, droop * 0.52);
    dy = lerp(dy, 1, droop * 0.68);
    const d = normalize2(dx, dy);
    const outerPrimary = i >= nMain - 6;
    let spanBoost = outerPrimary ? 1.48 + (i - (nMain - 6)) * 0.055 : 1;
    if (outerPrimary) {
      spanBoost *= lerp(0.62, 1.08, eEase) * lerp(1.0, 0.72, rEase);
    }
    let len =
      sw *
      (0.5 + 1.58 * pow(u, 0.26)) *
      openSide *
      refBoost *
      1.38 *
      spanBoost *
      TOP_WING_LEN_SCALE *
      armOpenBoost *
      restLenMul *
      (0.9 + 0.1 * sin(u * PI));
    const phase = i * 2.7 + u * 6.1;
    list.push({
      bx,
      by,
      dirx: d.x,
      diry: d.y,
      len,
      phase,
      layer: u * 0.02 + (outerPrimary ? 0.15 : 0),
      react: wingReact,
      kind: outerPrimary ? "wingPrimary" : "wing",
      plumeRest: min(1, pl.rest * (0.52 + 0.22 * r)),
    });
  }

  for (let j = 0; j < 14; j++) {
    const u = (j + 0.5) / 14;
    const along = (j - 6.5) * sw * 0.036 * TOP_WING_SPAN_SCALE * spanMul;
    const bx = root.x + tang.x * along + inVec.x * sw * 0.024;
    const by = root.y + tang.y * along - sw * 0.055;
    const mix = pow(u, 0.48) * towardOut * 0.92;
    let dx = lerp(0, out.x, mix);
    let dy = lerp(-1, out.y, mix);
    const droopC = r * (0.42 + 0.18 * u);
    dx = lerp(dx, 0, droopC * 0.5);
    dy = lerp(dy, 1, droopC * 0.65);
    const d = normalize2(dx, dy);
    const len =
      sw *
      (0.3 + 0.62 * pow(u, 0.36)) *
      openSide *
      refBoost *
      0.92 *
      TOP_WING_LEN_SCALE *
      armOpenBoost *
      restLenMul;
    const phase = j * 3.1 + u * 4;
    list.push({
      bx,
      by,
      dirx: d.x,
      diry: d.y,
      len,
      phase,
      layer: -0.5 + u * 0.01,
      react: wingReact,
      kind: "covert",
      plumeRest: min(1, pl.rest * (0.52 + 0.22 * r)),
    });
  }

  return list;
}

/**
 * Symmetric back wings: geometry from the right template, mirrored across torso midline.
 * Plumage and flutter follow both arms (blend); arms still drive expand/rest via smoothing.
 */
function collectBackWingFeathers(lm, motion, plumage) {
  const g = globalArmDrive(lm);
  const mx = mirrorAxisX(lm);
  const plW = smoothWingPlumage;
  const wr = stepSmoothWingReact(combineWingReact(motion?.left, motion?.right));
  const v11 = visibleEnough(lm[11], 0.18);
  const v12 = visibleEnough(lm[12], 0.18);
  const list = [];

  if (v11 && v12) {
    const rightSide = buildSideWingFeathers(lm, false, g, wr, plW);
    for (const f of rightSide) list.push(f);
    for (const f of rightSide) {
      list.push({
        ...mirrorFeatherAcrossX(f, mx),
        react: mirroredHorizontalReact(wr),
      });
    }
  } else if (v12) {
    list.push(...buildSideWingFeathers(lm, false, g, wr, plW));
  } else if (v11) {
    list.push(...buildSideWingFeathers(lm, true, g, wr, plW));
  }

  list.sort((a, b) => a.layer - b.layer);
  return list;
}

/**
 * Upper chest: sample only the right half of the torso (u > 0.5), mirror to the left.
 * Uses the same phase for each mirrored pair so motion stays paired.
 */
function collectChestFeathers(lm, motion) {
  const Pc = torsoCenterProj(lm);
  if (!visibleEnough(lm[11]) || !visibleEnough(lm[12]) || !Pc) return [];
  const p11 = project(lm[11]);
  const p12 = project(lm[12]);
  const p23 = visibleEnough(lm[23], 0.12) ? project(lm[23]) : null;
  const p24 = visibleEnough(lm[24], 0.12) ? project(lm[24]) : null;
  let hipMid = { ...Pc };
  if (p23 && p24) {
    hipMid.x = (p23.x + p24.x) * 0.5;
    hipMid.y = (p23.y + p24.y) * 0.5;
  }
  const shoulderMid = { x: (p11.x + p12.x) * 0.5, y: (p11.y + p12.y) * 0.5 };
  const down = normalize2(hipMid.x - shoulderMid.x, hipMid.y - shoulderMid.y);
  const sw = shoulderWidthPx(lm);
  const neck = visibleEnough(lm[0], 0.1) ? project(lm[0]) : shoulderMid;
  const react = motion ? motion.chest : null;
  const reactMir = mirroredHorizontalReact(react);
  const mx = mirrorAxisX(lm);
  const list = [];
  const colsHalf = 7;
  const rows = 7;
  for (let row = 0; row < rows; row++) {
    const v = row / (rows - 0.65);
    if (v > 0.92) continue;
    for (let ic = 0; ic < colsHalf; ic++) {
      const u = 0.5 + ((ic + 0.5) / colsHalf) * 0.5;
      const along = {
        x: p11.x + (p12.x - p11.x) * u,
        y: p11.y + (p12.y - p11.y) * u,
      };
      const mixNeck = pow(1 - v, 1.25) * 0.28;
      const base = {
        x: along.x * (1 - mixNeck) + neck.x * mixNeck + down.x * sw * v * 0.32,
        y: along.y * (1 - mixNeck) + neck.y * mixNeck + down.y * sw * v * 0.32,
      };
      const outward = normalize2(base.x - Pc.x, base.y - Pc.y);
      const dir = normalize2(
        outward.x * 0.5 + down.x * 0.18,
        outward.y * 0.5 + down.y * 0.18 + 0.42
      );
      const len =
        sw *
        (0.14 + 0.22 * (1 - v * 0.82) + 0.07 * sin((u - 0.5) * TWO_PI * 2 + row)) *
        1.35;
      const phase = row * 3.7 + (u - 0.5) * 12 + ic * 0.4;
      const feather = {
        bx: base.x,
        by: base.y,
        dirx: dir.x,
        diry: dir.y,
        len,
        phase,
        layer: -2 + v * 0.08 + (u - 0.5) * 0.02,
        react,
        kind: "body",
      };
      list.push(feather);
      list.push({
        ...mirrorFeatherAcrossX(feather, mx),
        react: reactMir,
      });
    }
  }
  return list;
}

function collectArmFeathersForSide(lm, arm, plumage) {
  const restDroop = plumage && plumage.rest !== undefined ? plumage.rest : 0.45;
  const expand = plumage && plumage.expand !== undefined ? plumage.expand : 0.55;
  const rowScale = lerp(0.58, 1.22, expand);
  const lenScale = lerp(0.48, 1.28, expand);
  const out = [];
  const rowOffsets = [-1, -0.5, 0, 0.5, 1];
  const segments = [
    { pair: [arm.sh, arm.el], steps: 12 },
    { pair: [arm.el, arm.wr], steps: 15 },
  ];
  for (const seg of segments) {
    const [iA, iB] = seg.pair;
    const steps = seg.steps;
    const frame = armSegmentFrame(lm, iA, iB);
    if (!frame) continue;
    const { a, b, nx, ny, segLen } = frame;
    for (let si = 1; si < steps; si++) {
      const t = si / steps;
      const p = {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
      };
      for (let ri = 0; ri < rowOffsets.length; ri++) {
        const r = rowOffsets[ri];
        const bx = p.x + nx * segLen * 0.085 * r * rowScale;
        const by = p.y + ny * segLen * 0.085 * r * rowScale;
        const towX = b.x - p.x;
        const towY = b.y - p.y;
        const tow = normalize2(towX, towY);
        const openDir = normalize2(
          -nx * 0.42 + tow.x * 0.28,
          -ny * 0.42 + tow.y * 0.28 + lerp(0.38, 0.58, expand)
        );
        const droopDir = normalize2(
          tow.x * 0.12 - nx * 0.08,
          0.55 + tow.y * 0.18
        );
        const db = constrain(restDroop * (0.75 + t * 0.2), 0, 1);
        const dir = normalize2(
          lerp(openDir.x, droopDir.x, db),
          lerp(openDir.y, droopDir.y, db)
        );
        let len =
          segLen *
          (0.17 + 0.26 * (1 - abs(r) * 0.32)) *
          (0.88 + 0.12 * sin(t * PI)) *
          1.28 *
          lenScale;
        len *= lerp(0.88, 1.06, 1 - abs(r) * 0.35);
        const phase = si * 2.1 + ri * 1.4 + t * 6 + restDroop * 3.2;
        out.push({
          bx,
          by,
          dirx: dir.x,
          diry: dir.y,
          len,
          phase,
          layer: -1.5 + t * 0.05 + ri * 0.001,
          react: arm.react,
          kind: "body",
          plumeRest: restDroop,
        });
      }
    }
  }
  return out;
}

/** Each arm’s cover feathers follow that arm’s pose and velocity (no mirroring). */
function collectArmCoverFeathers(lm, motion, plumage) {
  const list = [];

  if (visibleEnough(lm[13], 0.12)) {
    list.push(
      ...collectArmFeathersForSide(
        lm,
        {
          sh: 11,
          el: 13,
          wr: 15,
          react: motion ? motion.left : null,
        },
        plumage.left
      )
    );
  }
  if (visibleEnough(lm[14], 0.12)) {
    list.push(
      ...collectArmFeathersForSide(
        lm,
        {
          sh: 12,
          el: 14,
          wr: 16,
          react: motion ? motion.right : null,
        },
        plumage.right
      )
    );
  }

  return list;
}

/**
 * Small plumes along each finger and at the wrist so the backs of the hands read feathered.
 */
function collectHandCoverFeathers(lm, motion, plumage) {
  const list = [];
  const Pc = torsoCenterProj(lm);
  if (!Pc) return list;
  const sw = shoulderWidthPx(lm);
  const expandL = plumage?.left?.expand ?? 0.55;
  const expandR = plumage?.right?.expand ?? 0.55;
  const restL = plumage?.left?.rest ?? 0.45;
  const restR = plumage?.right?.rest ?? 0.45;

  function addHandFeather(bx, by, dirx, diry, len, phase, layer, react, restBlend) {
    list.push({
      bx,
      by,
      dirx,
      diry,
      len,
      phase,
      layer,
      react,
      kind: "hand",
      plumeRest: restBlend * 0.55,
    });
  }

  function side(isRight) {
    const wrI = isRight ? 16 : 15;
    const tips = isRight ? HAND_RIGHT_TIPS : HAND_LEFT_TIPS;
    const react = isRight ? motion?.right : motion?.left;
    const expand = isRight ? expandR : expandL;
    const restD = isRight ? restR : restL;
    if (!visibleEnough(lm[wrI], 0.08)) return;

    const W = project(lm[wrI]);
    const tipPts = [];
    for (const t of tips) {
      if (visibleEnough(lm[t], 0.06)) tipPts.push(project(lm[t]));
    }
    if (tipPts.length === 0) return;

    let cx = W.x;
    let cy = W.y;
    for (const p of tipPts) {
      cx += p.x;
      cy += p.y;
    }
    cx /= 1 + tipPts.length;
    cy /= 1 + tipPts.length;

    const lenScale = lerp(0.62, 1.12, expand);

    const tipDirs = [];
    for (const T of tipPts) {
      tipDirs.push(normalize2(T.x - W.x, T.y - W.y));
    }

    for (let h = 0; h < 9; h++) {
      const u = h / 8;
      const idx = constrain(floor(u * (tipDirs.length - 0.001)), 0, tipDirs.length - 1);
      const idx2 = min(idx + 1, tipDirs.length - 1);
      const w = u * (tipDirs.length - 1) - idx;
      const d0 = tipDirs[idx];
      const d1 = tipDirs[idx2];
      const d = normalize2(lerp(d0.x, d1.x, w), lerp(d0.y, d1.y, w));
      const perpX = -d.y;
      const perpY = d.x;
      const outSign = (W.x - Pc.x) * perpX + (W.y - Pc.y) * perpY >= 0 ? 1 : -1;
      const dir = normalize2(d.x * 0.32 + perpX * outSign * 0.94, d.y * 0.32 + perpY * outSign * 0.94);
      const bx = W.x + d.x * sw * 0.028;
      const by = W.y + d.y * sw * 0.028;
      const flen = sw * (0.042 + 0.018 * sin(h * 0.7)) * lenScale;
      addHandFeather(bx, by, dir.x, dir.y, flen, h * 2.8 + 1.2, 2.35 + h * 0.02, react, restD);
    }

    for (const T of tipPts) {
      const dx = T.x - W.x;
      const dy = T.y - W.y;
      const Lseg = Math.hypot(dx, dy) || 1;
      const ux = dx / Lseg;
      const uy = dy / Lseg;
      const px = -uy;
      const py = ux;
      const outSign =
        (W.x + dx * 0.5 - Pc.x) * px + (W.y + dy * 0.5 - Pc.y) * py >= 0 ? 1 : -1;
      const steps = 9;
      for (let s = 1; s < steps; s++) {
        const t = s / steps;
        if (t > 0.86) continue;
        const bx = W.x + dx * t;
        const by = W.y + dy * t;
        const dir = normalize2(ux * 0.26 + px * outSign * 0.95, uy * 0.26 + py * outSign * 0.95);
        const flen =
          sw * (0.032 + 0.048 * sin(t * PI)) * (0.5 + t * 0.75) * lenScale;
        addHandFeather(
          bx,
          by,
          dir.x,
          dir.y,
          flen,
          s * 2.6 + t * 9,
          2.45 + t * 0.12,
          react,
          restD
        );
      }
    }
  }

  side(false);
  side(true);
  return list;
}

function featherGeometry(bx, by, dirx, diry, L, phase, react, plumeRest, kind) {
  const knd = kind || "body";
  const wingish = knd === "wing" || knd === "wingPrimary" || knd === "covert";
  const flexM = wingish ? 1.36 : 1;
  const rest = plumeRest === undefined ? 0 : constrain(plumeRest, 0, 1);
  const restEase = 1 - rest * (wingish ? 0.5 : 0.58);
  const flexBlend = (0.2 + 0.8 * restEase) * flexM;
  const spdRaw = react ? react.speed : 0;
  const spd = constrain(spdRaw * 0.0118, 0, 3.45);
  const rvMul = wingish ? 0.0108 : 0.0096;
  const rvx = react ? react.vx * rvMul : 0;
  const rvy = react ? react.vy * rvMul : 0;
  const tx = bx + dirx * L;
  const ty = by + diry * L;
  const px = -diry;
  const py = dirx;
  const sway =
    sin(frameCount * (0.024 + spd * 0.013) + phase) *
    (0.038 + rest * 0.021 + (0.055 + spd * 0.062) * restEase) *
    L *
    flexM;
  const swaySlow =
    sin(frameCount * (0.011 + spd * 0.007) + phase * 0.62) *
    L *
    (0.016 + spd * 0.032) *
    flexBlend;
  const gust =
    sin(frameCount * (0.08 + rest * 0.028) + phase * 1.4) *
    spd *
    0.064 *
    L *
    restEase *
    flexM;
  const bend = wingish ? 0.92 : 0.84;
  const midx =
    (bx + tx) * 0.5 +
    px * (sway + swaySlow * 0.9) +
    rvx * L * bend * restEase +
    gust * 0.078;
  const midy =
    (by + ty) * 0.5 +
    py * (sway + swaySlow * 0.9) +
    rvy * L * bend * restEase +
    gust * 0.078;
  const flexAmp = L * (0.028 + spd * 0.054) * flexBlend;
  const wob = sin(frameCount * 0.0165 + phase) * flexAmp;
  const wob2 = sin(frameCount * 0.012 + phase * 1.18 + 1.05) * flexAmp * 0.82;
  const midx1 = midx + px * wob;
  const midy1 = midy + py * wob;
  const midx2 = midx - px * wob2 * 0.74;
  const midy2 = midy - py * wob2 * 0.74;
  return { tx, ty, px, py, midx1, midy1, midx2, midy2, spd };
}

function vaneWidthForKind(kind) {
  if (kind === "wingPrimary") return 1.86;
  if (kind === "wing") return 1.68;
  if (kind === "covert") return 1.38;
  if (kind === "hand") return 1.28;
  return 1.52;
}

/** True if feather spine + approximate vane width hits the face shield (skip draw). */
function featherIntersectsFaceShield(f, bounds) {
  if (!bounds) return false;
  const kind = f.kind || "body";
  const L = f.len * 1.02;
  const geom = featherGeometry(
    f.bx,
    f.by,
    f.dirx,
    f.diry,
    L,
    f.phase,
    f.react,
    f.plumeRest,
    kind
  );
  const { tx, ty, px, py, midx1, midy1, midx2, midy2, spd } = geom;
  const wBase =
    kind === "wingPrimary" || kind === "wing" || kind === "covert"
      ? 0.265
      : kind === "hand"
        ? 0.24
        : 0.27;
  const wMax =
    L * wBase * vaneWidthForKind(kind) * (0.86 + 0.14 * sin(f.phase)) * (1 + spd * 0.17);

  for (let i = 0; i <= 12; i++) {
    const t = i / 12;
    const sx = bezierPoint(f.bx, midx1, midx2, tx, t);
    const sy = bezierPoint(f.by, midy1, midy2, ty, t);
    const wbarb = wMax * sin(t * PI) * (0.38 + 0.62 * t) * 1.1;
    if (pointInFaceRect(sx, sy, bounds)) return true;
    if (pointInFaceRect(sx + px * wbarb, sy + py * wbarb, bounds)) return true;
    if (pointInFaceRect(sx - px * wbarb, sy - py * wbarb, bounds)) return true;
  }
  return false;
}

/**
 * pass: 'fill' = soft vane body (readability); 'stroke' = rachis, barbs, barring, hooks.
 * speedGlow 0–1: warmer, brighter plumage when arms move fast.
 */
function drawFeatherPlume(bx, by, dirx, diry, L, phase, alpha, react, kind, pass, plumeRest, speedGlow) {
  const k = kind || "body";
  const sg = speedGlow === undefined ? 0 : constrain(speedGlow, 0, 1);
  const rest = plumeRest === undefined ? 0 : constrain(plumeRest, 0, 1);
  const restEase = 1 - rest * 0.55;
  const wingish = k === "wing" || k === "wingPrimary" || k === "covert";
  const warmW = wingish ? 1.14 : 1;
  const faceRgb = tintPlumageWithSpeed(VANE_FACE, sg, 1.05 * warmW);
  const bodyRgb = tintPlumageWithSpeed(VANE_BODY, sg, 0.92 * warmW);
  const rachisRgb = tintPlumageWithSpeed(RACHIS, sg, 0.62);
  const rachisHiRgb = tintPlumageWithSpeed(RACHIS_HIGHLIGHT, sg, 1.02 * warmW);
  const shadowRgb = tintPlumageWithSpeed(VANE_SHADOW, sg, 0.55);
  const barTipRgb = tintPlumageWithSpeed(BAR_TIP, sg, 0.78 * warmW);
  const rustRgb = tintPlumageWithSpeed(ACCENT_RUST, sg, 0.82 * warmW);
  const aBoost = 1 + sg * 0.26 * warmW;
  const { tx, ty, px, py, midx1, midy1, midx2, midy2, spd } = featherGeometry(
    bx,
    by,
    dirx,
    diry,
    L,
    phase,
    react,
    plumeRest,
    k
  );
  const wBase =
    k === "wingPrimary" || k === "wing" || k === "covert"
      ? 0.265
      : k === "hand"
        ? 0.24
        : 0.27;
  const wMax =
    L * wBase * vaneWidthForKind(k) * (0.86 + 0.14 * sin(phase)) * (1 + spd * 0.17);

  const outerLip = wingish ? 0.88 : k === "hand" ? 0.82 : 0.76;
  const innerLip = wingish ? 0.88 : k === "hand" ? 0.82 : 0.76;

  if (pass === "fill") {
    const steps = 18;
    noStroke();
    const faceA = wingish ? 0.62 : k === "hand" ? 0.6 : 0.58;
    fill(faceRgb[0], faceRgb[1], faceRgb[2], alpha * faceA * aBoost);
    beginShape();
    for (let i = 0; i <= steps; i++) {
      const s = i / steps;
      const sx = bezierPoint(bx, midx1, midx2, tx, s);
      const sy = bezierPoint(by, midy1, midy2, ty, s);
      const w = wMax * sin(s * PI) * (0.32 + 0.68 * s);
      vertex(sx + px * w, sy + py * w);
    }
    for (let i = steps; i >= 0; i--) {
      const s = i / steps;
      const sx = bezierPoint(bx, midx1, midx2, tx, s);
      const sy = bezierPoint(by, midy1, midy2, ty, s);
      const w = wMax * sin(s * PI) * (0.32 + 0.68 * s) * outerLip;
      vertex(sx - px * w, sy - py * w);
    }
    endShape(CLOSE);

    const bodyA = wingish ? 0.34 : k === "hand" ? 0.36 : 0.42;
    fill(bodyRgb[0], bodyRgb[1], bodyRgb[2], alpha * bodyA * aBoost);
    beginShape();
    const brownNarrow = wingish ? 0.5 : k === "hand" ? 0.55 : 0.68;
    for (let i = 0; i <= steps; i++) {
      const s = i / steps;
      const sx = bezierPoint(bx, midx1, midx2, tx, s);
      const sy = bezierPoint(by, midy1, midy2, ty, s);
      const w = wMax * sin(s * PI) * (0.2 + 0.52 * s) * brownNarrow;
      vertex(sx + px * w, sy + py * w);
    }
    for (let i = steps; i >= 0; i--) {
      const s = i / steps;
      const sx = bezierPoint(bx, midx1, midx2, tx, s);
      const sy = bezierPoint(by, midy1, midy2, ty, s);
      const w = wMax * sin(s * PI) * (0.2 + 0.52 * s) * brownNarrow * innerLip;
      vertex(sx - px * w, sy - py * w);
    }
    endShape(CLOSE);
    return;
  }

  const rW =
    k === "wingPrimary"
      ? max(1.95, L * 0.042)
      : k === "wing"
        ? max(1.78, L * 0.038)
        : k === "covert"
          ? max(1.38, L * 0.03)
          : max(1.9, L * 0.038);
  noFill();
  stroke(rachisRgb[0], rachisRgb[1], rachisRgb[2], alpha * (wingish ? 0.88 : 1) * (1 + sg * 0.18));
  strokeWeight(rW);
  strokeCap(ROUND);
  bezier(bx, by, midx1, midy1, midx2, midy2, tx, ty);
  stroke(
    rachisHiRgb[0],
    rachisHiRgb[1],
    rachisHiRgb[2],
    alpha * (wingish ? 0.32 : 0.42) * (1 + sg * 0.42)
  );
  strokeWeight(max(0.65, rW * 0.2));
  bezier(bx, by, midx1, midy1, midx2, midy2, tx, ty);

  const steps = 14;
  const barbTipStart = k === "wingPrimary" ? 0.62 : k === "wing" ? 0.64 : 0.72;
  const tipFlare =
    k === "wingPrimary" ? 0.72 : k === "wing" ? 0.62 : k === "covert" ? 0.5 : 0.55;

  for (let j = 1; j < steps; j++) {
    const s = j / steps;
    if (s < 0.08 || s > 0.97) continue;
    const sx = bezierPoint(bx, midx1, midx2, tx, s);
    const sy = bezierPoint(by, midy1, midy2, ty, s);
    const tipWide = s > barbTipStart ? 1.0 + (s - barbTipStart) * tipFlare : 1;
    const w = wMax * sin(s * PI) * (0.4 + 0.6 * s) * tipWide;
    const flutter =
      sin(frameCount * (0.037 + spd * 0.017) + phase + s * 4.1) *
      (0.056 + spd * 0.078) *
      w *
      restEase;

    stroke(faceRgb[0], faceRgb[1], faceRgb[2], alpha * 0.72 * aBoost);
    strokeWeight(
      wingish ? max(0.48, L * 0.015 * (1 - s * 0.26)) : max(0.85, L * 0.024 * (1 - s * 0.28))
    );
    line(sx, sy, sx + px * (w + flutter), sy + py * (w + flutter));
    stroke(bodyRgb[0], bodyRgb[1], bodyRgb[2], alpha * (wingish ? 0.42 : 0.55) * aBoost);
    line(sx, sy, sx - px * (w - flutter), sy - py * (w - flutter));

    if (s > 0.58) {
      stroke(shadowRgb[0], shadowRgb[1], shadowRgb[2], alpha * (wingish ? 0.38 : 0.5) * (1 + sg * 0.08));
      strokeWeight(wingish ? max(0.42, L * 0.011) : max(0.6, L * 0.016));
      line(sx, sy, sx - px * w * 0.48, sy - py * w * 0.48);
      line(sx, sy, sx + px * w * 0.48, sy + py * w * 0.48);
    }

    if (s > barbTipStart + 0.04) {
      stroke(
        barTipRgb[0],
        barTipRgb[1],
        barTipRgb[2],
        alpha * (wingish ? 0.35 : 0.5 + (s - barbTipStart) * 1.1) * (1 + sg * 0.12)
      );
      strokeWeight(wingish ? max(0.38, L * 0.009) : max(0.5, L * 0.012));
      const bite = w * (0.12 + (s - barbTipStart) * 0.35);
      line(sx - px * 0.1, sy - py * 0.1, sx + px * bite, sy + py * bite);
      line(sx + px * 0.1, sy + py * 0.1, sx - px * bite, sy - py * bite);
    }
  }

  stroke(rustRgb[0], rustRgb[1], rustRgb[2], alpha * (wingish ? 0.38 : 0.52) * (1 + sg * 0.1));
  strokeWeight(wingish ? 0.52 : 0.75);
  for (let j = 2; j < steps - 1; j++) {
    const s = j / steps + 0.015;
    if (s > 0.86) continue;
    const sx = bezierPoint(bx, midx1, midx2, tx, s);
    const sy = bezierPoint(by, midy1, midy2, ty, s);
    const wr = wMax * sin(s * PI) * (wingish ? 0.16 : 0.22);
    line(sx, sy, sx + px * wr, sy + py * wr);
    line(sx, sy, sx - px * wr, sy - py * wr);
  }

  if (k === "wingPrimary" || k === "wing" || k === "covert") {
    const ep = k === "wingPrimary" ? 0.945 : 0.96;
    const sx = bezierPoint(bx, midx1, midx2, tx, ep);
    const sy = bezierPoint(by, midy1, midy2, ty, ep);
    const tdx = tx - sx;
    const tdy = ty - sy;
    const tL = Math.hypot(tdx, tdy) || 1;
    const ux = tdx / tL;
    const uy = tdy / tL;
    stroke(barTipRgb[0], barTipRgb[1], barTipRgb[2], alpha * 0.78 * (1 + sg * 0.15));
    strokeWeight(max(0.95, L * (k === "wingPrimary" ? 0.024 : 0.021)));
    const h = L * (k === "wingPrimary" ? 0.12 : 0.095);
    const spread = k === "wingPrimary" ? 0.58 : 0.48;
    line(sx, sy, sx + ux * h + px * h * spread, sy + uy * h + py * h * spread);
    line(sx, sy, sx + ux * h - px * h * spread, sy + uy * h - py * h * spread);
    if (k === "wingPrimary" || k === "wing") {
      stroke(rustRgb[0], rustRgb[1], rustRgb[2], alpha * 0.62 * (1 + sg * 0.12));
      strokeWeight(max(0.72, L * 0.016));
      const h2 = h * 0.72;
      line(sx, sy, sx + ux * h2 + px * h2 * 0.32, sy + uy * h2 + py * h2 * 0.32);
      line(sx, sy, sx + ux * h2 - px * h2 * 0.32, sy + uy * h2 - py * h2 * 0.32);
    }
  }
}

function drawFeatherList(feathers, faceBounds, speedGlow) {
  const sg = speedGlow === undefined ? 0 : speedGlow;
  if (feathers.length === 0) return;

  push();
  blendMode(BLEND);
  for (const f of feathers) {
    if (faceBounds && featherIntersectsFaceShield(f, faceBounds)) continue;
    const kind = f.kind || "body";
    drawFeatherPlume(
      f.bx,
      f.by,
      f.dirx,
      f.diry,
      f.len * 1.02,
      f.phase,
      235,
      f.react,
      kind,
      "fill",
      f.plumeRest,
      sg
    );
  }
  pop();

  push();
  blendMode(BLEND);
  for (const f of feathers) {
    if (faceBounds && featherIntersectsFaceShield(f, faceBounds)) continue;
    const kind = f.kind || "body";
    drawFeatherPlume(
      f.bx,
      f.by,
      f.dirx,
      f.diry,
      f.len * 1.02,
      f.phase,
      255,
      f.react,
      kind,
      "stroke",
      f.plumeRest,
      sg
    );
  }
  pop();
}

function drawBackWings(lm, motion, faceBounds, plumage, speedGlow) {
  drawFeatherList(collectBackWingFeathers(lm, motion, plumage), faceBounds, speedGlow);
}

function drawBodyFeathers(lm, motion, faceBounds, plumage, speedGlow) {
  const chest = collectChestFeathers(lm, motion);
  const arms = collectArmCoverFeathers(lm, motion, plumage);
  const hands = collectHandCoverFeathers(lm, motion, plumage);
  const all = chest.concat(arms).concat(hands);
  all.sort((a, b) => a.layer - b.layer);
  drawFeatherList(all, faceBounds, speedGlow);
}

function drawSegment(landmarks, i, j) {
  const a = landmarks[i];
  const b = landmarks[j];
  if (!visibleEnough(a) || !visibleEnough(b)) return;
  const pa = project(a);
  const pb = project(b);
  line(pa.x, pa.y, pb.x, pb.y);
}

/** Torso + arms to wrist only (no finger fan). Drawn under hand-feathers, rings, and the foreground hand pass. */
function drawSkeletonBody(landmarks) {
  push();
  stroke(200, 230, 255, inputMode === "image" ? 38 : 28);
  strokeWeight(2);
  strokeCap(ROUND);

  for (const [i, j] of TORSO) drawSegment(landmarks, i, j);
  for (const [i, j] of ARMS_LEFT) drawSegment(landmarks, i, j);
  for (const [i, j] of ARMS_RIGHT) drawSegment(landmarks, i, j);

  pop();
}

/** Long glossy nail past the fingertip (local +X = toward free edge). */
function drawBigFingernail(W, T) {
  const ang = atan2(T.y - W.y, T.x - W.x);
  const nailLen = inputMode === "image" ? 30 : 26;
  const nailW = inputMode === "image" ? 17 : 15;
  push();
  translate(T.x, T.y);
  rotate(ang);
  noStroke();
  fill(255, 232, 238, inputMode === "image" ? 250 : 238);
  ellipse(nailLen * 0.42, 0, nailLen * 1.2, nailW);
  fill(255, 210, 218, inputMode === "image" ? 120 : 100);
  ellipse(nailLen * 0.28, 0, nailLen * 0.55, nailW * 0.5);
  fill(255, 255, 255, inputMode === "image" ? 115 : 95);
  ellipse(nailLen * 0.32, -nailW * 0.22, nailLen * 0.4, nailW * 0.35);
  stroke(235, 185, 198, inputMode === "image" ? 200 : 175);
  strokeWeight(1.05);
  noFill();
  ellipse(nailLen * 0.42, 0, nailLen * 1.2, nailW);
  stroke(255, 255, 255, inputMode === "image" ? 140 : 120);
  strokeWeight(0.55);
  arc(nailLen * 0.52, 0, nailLen * 0.5, nailW * 0.65, -PI * 0.35, PI * 0.35);
  pop();
}

/**
 * Hands on top of hand-feathers: soft palm, layered rays, large nails past the tips.
 */
function drawHandsForeground(landmarks) {
  if (!landmarks) return;
  const relaxed = 0.06;
  push();
  strokeCap(ROUND);
  strokeJoin(ROUND);

  function drawOneHand(isRight) {
    const wrI = isRight ? 16 : 15;
    const tips = isRight ? HAND_RIGHT_TIPS : HAND_LEFT_TIPS;
    const wrLm = landmarks[wrI];
    let W = null;
    if (wrLm && visibleEnough(wrLm, relaxed)) W = project(wrLm);

    const tipPts = [];
    for (const t of tips) {
      const L = landmarks[t];
      if (L && visibleEnough(L, relaxed)) tipPts.push(project(L));
    }

    if (!W && tipPts.length === 0) return;
    if (!W && tipPts.length > 0) {
      let sx = 0;
      let sy = 0;
      for (const p of tipPts) {
        sx += p.x;
        sy += p.y;
      }
      W = { x: sx / tipPts.length, y: sy / tipPts.length };
    }

    if (tipPts.length >= 2) {
      noStroke();
      fill(255, 250, 242, inputMode === "image" ? 85 : 68);
      beginShape();
      vertex(W.x, W.y);
      for (const p of tipPts) vertex(p.x, p.y);
      endShape(CLOSE);
      fill(255, 255, 255, inputMode === "image" ? 28 : 22);
      beginShape();
      vertex(W.x, W.y);
      for (const p of tipPts) vertex(p.x, p.y);
      endShape(CLOSE);
    }

    if (tipPts.length === 0 && W) {
      noStroke();
      fill(255, 250, 242, inputMode === "image" ? 70 : 55);
      circle(W.x, W.y, 22);
      stroke(210, 232, 255, inputMode === "image" ? 180 : 155);
      strokeWeight(1.2);
      noFill();
      circle(W.x, W.y, 22);
    }

    for (let i = 0; i < tips.length; i++) {
      const t = tips[i];
      const L = landmarks[t];
      if (!L || !visibleEnough(L, relaxed)) continue;
      const T = project(L);
      stroke(255, 255, 255, inputMode === "image" ? 50 : 38);
      strokeWeight(5.5);
      line(W.x, W.y, T.x, T.y);
      stroke(255, 252, 248, inputMode === "image" ? 120 : 100);
      strokeWeight(2.35);
      line(W.x, W.y, T.x, T.y);
      stroke(210, 232, 255, inputMode === "image" ? 215 : 190);
      strokeWeight(1.05);
      line(W.x, W.y, T.x, T.y);
      drawBigFingernail(W, T);
    }
  }

  drawOneHand(false);
  drawOneHand(true);
  pop();
}

/**
 * Pin to BlazePose eye center (2 = left, 5 = right); span = inner→outer in screen px for sizing.
 */
function eagleEyeSlot(lm, innerIdx, centerIdx, outerIdx) {
  if (!lm[centerIdx] || !visibleEnough(lm[centerIdx], 0.05)) {
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (const i of [innerIdx, centerIdx, outerIdx]) {
      if (!lm[i] || !visibleEnough(lm[i], 0.06)) continue;
      const p = project(lm[i]);
      sx += p.x;
      sy += p.y;
      n++;
    }
    if (n < 2) return null;
    const c = { x: sx / n, y: sy / n, span: 0 };
    if (lm[innerIdx] && lm[outerIdx] && visibleEnough(lm[innerIdx], 0.06) && visibleEnough(lm[outerIdx], 0.06)) {
      const a = project(lm[innerIdx]);
      const b = project(lm[outerIdx]);
      c.span = Math.hypot(b.x - a.x, b.y - a.y);
    }
    return c;
  }
  const c = project(lm[centerIdx]);
  let span = 0;
  if (lm[innerIdx] && lm[outerIdx] && visibleEnough(lm[innerIdx], 0.05) && visibleEnough(lm[outerIdx], 0.05)) {
    const a = project(lm[innerIdx]);
    const b = project(lm[outerIdx]);
    span = Math.hypot(b.x - a.x, b.y - a.y);
  }
  return { x: c.x, y: c.y, span };
}

/** Second day — eagle eyes: prefer Face Landmarker iris (468/473), else pose eyes (2/5). */
function drawEagleEyes(lm, faceResult) {
  const faceRaw = faceMeshEyeSlots(faceResult);
  const kPos = 0.56;
  const kSpan = 0.4;

  let L = null;
  let R = null;

  if (faceRaw && faceRaw.L) {
    smoothFaceEagleL = smoothFaceEagleSlot(smoothFaceEagleL, faceRaw.L, kPos, kSpan);
    L = smoothFaceEagleL;
  } else {
    smoothFaceEagleL = null;
    L = lm ? eagleEyeSlot(lm, 1, 2, 3) : null;
  }

  if (faceRaw && faceRaw.R) {
    smoothFaceEagleR = smoothFaceEagleSlot(smoothFaceEagleR, faceRaw.R, kPos, kSpan);
    R = smoothFaceEagleR;
  } else {
    smoothFaceEagleR = null;
    R = lm ? eagleEyeSlot(lm, 4, 5, 6) : null;
  }

  if (!L && !R) return;

  let ang = 0;
  let eyeDist = 130;
  if (L && R) {
    ang = atan2(R.y - L.y, R.x - L.x);
    eyeDist = Math.hypot(R.x - L.x, R.y - L.y);
  } else if (L && lm && lm[0] && visibleEnough(lm[0], 0.1)) {
    const n = project(lm[0]);
    ang = atan2(n.y - L.y, n.x - L.x);
    eyeDist = Math.hypot(n.x - L.x, n.y - L.y) * 1.6;
  } else if (R && lm && lm[0] && visibleEnough(lm[0], 0.1)) {
    const n = project(lm[0]);
    ang = atan2(n.y - R.y, n.x - R.x);
    eyeDist = Math.hypot(n.x - R.x, n.y - R.y) * 1.6;
  } else if (L && faceResult?.faceLandmarks?.[0]?.[168]) {
    const n = projectFacePoint(faceResult.faceLandmarks[0][168]);
    if (n) {
      ang = atan2(n.y - L.y, n.x - L.x);
      eyeDist = Math.hypot(n.x - L.x, n.y - L.y) * 1.5;
    }
  } else if (R && faceResult?.faceLandmarks?.[0]?.[168]) {
    const n = projectFacePoint(faceResult.faceLandmarks[0][168]);
    if (n) {
      ang = atan2(n.y - R.y, n.x - R.x);
      eyeDist = Math.hypot(n.x - R.x, n.y - R.y) * 1.5;
    }
  }

  const fallbackW = constrain(eyeDist * 0.52, 52, 200);
  const breathe = 1 + sin(frameCount * 0.04) * 0.035;

  function widthForSlot(slot) {
    if (slot.span > 8) return constrain(slot.span * 1.68, 56, 240);
    return fallbackW;
  }

  function drawOneEagleEye(slot, mirrorX) {
    const eyeW = widthForSlot(slot) * breathe;
    const eyeH = constrain(eyeW * 0.46, 22, 108);
    const cx = slot.x;
    const cy = slot.y;

    push();
    translate(cx, cy);
    rotate(ang);
    if (mirrorX) scale(-1, 1);

    noStroke();
    fill(12, 10, 8, inputMode === "image" ? 210 : 185);
    beginShape();
    const w = eyeW * 0.5;
    const h = eyeH;
    vertex(w * 0.9, 0);
    vertex(w * 0.15, -h * 1.05);
    vertex(-w * 0.72, -h * 0.28);
    vertex(-w * 0.98, 0);
    vertex(-w * 0.62, h * 0.62);
    vertex(w * 0.12, h * 0.82);
    endShape(CLOSE);

    fill(225, 165, 48, inputMode === "image" ? 235 : 215);
    beginShape();
    vertex(w * 0.72, 0);
    vertex(w * 0.08, -h * 0.82);
    vertex(-w * 0.58, -h * 0.18);
    vertex(-w * 0.78, 0);
    vertex(-w * 0.48, h * 0.48);
    vertex(w * 0.05, h * 0.62);
    endShape(CLOSE);

    fill(255, 214, 120, inputMode === "image" ? 120 : 100);
    ellipse(-w * 0.08, -h * 0.12, w * 0.85, h * 0.55);

    fill(18, 14, 22, 250);
    ellipse(0, 0, w * 0.34, h * 1.05);

    fill(8, 6, 12, 240);
    rectMode(CENTER);
    rect(0, 0, w * 0.09, h * 1.18);

    fill(255, 255, 255, inputMode === "image" ? 200 : 175);
    ellipse(-w * 0.22, -h * 0.32, w * 0.2, h * 0.22);

    stroke(40, 32, 24, inputMode === "image" ? 140 : 120);
    strokeWeight(1.1);
    noFill();
    beginShape();
    vertex(w * 0.9, 0);
    vertex(w * 0.15, -h * 1.05);
    vertex(-w * 0.72, -h * 0.28);
    vertex(-w * 0.98, 0);
    vertex(-w * 0.62, h * 0.62);
    vertex(w * 0.12, h * 0.82);
    endShape(CLOSE);
    pop();
  }

  push();
  blendMode(BLEND);
  if (L) drawOneEagleEye(L, false);
  if (R) drawOneEagleEye(R, true);
  pop();
}

function drawModeHint() {
  push();
  fill(255, 255, 255, 160);
  noStroke();
  textAlign(LEFT, BOTTOM);
  textSize(12);
  const src = inputMode === "image" ? "reference still" : "webcam";
  const snd = wingWind.enabled ? "wind on" : "wind off";
  text(`Source: ${src}  ·  I = image  ·  V = video  ·  Space = ${snd}`, 14, height - 14);
  fill(255, 255, 255, 115);
  textSize(11);
  text("Wingspan — move with me", 14, height - 30);
  fill(255, 255, 255, 88);
  textSize(9);
  text("Pose tracks arms and body for the wings.", 14, height - 44);
  text("Face mesh finds your irises so the eagle eyes sit on you.", 14, height - 56);
  pop();
}

function draw() {
  background(10, 12, 18);

  const ready =
    poseLandmarker &&
    ((inputMode === "image" && refImg && refImg.width > 0) ||
      (inputMode === "video" && capture.width > 0));

  if (!ready) {
    fill(255);
    noStroke();
    textAlign(CENTER, CENTER);
    text("Wingspan\nstarting camera, pose & face…", width / 2, height / 2);
    return;
  }

  drawInputCover();

  const now = performance.now();
  const faceResult = faceLandmarker ? faceLandmarker.detectForVideo(poseSourceElt(), now) : null;
  const result = poseLandmarker.detectForVideo(poseSourceElt(), now);
  const lm = result.landmarks && result.landmarks[0];
  let motionForSound = SILENT_ARM_MOTION;
  if (lm) {
    const motion = smoothMotionBundle(armMotionBundle(lm));
    motionForSound = motion;
    const peak = max(motion.left.speed, motion.right.speed);
    const rawGlow = constrain(pow(peak * 0.034, 0.72), 0, 1);
    const kGlow = rawGlow > smoothSpeedGlow ? 0.32 : 0.14;
    smoothSpeedGlow = lerp(smoothSpeedGlow, rawGlow, kGlow);
    const faceBounds = computeFaceShieldBounds(lm);
    const plumage = updateSmoothedPlumageSides(lm, motion);
    updateSmoothHandSpread(lm);
    drawBackWings(lm, motion, faceBounds, plumage, min(1, smoothSpeedGlow * 1.28));
    drawBodyFeathers(lm, motion, faceBounds, plumage, smoothSpeedGlow);
    drawHandFingerRings(lm);
    drawSkeletonBody(lm);
    drawHandsForeground(lm);
    prevProj = snapshotProj(lm);
  } else {
    prevProj = null;
    const d = 0.86;
    for (const key of ["left", "right", "chest", "both"]) {
      const c = smoothMotionReact[key];
      c.vx *= d;
      c.vy *= d;
      c.speed *= d;
    }
    stepSmoothWingReact(null);
    smoothWingPlumage.expand = lerp(smoothWingPlumage.expand, 0.55, 0.07);
    smoothWingPlumage.rest = lerp(smoothWingPlumage.rest, 0.45, 0.07);
    smoothSpeedGlow = lerp(smoothSpeedGlow, 0, 0.07);
    updateSmoothHandSpread(null);
  }

  drawEagleEyes(lm, faceResult);

  updateWingWindSound(motionForSound);
  drawModeHint();
}

