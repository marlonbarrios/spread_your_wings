# Wingspan

**Wingspan** is a browser-based interactive piece built with **p5.js** and **MediaPipe**. Your **body pose** drives layered wings and plumage; your **face** (when visible) anchors eagle eyes on your irises. **Web Audio** turns movement into wind and an operatic voice—no audio files, everything is synthesized in real time.

## Run it

The sketch loads **ES modules** from the CDN (`@mediapipe/tasks-vision`) and needs **camera access** for video mode. Browsers require a **local or HTTPS** origin (opening `index.html` as `file://` often blocks the camera and modules).

From this folder:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` (or use any static server you prefer).

**Audio:** Click the canvas or press any key once to unlock sound (browser policy). After that, wind can start on its own when you move or when nobody is in frame (see below).

---

## Interactive parts

### Input

| Mode | How |
|------|-----|
| **Webcam** | Press **V** — live video, horizontally mirrored so movement feels natural |
| **Still image** | Press **I** — uses `assets/reference-angel.png` as a fixed pose source |

The video or image sits in a **rounded “portal”** frame: warm color grade, soft vignette in the letterbox, and a glowing border that pulses with your movement. When your body is tracked, **animated tethers** link your torso to the feed—wings and video feel plugged together. Pose and face run on that source every frame (throttled slightly for performance).

### Visual layers (when a body is detected)

Your pose landmarks are mapped to screen space. Layers are drawn in this order:

1. **Back wings** — Large feather fans from each shoulder, mirrored across your torso. Spread, length, and droop follow **arm pose** (elbow height, wrist reach, inward/outward direction) and **arm speed** (motion blur / flutter on feathers).
2. **Chest feathers** — Smaller plumage on the upper torso (right half mirrored to the left).
3. **Hand rings** — Concentric rings at each wrist; ring spacing grows when fingers spread.
4. **Body skeleton** — Simple limb lines for torso and arms.
5. **Hands (foreground)** — Larger hand shapes and fingernails drawn on top.
6. **Eagle eyes** — If the **face landmarker** sees you, eyes sit on **iris centers** (landmarks 468 / 473). Otherwise eyes use pose face points as a fallback.

**Face shield:** Feathers are skipped where they would overlap a rectangle around your face (pose face points + face mesh when available), so wings do not cover your eyes.

**Motion glow:** Fast arm movement warms and brightens feather colors briefly.

When **no body** is detected, wings and skeleton are not drawn; the background and input image/video remain, with optional soft wind only (see Sound).

### Motion → visuals (summary)

| Your movement | Visual response |
|---------------|-----------------|
| Arms down / close | Wings rest, shorter, more folded |
| Arms out / up | Wings expand, longer primaries, wider span |
| Fast arm motion | Feather flutter, stronger speed glow |
| Finger spread | Wider gaps between wrist rings |
| Face visible | Eagle eyes track irises |

---

## Controls

| Key / action | Effect |
|--------------|--------|
| **Click** or **any key** | Unlock Web Audio (required once per session) |
| **V** | Webcam input (mirrored) |
| **I** | Reference still image |
| **Space** | Toggle wind on/off (when off, wind stays off until you press Space again) |
| **O** | Toggle operatic voice on/off |

On-screen hints at the bottom repeat source mode and wind/voice state.

### Wind behavior (no key needed after unlock)

- **Nobody in frame:** Soft ambient wind only (low, slow “breathing” rustle; no gusts).
- **Body visible + movement:** Wind follows **arm speed**, **wing spread**, and **flaps** (acceleration spikes). Starts automatically when you move; **Space** forces it off.
- **Time of day:** Wind timbre slowly shifts brighter/higher from morning (~6:00) toward evening (~22:30), based on your system clock.

### Operatic voice behavior (press **O**)

Only active when **O** is on and a body is present:

| Input | Sound response |
|-------|----------------|
| **Hand height** | Main pitch: hands low → lower notes, hands high → higher notes (relative to shoulders/hips) |
| **Hands touching** | Voice fades to **silent** (wrists + fingertips on both hands) |
| **Arm speed / wing spread** | Volume, brightness, vibrato; slight pitch and stereo pan from left vs right arm |
| **Wind** | Automatically ducks (gets quieter) while the voice is singing so the voice stays forward |

---

## How sound is produced

All sound uses the browser **Web Audio API** (`AudioContext`)—no samples or MP3s. A single shared context feeds the speakers through **dynamics compressors** on each bus.

### 1. Wind (filtered noise)

Two looping noise sources are mixed:

- **Body layer** — **Brown noise** (low, rumbling) through high-pass and low-pass filters → “air” texture.
- **Gust layer** — **White noise** through brighter filters → short whooshes on fast movement.

**Modulation (each frame):**

- **Gain** from smoothed “drive” (arm speeds, wing openness, flap acceleration).
- **Filter cutoff** from drive + time-of-day phase (morning = darker/lower, evening = brighter).
- **Stereo pan** from difference between left and right arm speed.

**No pose:** Only the body layer at low level; gusts off; gentle slow amplitude wobble.

### 2. Operatic voice (formant synthesis)

Designed to suggest a sung **“ah”**-like vowel without recordings:

1. **Oscillators** — Sawtooth (fundamental) + triangle at 2× frequency (overtone).
2. **Vibrato** — Low-frequency sine modulates pitch (depth increases when wings are open).
3. **Formants** — Three parallel **bandpass filters** (~740 Hz, ~1160 Hz, ~2820 Hz) shape the spectrum like vocal resonances.
4. **Output** — Mixed formants → stereo panner → gain → compressor → speakers.

**Modulation:**

- **Frequency** from hand height (main), with small offsets for arm asymmetry, motion, and wing pose.
- **Formant frequencies and Q** brighten with movement.
- **Master gain** from motion envelope and wing openness, multiplied by a **gate** that goes to zero when hands touch.

Wind and voice are independent toggles (**Space** / **O**) but share the same audio context; the voice bus reduces wind level while it is audible.

---

## Tech notes

- **Pose:** MediaPipe Pose Landmarker (lite float16), loaded at runtime.
- **Face:** MediaPipe Face Landmarker (float16) for dense landmarks, including iris centers for eye placement.
- **Drawing:** p5.js 2D canvas; `p5.sound` is included but **sound is implemented with native Web Audio** in `sketch.js`.
- **Libraries:** vendored `p5.min.js` and `p5.sound.min.js` under `libraries/`.

Models and WASM are fetched from Google’s model bucket and jsDelivr (`@mediapipe/tasks-vision@0.10.14`); a network connection is required on first load.

## Repository

**Wingspan** — source: [github.com/marlonbarrios/spread_your_wings](https://github.com/marlonbarrios/spread_your_wings)
