# Spread Your Wings

A browser-based piece built with **p5.js** and **MediaPipe**: your **body pose** drives layered wings and plumage, and **face landmarks** (iris positions) anchor the eagle eyes when a face is visible, with a fallback to pose-based eye estimates.

## Run it

The sketch loads **ES modules** from the CDN (`@mediapipe/tasks-vision`) and needs **camera access**. Browsers require a **local or HTTPS** origin (opening `index.html` as `file://` often blocks the camera and modules).

From this folder:

```bash
# Python 3
python3 -m http.server 8080
```

Then open `http://localhost:8080` (or use any static server you prefer).

## Controls

| Key | Action |
|-----|--------|
| **V** | Webcam input (mirrored) |
| **I** | Reference still image (`assets/reference-angel.png`) |
| **Space** | Toggle wind / wing ambience (Web Audio) |

On-screen hints repeat the same shortcuts.

## Tech notes

- **Pose:** MediaPipe Pose Landmarker (lite float16 task), loaded at runtime.
- **Face:** MediaPipe Face Landmarker (float16 task) for dense landmarks, including iris centers used for eye placement.
- **Libraries:** vendored `p5.min.js` and `p5.sound.min.js` under `libraries/`.

Models and WASM are fetched from Google’s model bucket and jsDelivr (`@mediapipe/tasks-vision@0.10.14`); a network connection is required on first load.

## Repository

Source: [github.com/marlonbarrios/spread_your_wings](https://github.com/marlonbarrios/spread_your_wings)
