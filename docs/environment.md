# Environment

Inspected 2026-07-29 during planning.

## v0.1.0 release verification (2026-08-05)

Verified: full pytest suite **185/185**; package import; offscreen
application startup to MODEL READY (real inference worker, no webcam);
migration tests (v1→v2 upgrade in place); clean-room setup — fresh venv
built from the documented README commands, `pip install -r
requirements.txt`, full suite green (22 s), import OK. Repository privacy
audit: full git history contains only source/tests/docs (no databases,
images, models, venvs, settings, logs, or exports; largest blob ever 25 KB).

Exact versions at release: Python 3.11.14 (arm64), numpy 2.4.6,
opencv-python 5.0.0.93, insightface 1.0.1, onnxruntime 1.23.2, PySide6
6.11.1, pytest 9.1.1. Model buffalo_l; DB schema v2; app version 0.1.0.

## Machine

| Item | Value |
| --- | --- |
| macOS | 13.7.3 Ventura (build 22H417) |
| CPU | Apple M2 (arm64 / Apple Silicon) |
| Free disk | ~52 GB |

## Toolchain

| Item | Value |
| --- | --- |
| Project Python | `/opt/homebrew/bin/python3.11` → Python 3.11.14 (arm64) |
| Other Pythons present | python.org framework 3.11.4, /usr/local 3.9/3.10/3.11 (Intel-era installs) |
| Homebrew | 6.0.9 (`/opt/homebrew/bin/brew`) |
| Xcode | Xcode 15 at `/Applications/Xcode.app`, Apple clang 15.0.0 |
| cmake | `/usr/local/bin/cmake` (needed: `insightface` compiles a C++/Cython extension on install) |

## Virtual environment

Created with:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
```

A virtual environment is a private, disposable copy of Python inside the project
folder (`.venv/`). Packages installed into it cannot affect — or be affected by —
anything else on the Mac. It is gitignored and can be deleted and recreated at any
time.

## Dependency versions (verified working, CP2, 2026-07-30)

| Package | Version | Notes |
| --- | --- | --- |
| Python | 3.11.14 (arm64) | venv from `/opt/homebrew/bin/python3.11` |
| numpy | 2.4.6 | insightface 1.x is NumPy-2 compatible — no `<2` pin needed |
| opencv-python | 5.0.0.93 | see note below |
| insightface | 1.0.1 | prebuilt arm64 wheel — no source compilation needed |
| onnxruntime | 1.23.2 | CPUExecutionProvider (CoreML also available, unused) |
| PySide6 | 6.11.1 | |
| pytest | 9.1.1 | |

**OpenCV note:** the plan called for `opencv-python-headless`, but `insightface`
1.0.1 declares `opencv-python` (full) as a hard dependency, and installing both
corrupts the shared `cv2` package. Smallest fix, adopted: use `opencv-python`
only. On macOS its GUI backend is Cocoa (no bundled Qt), so the Qt-plugin
conflict the headless choice was guarding against does not arise.

## Model initialization measurements (CP2, Apple M2, CPU)

Model pack: **buffalo_l** — auto-downloaded (~275 MB, one time) to
`~/.insightface/models/buffalo_l/`. Loaded models: detection (`det_10g`),
recognition (`w600k_r50`, embedding dim **512**, float32), 2D/3D landmarks,
genderage.

| Measurement | Value |
| --- | --- |
| FaceAnalysis init (model cached) | ~0.3 s (7.6 s on first run including download) |
| Detection pass, 1280×720, det_size 640×640 | ~90 ms |
| Embedding pass, one aligned face | ~53 ms |

Implication: ~140–150 ms per frame with one face → ~6 inferences/s on CPU.
Live UI will run detection every Nth frame while video previews at full rate.
buffalo_l latency is acceptable; no need to drop to buffalo_s.

Installed `FaceAnalysis` API (confirmed from insightface 1.0.1, not old examples):
`FaceAnalysis(name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs)`;
`prepare(ctx_id, det_thresh=0.5, det_size=None)` — `det_size` must be passed
explicitly; `get(img, max_num=0, det_metric='default')`; faces expose
`normed_embedding`, `embedding_norm`, bbox/kps/det_score.

Reproduce measurements: `.venv/bin/python scripts/model_check.py`

## Webcam smoke test (CP3, 2026-07-30)

`scripts/smoke_test.py` passed: built-in camera at index 0 delivers 1920×1080
frames via AVFoundation; a real face was detected (score 0.89) with a 512-dim
embedding; camera released cleanly (green light confirmed off).

Findings:

- **Camera permission is granted to Terminal.app, not the Claude desktop app.**
  macOS attributes camera access to the hosting application. Requests from
  inside the Claude app crash (`Abort trap: 6`) without registering a TCC
  entry, so anything that touches the camera — smoke test, the live app —
  must be launched from Terminal. OpenCV also does not wait for the
  permission prompt: the very first run fails while macOS is still showing
  the dialog; clicking Allow and re-running succeeds.
- **First inference on a real face costs ~6.6 s** (one-time lazy init of the
  landmark/recognition ONNX sessions; a no-face image only exercises the
  detector). Steady-state is ~150 ms/frame as measured at CP2. The live app
  must do a warm-up inference during its loading state.
- Harmless `FutureWarning` from insightface's internal scikit-image usage
  (`SimilarityTransform.estimate` deprecation). Not our code; cosmetic.

## Live detection performance (CP5, 2026-07-31)

Observed in the running app on the M2 (1920×1080 capture, det_size 640×640,
CPU): preview ~30 FPS; inference latency ~130 ms average per pass
(detection ~90 ms + 106-point landmarks ~2–17 ms per face). Overlay boxes
update at roughly 7–8 Hz — stepped box motion during fast head movement is
expected until CP9's tracking layer interpolates between passes.

**Embeddings are intentionally disabled in the detection-only path**:
`VisionEngine.analyze(frame, with_embeddings=False)` skips the recognition
model (~50 ms/face, measured steady-state) because nothing consumes
embeddings before recognition lands (CP7+). Earlier full-pipeline latency
was 220–250 ms; recognition checkpoints will re-enable embeddings via the
`with_embeddings=True` flag and should expect roughly that cost again.
A further tuning knob, if needed later: det_size 640→480 roughly halves
detection cost at the expense of small/distant faces.

## Live recognition performance (CP8, 2026-08-03)

Observed with one enrolled identity (20 samples, CP7 enrollment without
pose verification), threshold 0.40, margin 0.08, top-k 3:

- straight ahead: similarity ~0.8–0.9
- covering mouth or eyes, turning to the side, sunglasses (never enrolled
  with sunglasses): ~0.6–0.7 — still comfortably recognized
- un-enrolled faces / phone photos: well below threshold → UNKNOWN

**Conclusion for the enrollment-quality question:** failures do NOT appear
correlated with the lack of pose verification during enrollment; the
embedding model is robust to modest pose/occlusion changes on top of
time-spread frontal-ish samples. Pose verification remains a nice-to-have
(see README known limitations), not a needed fix.

Latency with embeddings enabled on every pass: ~150–250 ms per inference
(vs ~130 ms detection-only), as expected.

## Temporal stabilization (CP9, 2026-08-03) — manually verified

Live verification passed: stable labels through normal and fast movement,
`HOLD n/6` through brief occlusion without losing the name, new
(never-reused, ascending) track id + re-promotion after leaving and
returning, steady stabilized label with sunglasses despite raw-similarity
variation. Parameters: promote after 3 agreeing observations, switch names
after 5 consecutive, decay to UNKNOWN after 6 consecutive misses, track
expiry 1.5 s (~5 inference updates).

## Smooth tracking pipeline (CP10.5, 2026-08-04)

Three rates: preview ~30 fps; **sparse Lucas–Kanade optical flow** moves
each track's displayed box on every preview frame (measured pixel motion on
a half-res grayscale image — median point translation + spread-based scale;
never interpolation); full detection corrects drift at the INFER INTERVAL
cadence (~5–7 Hz); embeddings/recognition run adaptively — every detection
pass while any visible track lacks a confirmed name, then relaxed to ~2/s.

Flow cost measured: median ~4.5 ms/frame at 1080p with two tracks
(synthetic worst-case texture) on the UI thread — inside the 33 ms budget.

Failure semantics: incoherent point motion (MAD gate), oversized single-
frame jumps, or too few valid points put the track in COAST (box frozen,
shown in debug) until the next detection re-anchors or the track expires.
Detection re-anchors are staleness-compensated (the detection is shifted by
the flow motion accumulated since its frame was submitted, via submit
tokens), deadbanded, gently blended when healthy, and snapped when
coasting or grossly disagreeing. Scale changes are damped and deadbanded
to prevent size "breathing". Flow points reseed on every anchor. Det-only
passes never count as recognition misses against a stabilized name.

**Outcome (manually verified 2026-08-04/05):** optical flow materially
improves box movement — described by the user as "10x better" in both size
stability and smoothness versus the detection-cadence boxes.

**Known limitation (accepted for v0.1):** sufficiently fast head motion
exceeds optical flow's capture range; the track can coast, fail IOU
association with the next detection, and be recreated — visible as a box
jump and a brief identity re-confirmation (PENDING → name). Understood and
deliberately not addressed in v0.1.

**Possible future work (documented, NOT implemented):** Kalman/SORT-style
constant-velocity prediction with association against predicted boxes;
ByteTrack-style low-confidence detection association to bridge motion
blur; embedding-based re-association of new tracks to recently dead
confirmed tracks (DeepSORT-style).

## Camera device mapping (CP5, 2026-07-30)

Camera names come from Qt (`QMediaDevices.videoInputs()`), which enumerates
AVFoundation devices with real names and stable unique ids without opening
them, and emits `videoInputsChanged` on hot-plug. OpenCV's AVFoundation
backend can only *open* a camera by numeric index — it has no device-id API.

**Mapping used:** a device's position in Qt's enumeration order is used as
the OpenCV capture index. Verified on this Mac (single camera: Qt position 0
= "FaceTime HD Camera" = the device OpenCV index 0 opened at CP3).

**Documented limitation:** with multiple cameras attached, Qt and OpenCV
both enumerate the same AVFoundation device list and are expected to agree
on order, but neither API guarantees it. If a selected external camera ever
opens the wrong device, this mapping is the place to look
(`identity_lab/camera/devices.py`). The saved selection is stored by unique
device id, so a disappeared device falls back to the system default rather
than silently opening whatever occupies its old index.
