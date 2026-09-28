# Identity Lab

A local desktop toolkit for experimenting with visual identity recognition
for embodied AI.

Identity Lab shows your Mac's webcam feed in an amber-on-black console-style
window, lets you **deliberately enroll** named people from ~20 quality-gated
face samples, and then recognizes them live — with temporal stabilization so
labels are calm and honest. Everything runs on your machine.

![Identity Lab recognizing an enrolled person, with live pipeline readouts
and recognition controls](docs/screenshot.png)

> **⚠️ Experimental research software — not a production biometric security
> system.** This is a laboratory for exploring recognition for a future
> embodied robot. Do not use it for access control, surveillance, or any
> consequential decision. **Use it only with the informed consent of every
> person whose face is enrolled or captured.** The source code is
> MIT-licensed, but the default face-recognition models are licensed for
> non-commercial research only — see [Licensing](#licensing).

## Features

- Live mirrored webcam preview with real camera-device selection by name
- Face detection with 106-point landmarks (toggleable)
- Guided, consent-based enrollment: pose checklist, ~20 time-spread samples,
  quality gates (single face, size, confidence, blur, embedding consistency)
- Persistent local identity storage (SQLite, versioned schema, documented
  embedding format)
- Live recognition: cosine similarity, top-k sample scoring, adjustable
  threshold and best-vs-second-best margin — `UNKNOWN` is always preferred
  over a forced guess
- Temporal stabilization: vote-based name promotion, switch hysteresis,
  decay, stable never-reused track ids
- Experimental optical-flow box smoothing (boxes move at preview rate)
- Identity management: list with thumbnails, rename, delete, reset-all —
  all reflected in live recognition instantly
- Diagnostics: FPS / tracking / detection / recognition rates and
  latencies, debug overlay, in-app event log

**Deliberately out of scope:** automatic enrollment of unknown people,
persistent tracking or profiling of strangers, background biometric
accumulation, voice identity, speech, cloud anything. Unknown people remain
transient `UNKNOWN` detections; no persistent identity exists without an
explicit enrollment action.

## Requirements

- macOS on Apple Silicon (developed on macOS 13 / M2; Intel untested)
- Python 3.11
- ~600 MB disk for the Python environment, plus ~280 MB model download
- Internet **once**, for the first model download — recognition runs fully
  locally afterward

## Setup

```bash
git clone https://github.com/cambaughn/identity-lab.git
cd identity-lab
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Launch

Run from the macOS **Terminal** app (camera permission is granted to the
hosting application — see troubleshooting below):

```bash
.venv/bin/python -m identity_lab
```

On first launch InsightFace downloads its model pack (~280 MB, one time) to
`~/.insightface/models/buffalo_l/`. The panel shows `MODEL LOADING` until
the model is warmed up (a few seconds when cached).

## Tests

No webcam or model download required:

```bash
.venv/bin/python -m pytest tests/
```

## Using it

**Enrollment:** START CAMERA → ENROLL PERSON → enter a name → follow the
pose checklist (forward / left / right / up / down, optional glasses).
Samples are captured over several seconds with plain-language feedback for
every acceptance or rejection. Cancel at any point stores nothing. Requires
exactly one visible face — enrollment holds while a second face is in view.

**Recognition:** enrolled people get a bright label (`CAM 0.72` — name plus
raw cosine similarity); everyone else reads `UNKNOWN`. Controls:

- **THRESHOLD** (default 0.40): minimum similarity to accept a match.
  Higher = stricter (more UNKNOWNs), lower = looser (more false accepts).
  Same-person scores typically run ~0.6–0.9; strangers well below 0.4.
- **MARGIN** (default 0.08): with 2+ identities enrolled, the best match
  must also beat the runner-up by this much, else `UNKNOWN` (ambiguity is
  never guessed away).
- **TOP-K**: how many of a person's best-matching enrollment samples are
  averaged into their score.
- **DEBUG INFO** shows per-face track id, detector score, best/second-best
  candidates, and the exact accept/reject/stabilizer reason.

Similarity is a cosine score, **not** a probability or confidence percent.

**Identity management:** MANAGE IDENTITIES lists everyone with thumbnail,
sample count, date, and model; rename, delete, or RESET ALL (with explicit
red confirmations — Enter always cancels). Changes apply to live
recognition immediately.

## What is stored — and what is not

Stored, locally only, in `~/Library/Application Support/IdentityLab/`:

- display names
- face embeddings (512 float32 numbers per sample — see
  [docs/storage.md](docs/storage.md) for the exact format)
- enrollment metadata (dates, model id, schema/enrollment versions)
- one small (~128 px) consented enrollment thumbnail per identity
- app settings (`settings.json` — thresholds, camera choice, window size)

**Never stored:** continuous video, continuous camera images, raw
enrollment footage, images or embeddings of un-enrolled people, cloud
telemetry of any kind. Nothing leaves the machine.

**Deleting all biometric data:** open MANAGE IDENTITIES → RESET ALL →
confirm. Deleted data is overwritten (SQLite `secure_delete`) and the
database file is compacted. For a scorched-earth alternative:

```bash
rm -rf "$HOME/Library/Application Support/IdentityLab"
```

The model cache (`~/.insightface/`) contains no personal data — only the
downloaded pretrained models — and may also be deleted freely.

## Camera-permission troubleshooting

macOS attributes camera access to the app that hosts the process — for a
terminal launch, that's Terminal itself.

- First run: macOS asks "Terminal would like to access the camera" → Allow.
  OpenCV does not wait for your answer, so the very first attempt may fail;
  just run the app again after allowing.
- No prompt / permanent `CAMERA ERROR`: System Settings → Privacy &
  Security → Camera → enable your terminal app, then quit and reopen it.
- `CAMERA OFFLINE` with no device listed: no camera detected (check
  Continuity Camera / USB connections). The INPUT selector lists devices by
  name and updates on hot-plug.

## Architecture

Camera → detection → optional optical-flow propagation → face embedding →
gallery matching → track association → temporal stabilization → UI
observation. Stages run at different rates on worker threads (preview
~30 fps; flow every frame; detection ~5–7 Hz; recognition adaptive ~2 Hz
once names are stable) with latest-frame-wins hand-offs throughout — the
full design is in [docs/architecture.md](docs/architecture.md), storage
details in [docs/storage.md](docs/storage.md), and environment/measurement
notes in [docs/environment.md](docs/environment.md).

## Current limitations

- Very fast head motion can outrun the optical-flow tracker: the box may
  coast, jump, and the name briefly re-confirm (Kalman/SORT-style
  prediction is documented future work).
- Enrollment pose prompts are guidance, not verified poses (measured to be
  sufficient in practice — see docs/environment.md).
- Motion-blurred faces can briefly read `UNKNOWN` during rapid movement.
- Single model pack (buffalo_l, CPU); no GPU/ANE acceleration.
- Multi-camera index mapping between Qt and OpenCV is positional
  (documented assumption in docs/environment.md).

## Licensing

**Identity Lab's original source code — everything authored in this
repository — is released under the [MIT License](LICENSE).** You may use,
copy, modify, redistribute, and incorporate it into other projects,
including commercially.

**The MIT grant covers only this repository's original code.** Third-party
software, models, model weights, datasets, and other assets retain their
own licenses and terms. An MIT license here cannot and does not relicense
any of them.

### ⚠️ The face-recognition models are NOT MIT — read this

- Identity Lab's default configuration **automatically downloads the
  InsightFace `buffalo_l` model pack on first run.** Per the InsightFace
  project, its pretrained models — explicitly including auto-downloaded
  ones — are available **for non-commercial research purposes only**
  (a restriction inherited from the models' training data terms).
  InsightFace's 2025 licensing update additionally directs commercial
  licensing inquiries for the open-sourced recognition pack (`buffalo_l`)
  to recognition-oss-pack@insightface.ai.
- This repository does **not** contain or redistribute those weights, and
  the MIT license on this code grants **no** rights to them of any kind.
  Cloning Identity Lab does not, by itself, grant anyone the right to
  deploy the default `buffalo_l` model commercially.
- **If you want to use Identity Lab commercially**, the application code
  does not restrict you — but you are responsible for either licensing the
  models from InsightFace or substituting models whose terms permit your
  intended use.

### Third-party components

None of the following are vendored into this repository; pip installs them
from their own sources under their own licenses:

- **InsightFace code** (pip package): MIT — "no limitation for both
  academic and commercial usage." Distinct from the model weights above.
- **ONNX Runtime:** MIT · **OpenCV / opencv-python:** Apache-2.0 ·
  **NumPy:** BSD-3-Clause · **PySide6 / Qt:** LGPL-3.0 (as dynamically
  loaded here that imposes nothing on this code; consult Qt's terms if you
  redistribute Qt itself)
- Transitive dependencies carry their own MIT / BSD / Apache-2.0 / MPL-2.0
  terms (scikit-image, SciPy, Pillow, requests, certifi, tqdm, protobuf,
  and friends).
