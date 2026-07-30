# Environment

Inspected 2026-07-29 during planning.

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
