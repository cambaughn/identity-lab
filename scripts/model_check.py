"""CP2 model check: verify the full AI stack initializes and runs — no webcam.

Proves that every dependency imports, ONNX Runtime loads, InsightFace
FaceAnalysis initializes (downloading the model pack on first run), and one
inference pass runs on a synthetic image. Prints exact versions and latency
measurements for docs/environment.md.

Run:  .venv/bin/python scripts/model_check.py
"""

import platform
import sys
import time

import cv2
import insightface
import numpy as np
import onnxruntime
import PySide6
from insightface.app import FaceAnalysis

MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)


def main() -> int:
    print("=== versions ===")
    print(f"python       {platform.python_version()} ({platform.machine()})")
    print(f"numpy        {np.__version__}")
    print(f"opencv       {cv2.__version__}")
    print(f"insightface  {insightface.__version__}")
    print(f"onnxruntime  {onnxruntime.__version__}")
    print(f"pyside6      {PySide6.__version__}")
    print(f"ort providers available: {onnxruntime.get_available_providers()}")

    print(f"\n=== initializing FaceAnalysis ({MODEL_PACK}, CPU) ===")
    t0 = time.perf_counter()
    app = FaceAnalysis(name=MODEL_PACK, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    init_s = time.perf_counter() - t0
    print(f"init time: {init_s:.2f}s (includes model download on first run)")
    print(f"loaded models: {sorted(app.models.keys())}")

    # Detection pass on a synthetic image: proves the detector runs end-to-end.
    # A random image contains no face, so 0 detections is the expected result.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    faces = app.get(img)
    warmup_ms = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        faces = app.get(img)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\n=== detection pass (synthetic 1280x720, no face expected) ===")
    print(f"faces detected: {len(faces)} (expected 0)")
    print(f"latency: warmup {warmup_ms:.0f}ms, then {min(times):.0f}-{max(times):.0f}ms over 5 runs")

    # Embedding model direct pass: proves the recognition model runs and
    # reports the embedding dimension without needing a real face.
    rec = app.models["recognition"]
    aligned = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    feat = rec.get_feat(aligned)
    rec_ms = (time.perf_counter() - t0) * 1000
    emb = np.asarray(feat).ravel()
    print(f"\n=== recognition model direct pass (synthetic aligned 112x112) ===")
    print(f"model: {getattr(rec, 'model_file', '?')}")
    print(f"embedding dim: {emb.shape[0]}, dtype: {emb.dtype}")
    print(f"embedding latency: {rec_ms:.0f}ms")

    ok = emb.shape[0] == 512 and len(faces) == 0
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
