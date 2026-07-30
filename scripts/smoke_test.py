"""CP3 webcam smoke test: one real frame through the full pipeline — no UI.

Opens the webcam, lets exposure/autofocus settle, captures one frame, runs
InsightFace on it, prints the results, and releases the camera. The captured
frame is never written to disk. The camera release path runs even if
inference raises (try/finally).

Exit codes: 0 = ran and detected >=1 face, 2 = ran but no face detected
(not a dependency failure — retry by re-running), 1 = camera or model error.

Run:  .venv/bin/python scripts/smoke_test.py [camera_index]
"""

import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

WARMUP_SECONDS = 2.0   # let exposure and autofocus settle before the real capture
DETECT_ATTEMPTS = 3    # re-capture up to this many times if no face is found
ATTEMPT_GAP_SECONDS = 1.5

PERMISSION_HELP = """\
Could not read frames from the camera. Most likely causes:
  - The macOS camera permission was denied for the terminal app.
    Fix: System Settings -> Privacy & Security -> Camera -> enable it,
    then quit and reopen the terminal app and re-run this script.
  - Another app is using the camera, or the camera index is wrong
    (try: .venv/bin/python scripts/smoke_test.py 1).\
"""


def main() -> int:
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Initialize the model BEFORE touching the camera so the camera (and its
    # green light) is only active for the few seconds of actual capture.
    print("Initializing FaceAnalysis (buffalo_l, CPU)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model ready.")

    print(f"\nOpening camera index {cam_index} via AVFoundation...")
    print(">>> If macOS shows a camera-permission prompt, click Allow. <<<")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            print(f"\nERROR: camera index {cam_index} could not be opened.")
            print(PERMISSION_HELP)
            return 1

        # Warm-up: read and discard frames while the sensor settles.
        t_end = time.time() + WARMUP_SECONDS
        warm_ok = False
        while time.time() < t_end:
            ok, _ = cap.read()
            warm_ok = warm_ok or ok
        if not warm_ok:
            print("\nERROR: camera opened but returned no frames.")
            print(PERMISSION_HELP)
            return 1

        for attempt in range(1, DETECT_ATTEMPTS + 1):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("\nERROR: failed to capture a frame.")
                print(PERMISSION_HELP)
                return 1

            h, w = frame.shape[:2]
            print(f"\nAttempt {attempt}/{DETECT_ATTEMPTS}: captured frame "
                  f"{w}x{h} from camera index {cam_index}")

            t0 = time.perf_counter()
            faces = app.get(frame)
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"inference latency: {latency_ms:.0f}ms")
            print(f"faces detected: {len(faces)}")
            for i, face in enumerate(faces):
                emb = np.asarray(face.normed_embedding).ravel()
                print(f"  face {i}: detection score {face.det_score:.3f}, "
                      f"embedding dim {emb.shape[0]}")

            if faces:
                print("\nRESULT: PASS (camera + capture + detection + embedding)")
                return 0
            if attempt < DETECT_ATTEMPTS:
                print(f"No face found - re-capturing in {ATTEMPT_GAP_SECONDS}s "
                      f"(face the camera)...")
                time.sleep(ATTEMPT_GAP_SECONDS)

        print("\nRESULT: NO FACE DETECTED. Camera and model both worked; "
              "re-run the script while facing the camera to retry.")
        return 2
    finally:
        cap.release()
        print("Camera released.")


if __name__ == "__main__":
    sys.exit(main())
