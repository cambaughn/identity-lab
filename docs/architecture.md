# Identity Lab — architecture

## Layers

```
identity_lab/
  camera/       camera discovery (Qt device list) and frame capture (OpenCV)
  vision/       InsightFace wrapper: detection, landmarks, embeddings
  identity/     enrollment, storage (SQLite), matching, typed errors
  tracking/     short-lived IOU tracks, optical-flow box propagation,
                temporal identity stabilization
  ui/           PySide6 window, dialogs, overlay painting, theme tokens
  config/       persisted user settings (JSON)
  diagnostics/  FPS/rate counters, in-app status event log
```

The UI never owns model inference. Two worker threads do the heavy work:

- **CameraWorker** (one per camera session): owns `cv2.VideoCapture`, reads
  frames, publishes them through a latest-frame-wins holder (a slow consumer
  can never cause a backlog), releases the camera in a `finally` block on
  every exit path.
- **InferenceWorker** (one per app run): loads the InsightFace model at
  launch (with a warm-up pass so first inference is not slow), then analyzes
  submitted frames — also latest-frame-wins. Each submitted frame carries a
  per-frame "with embeddings" flag and an opaque token echoed back on the
  result.

## Perception pipeline and rates

Different stages deliberately run at different rates:

```
Camera frame (~30 fps)
  → mirrored preview display                      (every frame,   UI thread)
  → optical-flow box propagation                  (every frame,   UI thread)
  → [every Nth frame] submit to InferenceWorker   (latest wins)
        → face detection + landmarks              (~5–7 Hz, worker thread)
        → face embedding (adaptive: eager until   (~2 Hz steady,  worker)
          every visible track has a confirmed
          name, then relaxed)
  → on result (UI thread):
        → min-face-size filter
        → IOU track association (stable, never-reused track ids)
        → staleness-compensated flow re-anchor
        → gallery matching (cosine, top-k mean,
          threshold + second-best margin)          [embedding passes only]
        → temporal stabilization (vote promotion,
          switch hysteresis, decay)                [embedding passes only]
        → IdentityObservation → overlay labels + readouts
```

Key mechanisms:

- **Latest-frame-wins everywhere.** Neither display nor inference can queue
  up; expensive work operates on the newest available frame only.
- **Optical flow** (`tracking/flow.py`): sparse Lucas–Kanade on a half-res
  grayscale image moves each track's displayed box by measured pixel motion
  every preview frame. Incoherent or oversized motion puts the track in
  COAST (frozen box, no guessing) until the next detection re-anchors it.
- **Staleness compensation:** a detection describes a frame ~100–200 ms old.
  Results carry the submit token; the detection box is shifted by the flow
  motion accumulated since that submit before it corrects the track.
- **Matching** (`identity/matcher.py`): cosine similarity of unit-normalized
  embeddings; per-identity score = mean of the top-k best enrollment
  samples; a match must clear the threshold and beat the runner-up by a
  margin, else `UNKNOWN`. Similarity is a score, never a probability.
- **Stabilization** (`tracking/stabilizer.py`): per-track voting — a name
  appears after 3 agreeing observations, switches only after 5 consecutive
  disagreements in favor of another name, decays to UNKNOWN after 6 misses,
  and resets when the track dies.

## Storage

See [storage.md](storage.md): SQLite in the macOS app-data directory,
explicit versioned migrations, documented float32 little-endian embedding
blobs, typed exceptions, secure_delete + VACUUM hygiene.
