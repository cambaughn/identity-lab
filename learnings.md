# Learnings

Captured at v0.1.0 (August 2026).

## Question

Can a local computer reliably recognize people?

## Answer

Yes.

## What surprised me

- Recognition was easier than tracking.
- Stabilization mattered more than the embedding model.
- Local inference was fast enough.
- Consent boundaries became one of the most important design questions.

## The evidence behind each

- **Recognition was easier than tracking.** The matcher (cosine similarity,
  top-k, threshold + margin) worked on the first live test — 0.8–0.9
  similarity frontal, 0.6–0.7 through occlusion and never-enrolled
  sunglasses. Meanwhile smooth box tracking took a dedicated checkpoint
  (optical flow, staleness compensation, coherence gates) and *still* has a
  documented fast-motion failure mode with Kalman/SORT listed as future work.
- **Stabilization mattered more than the embedding model.** Raw per-frame
  matches were accurate but flickered; the label became trustworthy only
  with per-track voting, switch hysteresis, and decay
  (`tracking/stabilizer.py`) — plain counters, no ML.
- **Local inference was fast enough.** buffalo_l on an M2 CPU: ~90 ms
  detection, ~50 ms embedding per face, model warm in ~1 s. No GPU, no
  cloud, and the preview never dropped below ~30 fps thanks to
  latest-frame-wins hand-offs and split stage rates.
- **Consent boundaries became one of the most important design questions.**
  What began as a privacy checklist turned into product architecture: unknown
  people are transient by design, identities exist only through a deliberate
  enrollment ritual, deletion is first-class (secure_delete + VACUUM), and
  the roadmap's continual-enrichment idea is defined primarily by its privacy
  boundary rather than its algorithm.
