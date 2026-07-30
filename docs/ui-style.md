# Approved UI direction — vintage industrial IMAX console

Recorded 2026-07-30, before CP2. Applies to the UI checkpoints (CP4 onward).
Reference: photo of an IMAX projection console — an old but highly reliable
industrial control panel with amber-on-black display.

## Desired qualities

- Predominantly black background
- Monochrome amber/yellow foreground
- Thin amber borders, separators, and outlined controls
- Dense but orderly instrument-panel composition
- Mostly square corners or very small corner radii
- Blocky terminal, bitmap-like, or industrial display typography
- Uppercase operational labels where appropriate
- Compact status readouts and grouped control regions
- Restrained amber glow and mild CRT softness
- Clear operational states, e.g.:
  - `CAMERA OFFLINE`
  - `CAMERA READY`
  - `MODEL LOADING`
  - `SYSTEM READY`
  - `FACE DETECTED`
  - `IDENTITY: UNKNOWN`
  - `TRACK 003`
  - `ENROLLMENT 12 / 20`

## Avoid (this is not generic neon cyberpunk)

- Blue or purple neon
- Glassmorphism
- Large rounded modern cards
- Gradients used as decoration
- Excessive animation
- Fake malfunction effects
- Strong scanlines, distortion, flicker, or noise that hurts readability
- Sacrificing usability merely to imitate an old display

The result should feel like an old professional machine that remains dependable
and legible. Modern interaction behavior and accessibility underneath the visual
treatment.

## Implementation notes for the UI checkpoints

Centralize visual tokens (single module, e.g. `identity_lab/ui/theme.py`) so the
appearance can be adjusted consistently. At minimum, shared values for:

- background
- primary amber
- dim amber
- warning/error state
- border thickness
- spacing
- corner radius
- typography
- glow intensity

Font: use an available system or redistributable font — no proprietary font file
dependency. Prefer a practical monospace fallback stack (e.g. Menlo / SF Mono /
Courier-style) if an ideal bitmap-style font is unavailable.

Warning and error states may use a second restrained console-style color when
necessary rather than ambiguous amber text. The informed-consent notice, camera
errors, and destructive confirmations must remain clearly legible with
sufficient contrast.
