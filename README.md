# Identity Lab

A local macOS proof-of-concept exploring visual identity recognition for a future
embodied robot. It shows the Mac webcam feed in a desktop window, lets you enroll
named people using face embeddings, and then recognizes them live.

> **⚠️ Experimental local recognition tool.** This is a private research prototype,
> not a production biometric system. Use it only with the informed consent of every
> person whose face is enrolled or captured. Everything runs locally on this machine:
> no images, embeddings, names, or telemetry are ever uploaded.

## Status

Under construction — being built checkpoint by checkpoint. This README will be
completed as the application takes shape.

## Environment

See [docs/environment.md](docs/environment.md) for the inspected machine details.

## Setup (so far)

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Launch

Run from the macOS **Terminal** app — camera permission is attributed to the
hosting application, and Terminal is the one that has it (see
[docs/environment.md](docs/environment.md)):

```bash
.venv/bin/python -m identity_lab
```

## Tests (no webcam required)

```bash
.venv/bin/python -m pytest tests/
```
