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

## Dependency versions

To be recorded at CP2 once a working combination is pinned.

## Model initialization measurements

To be recorded at CP2 (model load + inference latency on CPU).
