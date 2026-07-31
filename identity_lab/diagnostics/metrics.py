"""Timing counters and the status event log.

Diagnostics never contain images, embeddings, or video — text events and
numbers only. Nothing here persists to disk.
"""

import time
from collections import deque


class FpsCounter:
    """Rolling frames-per-second over a recent window of tick timestamps."""

    def __init__(self, window: int = 30, clock=time.monotonic) -> None:
        self._clock = clock
        self._ticks: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._ticks.append(self._clock())

    def reset(self) -> None:
        self._ticks.clear()

    @property
    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return 0.0
        return (len(self._ticks) - 1) / span


class StatusLog:
    """Bounded in-memory event log: (HH:MM:SS, message) pairs, newest last."""

    def __init__(self, capacity: int = 200, clock=time.localtime) -> None:
        self._clock = clock
        self._events: deque[tuple[str, str]] = deque(maxlen=capacity)

    def add(self, message: str) -> tuple[str, str]:
        t = self._clock()
        stamp = f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
        event = (stamp, message)
        self._events.append(event)
        return event

    def events(self) -> list[tuple[str, str]]:
        return list(self._events)

    @staticmethod
    def format_event(event: tuple[str, str]) -> str:
        return f"{event[0]}  {event[1]}"
