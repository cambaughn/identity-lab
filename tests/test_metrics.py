"""Diagnostics calculations — injectable clocks, fully deterministic."""

import time

from identity_lab.diagnostics.metrics import FpsCounter, StatusLog


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_fps_zero_without_enough_ticks():
    counter = FpsCounter(clock=FakeClock())
    assert counter.fps == 0.0
    counter.tick()
    assert counter.fps == 0.0


def test_fps_steady_rate():
    clock = FakeClock()
    counter = FpsCounter(window=30, clock=clock)
    for _ in range(30):
        counter.tick()
        clock.t += 1 / 30  # exactly 30 fps
    assert abs(counter.fps - 30.0) < 0.5


def test_fps_uses_rolling_window():
    clock = FakeClock()
    counter = FpsCounter(window=10, clock=clock)
    for _ in range(10):
        counter.tick()
        clock.t += 1.0  # slow: 1 fps
    for _ in range(10):
        counter.tick()
        clock.t += 0.1  # fast: 10 fps — old slow ticks must age out
    assert counter.fps > 5.0


def test_fps_reset():
    clock = FakeClock()
    counter = FpsCounter(clock=clock)
    counter.tick()
    clock.t += 0.1
    counter.tick()
    assert counter.fps > 0
    counter.reset()
    assert counter.fps == 0.0


def test_status_log_capacity_and_order():
    log = StatusLog(capacity=3)
    for i in range(5):
        log.add(f"EVENT {i}")
    events = log.events()
    assert len(events) == 3
    assert [m for _, m in events] == ["EVENT 2", "EVENT 3", "EVENT 4"]


def test_status_log_timestamp_format():
    fixed = time.struct_time((2026, 7, 30, 9, 5, 7, 0, 0, 0))
    log = StatusLog(clock=lambda: fixed)
    event = log.add("HELLO")
    assert event == ("09:05:07", "HELLO")
    assert StatusLog.format_event(event) == "09:05:07  HELLO"
