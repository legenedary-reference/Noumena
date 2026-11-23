import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

from noumena.eye import Eye, OnEvent, OnStream, Periodic, SensorSpec, SummaryModel
from noumena.lens import IMUAdapter, RawSample
from noumena.sensors import SimIMU

FAST = ["imu", "distance", "motion"]


def shm_names():
    try:
        return {n for n in os.listdir("/dev/shm") if n.startswith("nm")}
    except FileNotFoundError:  # macOS has no /dev/shm listing
        return set()


def test_end_to_end_process_model():
    before = shm_names()
    results = []
    eye = Eye([SensorSpec.simulated(m) for m in FAST], triggers=[Periodic(10)])
    with eye:
        stats = eye.run(seconds=8, max_results=5, on_result=lambda p, r, s: results.append((p, r, s)))
    assert stats.results >= 5
    p, r, s = results[-1]
    assert set(p.streams) == set(FAST)
    assert "Tensors:" in r and not s
    assert all(stats.frames[m] > 0 for m in FAST)
    assert shm_names() == before  # every ring unlinked


class ShapeModel:
    def setup(self):
        self.calls = 0

    def perceive(self, percept):
        self.calls += 1
        return {s: (t.shape, t.flags.writeable) for s, t in percept.tensors.items()}


def test_inline_model_gets_readonly_views():
    got = []
    eye = Eye([SensorSpec.simulated("imu")], model=ShapeModel, triggers=[OnStream("imu")], inline=True)
    with eye:
        eye.run(seconds=5, max_results=3, on_result=lambda p, r, s: got.append(r))
    assert got and got[-1]["imu"] == ((12,), False)


def test_events_trigger_dispatch():
    # The IMU simulator has a vibration burst at 1.3 s.
    got = []
    eye = Eye([SensorSpec.simulated("imu")], triggers=[OnEvent()], inline=True)
    with eye:
        eye.run(seconds=6, max_results=1, on_result=lambda p, r, s: got.append(p))
    assert got and got[0].reason == "event on imu" and got[0].events


def test_hardware_sync_keeps_sensors_in_step():
    specs = [SensorSpec.simulated(m, triggered=True) for m in ("imu", "motion")]
    eye = Eye(specs, triggers=[Periodic(5)], clock_hz=40, inline=True)
    with eye:
        stats = eye.run(seconds=1.5)
    a, b = stats.frames["imu"], stats.frames["motion"]
    assert a > 20 and abs(a - b) <= 2


@dataclass
class CrashingIMU(SimIMU):
    crash_after: int = 10

    def read(self):
        if self._n >= self.crash_after:
            os._exit(3)
        return super().read()


def test_supervisor_restarts_crashed_sensor():
    spec = SensorSpec("imu", CrashingIMU, IMUAdapter, source_kwargs={"stream": "imu"})
    eye = Eye([spec], triggers=[Periodic(10)], inline=True, max_restarts=2, supervise_every=0.05)
    with eye:
        stats = eye.run(seconds=3)
    assert stats.restarts["imu"] == 2
    assert stats.frames["imu"] >= 30  # three lives of ten frames, ring kept counting


class FlakyModel:
    def perceive(self, percept):
        if percept.packet.seq == 1:
            os._exit(4)
        return "ok"


def test_supervisor_restarts_crashed_model():
    got = []
    eye = Eye([SensorSpec.simulated("imu")], model=FlakyModel, triggers=[Periodic(20)], supervise_every=0.05)
    with eye:
        stats = eye.run(seconds=10, max_results=3, on_result=lambda p, r, s: got.append(r))
    assert stats.restarts["model"] == 1
    assert got == ["ok"] * 3


def test_duplicate_sensor_names_rejected():
    with pytest.raises(ValueError):
        Eye([SensorSpec.simulated("imu"), SensorSpec.simulated("imu")])
