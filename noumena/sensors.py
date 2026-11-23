"""
Sensor sources. A source is anything with ``read() -> RawSample``.

The simulated sources here let the whole stack run on a laptop with no
hardware. Each one paces itself at its own rate on an absolute schedule (so
it does not drift), unless the eye runs it in triggered mode, where a
hardware clock tick decides when it captures instead.

Each simulation has a scripted moment that should register as an event, so
an end-to-end run has something to notice:

    camera    a second bright object enters at 1.0 s
    imu       a vibration burst from 1.3 s to 1.6 s
    audio     a 1 kHz tone from 1.6 s to 2.0 s
    distance  something approaches from 3.5 m to 0.8 m from 1.9 s on
    motion    a camera pan from 2.2 s, with one object moving against it
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .lens import RawSample


@dataclass
class Paced:
    """Mixin: sleep until the next tick of an absolute schedule."""

    rate: float = 30.0
    paced: bool = True
    _t0: int = field(default=0, repr=False)
    _n: int = field(default=0, repr=False)

    def _tick(self) -> int:
        now = time.monotonic_ns()
        if self._t0 == 0:
            self._t0 = now
        if self.paced:
            due = self._t0 + int(self._n * 1e9 / self.rate)
            if due > now:
                time.sleep((due - now) / 1e9)
                now = time.monotonic_ns()
        self._n += 1
        return now

    @property
    def elapsed(self) -> float:
        return self._n / self.rate


@dataclass
class SimCamera(Paced):
    height: int = 240
    width: int = 320
    stream: str = "camera"
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        self._yy, self._xx = yy, xx
        self._bg = (40 + 60 * xx / self.width).astype(np.float32)

    def read(self) -> RawSample:
        t = self._tick()
        s = self.elapsed
        img = self._bg * (1.0 + 0.02 * np.sin(s))  # slow lighting drift: ordinary, not an event
        if s >= 1.0:
            img[(self._xx - self.width * 0.8) ** 2 + (self._yy - self.height * 0.25) ** 2 < 30**2] = 250
        img += self._rng.normal(0, 2, img.shape)
        rgb = np.clip(np.stack([img, img * 0.9, img * 0.8], axis=-1), 0, 255).astype(np.uint8)
        return RawSample(self.stream, t, rgb)


@dataclass
class SimIMU(Paced):
    rate: float = 20.0  # bursts per second
    sample_rate: float = 200.0
    stream: str = "imu"
    seed: int = 1

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._k = 0

    def read(self) -> RawSample:
        t = self._tick()
        n = int(self.sample_rate / self.rate)
        ts = (self._k + np.arange(n)) / self.sample_rate
        self._k += n
        roll = 0.1 * np.sin(ts * 0.5)
        acc = np.stack([np.zeros(n), 9.80665 * np.sin(roll), 9.80665 * np.cos(roll)], axis=1)
        gyro = np.stack([0.05 * np.cos(ts * 0.5), np.zeros(n), np.zeros(n)], axis=1) + 0.01  # 0.01 rad/s bias
        acc += self._rng.normal(0, 0.02, acc.shape)
        gyro += self._rng.normal(0, 0.002, gyro.shape)
        burst = (ts >= 1.3) & (ts < 1.6)
        acc[burst] += self._rng.normal(0, 3.0, (burst.sum(), 3))
        return RawSample(self.stream, t, np.concatenate([acc, gyro], axis=1).astype(np.float32))


@dataclass
class SimAudio(Paced):
    rate: float = 10.0  # chunks per second
    sample_rate: int = 16_000
    stream: str = "audio"
    seed: int = 2

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._k = 0

    def read(self) -> RawSample:
        t = self._tick()
        n = int(self.sample_rate / self.rate)
        ts = (self._k + np.arange(n)) / self.sample_rate
        self._k += n
        x = self._rng.normal(0, 0.01, n)
        tone = (ts >= 1.6) & (ts < 2.0)
        x[tone] += 0.5 * np.sin(2 * np.pi * 1000 * ts[tone])
        return RawSample(self.stream, t, (np.clip(x, -1, 1) * 32767).astype(np.int16))


@dataclass
class SimDistance(Paced):
    rate: float = 15.0
    zones: tuple[int, int] = (8, 8)
    stream: str = "distance"
    seed: int = 3

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def read(self) -> RawSample:
        t = self._tick()
        s = self.elapsed
        r = np.full(self.zones, 3.5)
        r[0, :] = np.nan  # top row looks over the wall: no return
        if s >= 1.9:
            d = max(0.8, 3.5 - 2.0 * (s - 1.9))
            r[3:6, 2:5] = d
        r += self._rng.normal(0, 0.01, r.shape)
        return RawSample(self.stream, t, r.astype(np.float32))


@dataclass
class SimMotion(Paced):
    blocks: tuple[int, int] = (30, 40)
    stream: str = "motion"
    seed: int = 4

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def read(self) -> RawSample:
        t = self._tick()
        s = self.elapsed
        h, w = self.blocks
        v = self._rng.normal(0, 0.2, (h, w, 2)).astype(np.float32)
        if s >= 2.2:
            # The camera pans right, so every block moves, and one object also
            # moves left against the pan.
            v[..., 0] += 6.0
            v[h // 3 : h // 3 + 6, w // 2 : w // 2 + 6, 0] -= 10.0
        return RawSample(self.stream, t, v)


SIMULATORS = {
    "camera": SimCamera,
    "imu": SimIMU,
    "audio": SimAudio,
    "distance": SimDistance,
    "motion": SimMotion,
}
