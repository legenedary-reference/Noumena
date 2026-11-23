"""
The a priori forms.

Kant's move was that space and time are not things we find in the world but
the shape any experience has to take before it can be an experience at all,
and that categories like substance and cause are how the understanding binds
those experiences into objects and events. Noumena takes that literally: the
forms exist before any sensor is plugged in, and every modality adapter in
``lens.py`` has to project its raw data into them.

Each form is a small protocol with one default implementation. Swap any of
them by passing your own object to :class:`Forms`; nothing else changes.

    Time       one clock for every process; aligns streams that tick at
               different rates onto a single reference instant
    Space      one body frame; sensor poses, ray casting, and a grid where
               unseen cells stay unknown rather than empty
    Substance  what persists; a running baseline per stream, so the lens can
               say how far the present departs from what usually is
    Cause      what follows what; turns departures into events, gives each a
               refractory period, and links it to the event that preceded it
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ======================================================================= time
@runtime_checkable
class TimeForm(Protocol):
    def now_ns(self) -> int: ...

    def align(
        self, t_ref: int, candidates: dict[str, list[tuple[int, int]]]
    ) -> dict[str, tuple[int, int]]:
        """Given (index, t_ns) candidates per stream, pick one index per stream
        for the instant ``t_ref``. Returns {stream: (index, lag_ns)}; a stream
        with nothing close enough is left out."""
        ...


@dataclass
class MonotonicTime:
    """Nearest-neighbour alignment on the system monotonic clock.

    ``time.monotonic_ns`` is system-wide on Linux and macOS, so timestamps taken
    in different processes are directly comparable. A stream is dropped from a
    packet when its nearest frame is further than ``tolerance_ns`` from the
    reference, which is how a dead or lagging sensor shows up as missing
    rather than silently stale. Per-stream tolerances override the default.
    """

    tolerance_ns: int = 100_000_000
    per_stream: dict[str, int] = field(default_factory=dict)
    prefer_past: bool = True

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def align(self, t_ref, candidates):
        out = {}
        for stream, items in candidates.items():
            if not items:
                continue
            tol = self.per_stream.get(stream, self.tolerance_ns)
            pool = [(i, t_ref - t) for i, t in items]
            if self.prefer_past:
                # Do not let the future leak into the present, unless there is
                # nothing from the past at all.
                past = [p for p in pool if p[1] >= 0]
                pool = past or pool
            best = min(pool, key=lambda p: abs(p[1]))
            if abs(best[1]) <= tol:
                out[stream] = best
        return out


# ====================================================================== space
@dataclass(frozen=True)
class Pose:
    """Where a sensor sits in the body frame: x forward, y left, z up (metres)."""

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def from_euler(roll=0.0, pitch=0.0, yaw=0.0, xyz=(0.0, 0.0, 0.0)) -> "Pose":
        return Pose(euler_to_matrix(roll, pitch, yaw), np.asarray(xyz, float))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@runtime_checkable
class SpaceForm(Protocol):
    grid_shape: tuple[int, int]

    def to_body(self, points: np.ndarray, pose: Pose) -> np.ndarray: ...

    def carve(self, origin: np.ndarray, hits: np.ndarray, valid: np.ndarray, out: np.ndarray) -> None:
        """Write a (2, H, W) occupancy grid into ``out``: channel 0 is where
        rays ended (occupied), channel 1 is where rays passed or ended
        (observed). Cells in neither channel are unknown."""
        ...


@dataclass
class BodyGrid:
    """A top-down grid around the body. ``extent`` is (x_min, x_max, y_min, y_max)
    in metres. Row 0 is the far edge (x_max), column 0 is the left edge (y_max),
    so the grid reads like a map with the body looking up the page."""

    extent: tuple[float, float, float, float] = (0.0, 4.0, -2.0, 2.0)
    resolution: float = 0.1

    @property
    def grid_shape(self) -> tuple[int, int]:
        x0, x1, y0, y1 = self.extent
        return (int(round((x1 - x0) / self.resolution)), int(round((y1 - y0) / self.resolution)))

    def to_body(self, points, pose):
        return pose.apply(np.asarray(points, float))

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0, x1, y0, y1 = self.extent
        h, w = self.grid_shape
        r = np.floor((x1 - xy[..., 0]) / self.resolution).astype(np.int64)
        c = np.floor((y1 - xy[..., 1]) / self.resolution).astype(np.int64)
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        return r, c, inside

    def carve(self, origin, hits, valid, out):
        out[...] = 0
        h, w = self.grid_shape
        origin = np.asarray(origin, float)
        hits = np.asarray(hits, float)
        # March every ray at half a cell per step. Invalid rays (no return) are
        # not marked at all: no return means unknown, not free.
        lengths = np.linalg.norm(hits - origin, axis=-1)
        steps = int(np.ceil(np.nanmax(np.where(valid, lengths, 0.0), initial=0.0) / (self.resolution / 2))) + 1
        if valid.any():
            ts = np.linspace(0.0, 1.0, steps)[None, :, None]
            v_hits = hits[valid]
            path = origin + (v_hits[:, None, :] - origin) * ts
            r, c, inside = self.cells(path)
            out[1][r[inside], c[inside]] = 1.0
            r, c, inside = self.cells(v_hits)
            out[0][r[inside], c[inside]] = 1.0
            out[1][r[inside], c[inside]] = 1.0


# ================================================================== substance
@runtime_checkable
class SubstanceForm(Protocol):
    def observe(self, stream: str, x: np.ndarray) -> float:
        """Fold ``x`` into the stream's baseline and return its novelty: how far
        it departs from what persists, as a non-negative number where about 1
        is ordinary variation."""
        ...


def _half_normal_tail_mean(frac: float) -> float:
    """Mean of the largest ``frac`` of |N(0, 1)| samples, so that Gaussian
    noise scores 1 whatever the focus fraction."""
    lo, hi = 0.0, 10.0
    for _ in range(60):  # bisect for t with P(|z| > t) = frac
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if math.erfc(mid / math.sqrt(2)) > frac else (lo, mid)
    t = (lo + hi) / 2
    return math.exp(-t * t / 2) / math.sqrt(2 * math.pi) * 2 / frac


@dataclass
class RunningBaseline:
    """An exponentially weighted mean and variance per element, per stream.

    Novelty is the mean of the top ``focus`` fraction of absolute z-scores,
    ignoring NaN (unknown) elements, scaled so Gaussian noise scores about 1.
    Taking the top of the distribution rather than the whole mean matters:
    a small object in a big frame changes a few elements a lot, and a plain
    mean would dilute it away. ``warmup`` frames are absorbed before novelty
    is reported, so the first frames of a stream are not mistaken for an
    event. Large tensors are strided down to ``max_elems`` to stay cheap on a Pi.
    The per-element noise floor is ``floor`` plus ``relative_floor`` times the
    stream's typical spread across elements.
    """

    alpha: float = 0.05
    warmup: int = 10
    focus: float = 0.005
    max_elems: int = 4096
    floor: float = 1e-3
    relative_floor: float = 0.1
    _state: dict = field(default_factory=dict, repr=False)

    def _sample(self, x: np.ndarray) -> np.ndarray:
        flat = np.asarray(x, np.float32).reshape(-1)
        if flat.size > self.max_elems:
            flat = flat[:: int(np.ceil(flat.size / self.max_elems))]
        return flat

    def _focus(self, z: np.ndarray) -> float:
        k = max(1, int(np.ceil(z.size * self.focus)))
        top = np.partition(z, z.size - k)[z.size - k :]
        return float(top.mean() / _half_normal_tail_mean(self.focus))

    def observe(self, stream, x):
        v = self._sample(x)
        st = self._state.get(stream)
        if st is None or st["mean"].shape != v.shape:
            self._state[stream] = {
                "mean": np.nan_to_num(v.copy()),
                "var": np.full_like(v, self.floor**2),
                "spread": float(np.nanstd(v)) if np.isfinite(v).any() else 0.0,
                "n": 1,
            }
            return 0.0
        mean, var = st["mean"], st["var"]
        known = ~np.isnan(v)
        d = np.where(known, v - mean, 0.0)
        # An element that has never varied would make any flicker look
        # infinitely surprising (a binary occupancy cell on a grid boundary,
        # say). So the noise floor is never below a tenth of how much the
        # stream's elements differ from one another.
        floor2 = self.floor**2 + (self.relative_floor * st["spread"]) ** 2
        z = np.abs(d) / np.sqrt(var + floor2)
        novelty = self._focus(z[known]) if known.any() else 0.0
        a = self.alpha if st["n"] >= self.warmup else 1.0 / (st["n"] + 1)
        mean += a * d
        var += a * (d * d - var) * known
        if known.any():
            st["spread"] += a * (float(np.std(v[known])) - st["spread"])
        st["n"] += 1
        return novelty if st["n"] > self.warmup else 0.0


# ====================================================================== cause
@dataclass(frozen=True)
class Event:
    id: int
    stream: str
    t_ns: int
    novelty: float
    caused_by: int | None  # id of the event that came just before, from another stream


@runtime_checkable
class CauseForm(Protocol):
    def observe(self, stream: str, t_ns: int, novelty: float) -> Event | None: ...


@dataclass
class RefractoryCause:
    """An event fires when a stream's novelty crosses ``threshold``. After it
    fires, that stream is refractory for ``refractory_ns`` and cannot fire again,
    the way a neuron cannot fire twice in a row. Each event points to the most
    recent event on a different stream within ``window_ns`` before it, which is
    the weakest honest claim about cause: this came after that, nearby in time.
    """

    threshold: float = 3.0
    refractory_ns: int = 500_000_000
    window_ns: int = 500_000_000
    per_stream: dict[str, float] = field(default_factory=dict)
    history: list[Event] = field(default_factory=list, repr=False)
    max_history: int = 256
    _last_fire: dict = field(default_factory=dict, repr=False)
    _next_id: int = 0

    def observe(self, stream, t_ns, novelty):
        if novelty < self.per_stream.get(stream, self.threshold):
            return None
        last = self._last_fire.get(stream)
        if last is not None and t_ns - last < self.refractory_ns:
            return None
        cause = None
        for e in reversed(self.history):
            if t_ns - e.t_ns > self.window_ns:
                break
            if e.stream != stream and e.t_ns <= t_ns:
                cause = e.id
                break
        ev = Event(self._next_id, stream, t_ns, float(novelty), cause)
        self._next_id += 1
        self._last_fire[stream] = t_ns
        self.history.append(ev)
        del self.history[: -self.max_history]
        return ev


# ===================================================================== bundle
@dataclass
class Forms:
    """The four forms, bundled. ``Forms()`` gives the defaults."""

    time: TimeForm = field(default_factory=MonotonicTime)
    space: SpaceForm = field(default_factory=BodyGrid)
    substance: SubstanceForm = field(default_factory=RunningBaseline)
    cause: CauseForm = field(default_factory=RefractoryCause)
