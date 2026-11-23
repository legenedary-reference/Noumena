"""
lens.py: the interpretable focus.

The lens is the content of perception. Raw sensor output comes in: frames of
the wrong size with barrel distortion, PCM integers, gyro readings with bias,
time-of-flight zones that sometimes return nothing, encoder motion vectors
mixed with camera pan. What comes out is a fixed-shape, normalised tensor per
stream, stamped on the common clock and projected into the forms of
``forms.py``. The model never sees anything else.

There are two stages, and they run in different processes:

1. ``Adapter.refract(raw, out)`` runs inside each sensor daemon. It writes its
   result straight into a shared memory slot (``out`` is a view into the
   ring), so refraction is parallel across cores and the raw data never
   leaves the process that captured it.

2. ``Lens.apprehend`` and ``Lens.synthesize`` run in the eye. Apprehension
   folds each new frame into the substance and cause forms; synthesis aligns
   the latest frame of every stream onto one instant (the time form) and
   bundles the result into a :class:`PerceptionPacket` that the model reads.

Every adapter declares ``out_spec() -> (shape, dtype)`` up front so the eye
can size the ring before the first frame arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import warnings

import numpy as np

from .forms import Event, Forms, Pose, euler_to_matrix
from .ring import SlotRef


# ============================================================ raw input type
@dataclass
class RawSample:
    """What a sensor source hands the lens. ``data`` is whatever the device
    produced; ``t_ns`` is the capture instant on the common monotonic clock."""

    stream: str
    t_ns: int
    data: np.ndarray
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Adapter(Protocol):
    modality: str

    def out_spec(self) -> tuple[tuple[int, ...], np.dtype]: ...

    def refract(self, raw: RawSample, out: np.ndarray, forms: Forms) -> None: ...


# ================================================================ primitives
def bilinear_sample(img: np.ndarray, map_y: np.ndarray, map_x: np.ndarray) -> np.ndarray:
    """Sample ``img`` (H, W) or (H, W, C) at fractional coordinates. Points
    outside the image clamp to the border."""
    h, w = img.shape[:2]
    x = np.clip(map_x, 0, w - 1)
    y = np.clip(map_y, 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    im = img.astype(np.float32, copy=False)
    top = im[y0, x0] * (1 - fx) + im[y0, x1] * fx
    bot = im[y1, x0] * (1 - fx) + im[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def block_mean(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Average-pool the first two axes of ``a`` down to ``out_hw``, ignoring
    NaN. Trims edges that do not divide evenly."""
    h, w = a.shape[:2]
    oh, ow = out_hw
    bh, bw = max(h // oh, 1), max(w // ow, 1)
    a = a[: bh * oh, : bw * ow]
    rest = a.shape[2:]
    blocks = a.reshape(oh, bh, ow, bw, *rest)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are fine: they stay NaN
        return np.nanmean(blocks, axis=(1, 3))


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft // 2 + 1)."""
    fmax = fmax or sr / 2
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    inv = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, bins.size), np.float32)
    for m in range(n_mels):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def mulaw_encode(x: np.ndarray, levels: int = 256) -> np.ndarray:
    """Floats in [-1, 1] to integer tokens in [0, levels)."""
    mu = levels - 1
    y = np.sign(x) * np.log1p(mu * np.abs(np.clip(x, -1, 1))) / np.log1p(mu)
    return np.clip(np.round((y + 1) / 2 * mu), 0, mu).astype(np.uint8 if levels <= 256 else np.uint16)


def mulaw_decode(tokens: np.ndarray, levels: int = 256) -> np.ndarray:
    mu = levels - 1
    y = tokens.astype(np.float32) / mu * 2 - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def quantize_uniform(x: np.ndarray, lo: float, hi: float, levels: int = 256) -> np.ndarray:
    """Clip to [lo, hi] and map to integer tokens in [0, levels)."""
    q = np.round((np.clip(x, lo, hi) - lo) / (hi - lo) * (levels - 1))
    return q.astype(np.uint8 if levels <= 256 else np.uint16)


# ================================================================== adapters
@dataclass
class Intrinsics:
    """Pinhole camera with two-term radial distortion (k1, k2)."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0


@dataclass
class CameraAdapter:
    """Video frames to a normalised (C, H, W) float32 tensor.

    rectify   undo radial lens distortion with ``intrinsics`` (map is built once)
    crop      centre-crop to the output aspect ratio, or an explicit ROI
              (top, left, height, width) in source pixels
    resize    bilinear, to ``size`` (H, W)
    normalise to [0, 1], then per-channel (x - mean) / std
    """

    size: tuple[int, int] = (96, 128)
    grayscale: bool = False
    intrinsics: Intrinsics | None = None
    roi: tuple[int, int, int, int] | None = None
    mean: tuple[float, ...] = (0.5,)
    std: tuple[float, ...] = (0.25,)
    modality: str = "camera"
    _map: tuple | None = field(default=None, repr=False)
    _map_for: tuple | None = field(default=None, repr=False)

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3

    def out_spec(self):
        return (self.channels, *self.size), np.dtype(np.float32)

    def _build_map(self, src_hw: tuple[int, int]):
        sh, sw = src_hw
        oh, ow = self.size
        if self.roi is not None:
            top, left, rh, rw = self.roi
        else:
            aspect = ow / oh
            rw = min(sw, int(round(sh * aspect)))
            rh = min(sh, int(round(rw / aspect)))
            top, left = (sh - rh) // 2, (sw - rw) // 2
        # Output pixel centres in source (undistorted) coordinates.
        ys = top + (np.arange(oh) + 0.5) * rh / oh - 0.5
        xs = left + (np.arange(ow) + 0.5) * rw / ow - 0.5
        uu, vv = np.meshgrid(xs, ys)
        if self.intrinsics is not None:
            k = self.intrinsics
            x = (uu - k.cx) / k.fx
            y = (vv - k.cy) / k.fy
            r2 = x * x + y * y
            f = 1 + k.k1 * r2 + k.k2 * r2 * r2
            uu = k.fx * x * f + k.cx
            vv = k.fy * y * f + k.cy
        self._map = (vv.astype(np.float32), uu.astype(np.float32))
        self._map_for = src_hw

    def refract(self, raw, out, forms):
        img = raw.data
        if img.ndim == 2:
            img = img[..., None]
        if self._map_for != img.shape[:2]:
            self._build_map(img.shape[:2])
        scale = 255.0 if img.dtype == np.uint8 else 1.0
        if self.grayscale and img.shape[2] == 3:
            img = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            img = img[..., None]
        elif not self.grayscale and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        sampled = bilinear_sample(img, *self._map)  # (H, W, C)
        sampled /= scale
        mean = np.resize(np.asarray(self.mean, np.float32), self.channels)
        std = np.resize(np.asarray(self.std, np.float32), self.channels)
        out[...] = ((sampled - mean) / std).transpose(2, 0, 1)


@dataclass
class AudioAdapter:
    """PCM chunks to log-mel frames (T, n_mels) float32, or to mu-law tokens.

    A chunk of ``chunk`` samples at ``sample_rate`` gives
    ``1 + (chunk - n_fft) // hop`` frames. Stereo is mixed to mono. With
    ``mode="mulaw"`` the output is (chunk,) uint8 tokens instead, for models
    that read audio as a token stream.
    """

    sample_rate: int = 16_000
    chunk: int = 1600
    n_fft: int = 400
    hop: int = 160
    n_mels: int = 40
    preemphasis: float = 0.97
    mode: str = "logmel"
    modality: str = "audio"
    _fb: np.ndarray | None = field(default=None, repr=False)
    _win: np.ndarray | None = field(default=None, repr=False)
    _prev: float = 0.0

    @property
    def frames(self) -> int:
        return 1 + (self.chunk - self.n_fft) // self.hop

    def out_spec(self):
        if self.mode == "mulaw":
            return (self.chunk,), np.dtype(np.uint8)
        return (self.frames, self.n_mels), np.dtype(np.float32)

    def _pcm(self, data: np.ndarray) -> np.ndarray:
        x = data
        if x.ndim == 2:
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / float(np.iinfo(data.dtype).max)
        x = np.asarray(x, np.float32)
        if x.size < self.chunk:
            x = np.pad(x, (0, self.chunk - x.size))
        return x[: self.chunk]

    def refract(self, raw, out, forms):
        x = self._pcm(raw.data)
        if self.mode == "mulaw":
            out[...] = mulaw_encode(x)
            return
        if self._fb is None:
            self._fb = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)
            self._win = np.hanning(self.n_fft).astype(np.float32)
        # Pre-emphasis, carrying the last sample across chunks.
        y = np.empty_like(x)
        y[0] = x[0] - self.preemphasis * self._prev
        y[1:] = x[1:] - self.preemphasis * x[:-1]
        self._prev = float(x[-1])
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(self.frames)[:, None]
        spec = np.abs(np.fft.rfft(y[idx] * self._win, axis=1)) ** 2
        out[...] = np.log(spec @ self._fb.T + 1e-10)


@dataclass
class IMUAdapter:
    """Accelerometer and gyroscope bursts (N, 6) to a 12-element state vector.

    Input rows are [ax, ay, az, gx, gy, gz] in m/s^2 and rad/s, sampled at
    ``rate``. Gyro bias is learned from the first ``calibrate`` samples, which
    assumes the device starts still. Orientation comes from a complementary
    filter: gyro integration corrected towards the gravity direction.

    Output: [roll, pitch, yaw, wx, wy, wz, lin_ax, lin_ay, lin_az,
             vib_x, vib_y, vib_z], where lin_a is acceleration with gravity
    removed (rotated into the level frame) and vib is per-axis RMS
    acceleration around the burst mean.
    """

    rate: float = 100.0
    blend: float = 0.98
    calibrate: int = 50
    gravity: float = 9.80665
    pose: Pose = field(default_factory=Pose)
    modality: str = "imu"
    _bias: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _cal: list = field(default_factory=list, repr=False)
    _rpy: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _init: bool = False

    def out_spec(self):
        return (12,), np.dtype(np.float32)

    def refract(self, raw, out, forms):
        s = np.asarray(raw.data, np.float64).reshape(-1, 6)
        acc = s[:, :3] @ self.pose.rotation.T
        gyr = s[:, 3:] @ self.pose.rotation.T
        if len(self._cal) < self.calibrate:
            self._cal.extend(gyr.tolist())
            self._bias = np.mean(self._cal, axis=0)
        gyr = gyr - self._bias
        dt = 1.0 / self.rate
        for a, w in zip(acc, gyr):
            roll_a = np.arctan2(a[1], a[2])
            pitch_a = np.arctan2(-a[0], np.hypot(a[1], a[2]))
            if not self._init:
                self._rpy[:] = (roll_a, pitch_a, 0.0)
                self._init = True
                continue
            self._rpy += w * dt
            self._rpy[0] = self.blend * self._rpy[0] + (1 - self.blend) * roll_a
            self._rpy[1] = self.blend * self._rpy[1] + (1 - self.blend) * pitch_a
        roll, pitch, yaw = self._rpy
        # Rotate the burst mean into the level frame and take gravity off.
        level = euler_to_matrix(roll, pitch, 0.0) @ acc.mean(axis=0)
        lin = level - np.array([0.0, 0.0, self.gravity])
        vib = np.sqrt(np.mean((acc - acc.mean(axis=0)) ** 2, axis=0))
        out[...] = np.concatenate([[roll, pitch, (yaw + np.pi) % (2 * np.pi) - np.pi], gyr.mean(axis=0), lin, vib])


@dataclass
class DistanceAdapter:
    """Range readings to a top-down (2, H, W) occupancy grid in the body frame.

    Input is a (rows, cols) grid of ranges in metres from a multizone
    time-of-flight sensor (VL53L5CX style, 8x8) or a (N,) scan; NaN, zero, or
    beyond ``max_range`` means no return. Each zone becomes a ray through the
    sensor's field of view, placed in the body frame by ``pose``, and the space
    form carves it into the grid: channel 0 marks where something was hit,
    channel 1 marks everything the sensor actually observed. A cell that is 0
    in both is unknown, never assumed empty.
    """

    zones: tuple[int, int] = (8, 8)
    fov_deg: tuple[float, float] = (45.0, 45.0)
    max_range: float = 4.0
    pose: Pose = field(default_factory=lambda: Pose.from_euler(xyz=(0.0, 0.0, 0.3)))
    modality: str = "distance"
    _dirs: np.ndarray | None = field(default=None, repr=False)

    def out_spec(self, forms: Forms | None = None):
        shape = (forms or Forms()).space.grid_shape
        return (2, *shape), np.dtype(np.float32)

    def _rays(self) -> np.ndarray:
        if self._dirs is None:
            rows, cols = self.zones
            vf, hf = np.radians(self.fov_deg[0]), np.radians(self.fov_deg[1])
            el = (0.5 - (np.arange(rows) + 0.5) / rows) * vf  # top row looks up
            az = (0.5 - (np.arange(cols) + 0.5) / cols) * hf  # left column looks left
            az, el = np.meshgrid(az, el)
            d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
            self._dirs = d.reshape(-1, 3)
        return self._dirs

    def refract(self, raw, out, forms):
        r = np.asarray(raw.data, np.float64).reshape(-1)
        valid = np.isfinite(r) & (r > 0) & (r <= self.max_range)
        r = np.where(valid, r, 0.0)
        local = self._rays() * r[:, None]
        hits = forms.space.to_body(local, self.pose)
        forms.space.carve(self.pose.translation, hits, valid, out)


@dataclass
class MotionAdapter:
    """Motion vectors (H, W, 2) to a (3, gh, gw) flow tensor.

    Input is a dense or per-macroblock field of (dx, dy) in pixels per frame,
    such as the vectors a Pi's H.264 encoder emits for free. Ego-motion is the
    median vector; with ``remove_ego`` it is subtracted, so a panning camera
    does not look like a moving world. Output channels are dx, dy, and
    magnitude, pooled to ``grid`` and scaled by ``max_speed``.
    """

    grid: tuple[int, int] = (12, 16)
    max_speed: float = 16.0
    remove_ego: bool = True
    modality: str = "motion"
    last_ego: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)

    def out_spec(self):
        return (3, *self.grid), np.dtype(np.float32)

    def refract(self, raw, out, forms):
        v = np.asarray(raw.data, np.float32)
        if self.remove_ego:
            self.last_ego = np.nanmedian(v.reshape(-1, 2), axis=0)
            v = v - self.last_ego
        pooled = block_mean(v, self.grid)  # (gh, gw, 2)
        pooled = np.nan_to_num(pooled) / self.max_speed
        out[0] = pooled[..., 0]
        out[1] = pooled[..., 1]
        out[2] = np.hypot(pooled[..., 0], pooled[..., 1])


ADAPTERS = {
    "camera": CameraAdapter,
    "audio": AudioAdapter,
    "imu": IMUAdapter,
    "distance": DistanceAdapter,
    "motion": MotionAdapter,
}


def out_spec_of(adapter: Adapter, forms: Forms) -> tuple[tuple[int, ...], np.dtype]:
    try:
        return adapter.out_spec(forms)  # type: ignore[call-arg]
    except TypeError:
        return adapter.out_spec()


# ============================================================ packet + lens
@dataclass
class StreamState:
    ref: SlotRef
    lag_ns: int
    novelty: float


@dataclass
class PerceptionPacket:
    """What the model receives. Small and picklable: tensors travel as
    :class:`SlotRef` pointers and are resolved to shared memory views on the
    model side, so the only thing copied is this header."""

    seq: int
    t_ref: int
    streams: dict[str, StreamState]
    events: list[Event]
    reason: str

    def describe(self) -> str:
        parts = []
        for name, s in sorted(self.streams.items()):
            parts.append(f"{name} lag {s.lag_ns / 1e6:.0f} ms novelty {s.novelty:.1f}")
        text = f"Packet {self.seq} ({self.reason}): " + "; ".join(parts) + "."
        if self.events:
            ev = ", ".join(
                f"{e.stream}" + (f" after event {e.caused_by}" if e.caused_by is not None else "")
                for e in self.events
            )
            text += f" Events: {ev}."
        return text


class Lens:
    """The eye-side half of the lens: apprehension and synthesis."""

    def __init__(self, forms: Forms | None = None):
        self.forms = forms or Forms()
        self.novelty: dict[str, float] = {}
        self._seq = 0

    def apprehend(self, stream: str, t_ns: int, array: np.ndarray) -> Event | None:
        """Fold one new frame into the forms of substance and cause."""
        nov = self.forms.substance.observe(stream, array)
        self.novelty[stream] = nov
        return self.forms.cause.observe(stream, t_ns, nov)

    def synthesize(
        self,
        t_ref: int,
        candidates: dict[str, tuple[str, list[tuple[int, int]]]],
        events: list[Event],
        reason: str,
    ) -> PerceptionPacket:
        """Align every stream onto ``t_ref`` and bundle the packet.
        ``candidates`` maps stream to (ring name, [(index, t_ns), ...])."""
        chosen = self.forms.time.align(t_ref, {s: c[1] for s, c in candidates.items()})
        streams = {}
        for s, (index, lag) in chosen.items():
            ring = candidates[s][0]
            t = dict(candidates[s][1])[index]
            streams[s] = StreamState(SlotRef(ring, index, t), lag, self.novelty.get(s, 0.0))
        pkt = PerceptionPacket(self._seq, t_ref, streams, list(events), reason)
        self._seq += 1
        return pkt


__all__ = [
    "Adapter",
    "ADAPTERS",
    "AudioAdapter",
    "CameraAdapter",
    "DistanceAdapter",
    "IMUAdapter",
    "Intrinsics",
    "Lens",
    "MotionAdapter",
    "PerceptionPacket",
    "RawSample",
    "StreamState",
    "bilinear_sample",
    "block_mean",
    "mel_filterbank",
    "mulaw_decode",
    "mulaw_encode",
    "out_spec_of",
    "quantize_uniform",
]
