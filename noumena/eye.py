"""
eye.py: the locus of control.

The eye owns everything that is not the content of perception: the shared
memory rings, the processes that fill them, the clock that keeps sensors in
step, the decision of when to look, and the hand-off to the model. It never
touches pixel or sample data itself except through zero-copy views.

Process layout::

     sensor daemon (one per sensor)            eye (main process)              model worker
    +-----------------------------+     +------------------------------+    +------------------+
    | source.read()               |     | wait(doorbells, results)     |    | recv(packet)     |
    | adapter.refract(raw, slot) -+---->| ring.latest() -> apprehend   |    | resolve SlotRefs |
    |   writes into shared memory |bell | triggers -> synthesize       +--->|   to shm views   |
    | ring.commit(); ring the bell|     | supervise, restart, shutdown |pkt | model.perceive() |
    +-----------------------------+     +------------------------------+<---+ send(result)     |
                 ^                                                    result +------------------+
                 | tick (optional)
          hardware clock / GPIO strobe

Data path: the sensor daemon refracts straight into a ring slot. The eye reads
that slot as a view to update substance and cause. The model worker maps the
same pages and reads the same view. The only bytes that cross a pipe are a
one-byte doorbell per frame and a small packet header per dispatch.

Event-driven: the eye blocks in ``multiprocessing.connection.wait`` on the
doorbells and the model's result pipe, with a timeout that is exactly the
time until the next periodic trigger or supervision check. There is no
polling loop and no sleep.

Backpressure: at most one packet is in flight. If triggers fire while the
model is busy, they are coalesced, and the freshest data is sent the moment it
is free. Sensors never wait for the model; slow readers lose old frames, not
new ones.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import secrets
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .forms import Event, Forms
from .lens import ADAPTERS, Lens, PerceptionPacket, out_spec_of
from .ring import SharedRing, make_name
from .sensors import SIMULATORS


# ================================================================== config
@dataclass
class SensorSpec:
    """How to build one sensor daemon. Classes and kwargs, not instances, so
    the spec pickles cleanly under the ``spawn`` start method that macOS uses."""

    name: str
    source: type
    adapter: type
    source_kwargs: dict = field(default_factory=dict)
    adapter_kwargs: dict = field(default_factory=dict)
    slots: int = 16
    triggered: bool = False

    @staticmethod
    def simulated(modality: str, name: str | None = None, triggered: bool = False, **adapter_kwargs) -> "SensorSpec":
        name = name or modality
        return SensorSpec(
            name=name,
            source=SIMULATORS[modality],
            adapter=ADAPTERS[modality],
            source_kwargs={"stream": name, "paced": not triggered},
            adapter_kwargs=adapter_kwargs,
            triggered=triggered,
        )


# =========================================================== hardware sync
class TickSync:
    """A shared tick counter that triggered sensors wait on, so several
    sensors capture on the same edge. Fire it from a timer process
    (:func:`clock_daemon`) or from a GPIO interrupt (:func:`gpio_tick`)."""

    def __init__(self, ctx):
        self._count = ctx.Value("Q", 0, lock=False)
        self._cond = ctx.Condition()

    @property
    def count(self) -> int:
        return int(self._count.value)

    def fire(self) -> None:
        with self._cond:
            self._count.value += 1
            self._cond.notify_all()

    def wait_after(self, seen: int, timeout: float) -> int | None:
        """Block until the tick count passes ``seen``. Returns the new count,
        or None on timeout."""
        with self._cond:
            if self._cond.wait_for(lambda: self._count.value > seen, timeout):
                return int(self._count.value)
        return None


def clock_daemon(sync: TickSync, hz: float, stop) -> None:
    """Fire ``sync`` at ``hz`` on an absolute schedule, like a camera's frame
    sync strobe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    t0 = time.monotonic()
    n = 0
    while not stop.is_set():
        n += 1
        delay = t0 + n / hz - time.monotonic()
        if delay > 0 and stop.wait(delay):
            break
        sync.fire()


def gpio_tick(sync: TickSync, pin: int):  # pragma: no cover - needs a Pi
    """Fire ``sync`` on each rising edge of a GPIO pin (Raspberry Pi, via
    gpiozero). Keep the returned object alive. Not tested on hardware."""
    from gpiozero import Button  # type: ignore[import-not-found]

    button = Button(pin, pull_up=False)
    button.when_pressed = lambda: sync.fire()
    return button


# ============================================================ sensor daemon
def _ring_bell(fd: int) -> None:
    try:
        os.write(fd, b"\x01")
    except (BlockingIOError, BrokenPipeError):
        pass  # the eye is behind; the ring still has the frame


def sensor_daemon(spec: SensorSpec, ring_name: str, bell, stop, heartbeat, sync, forms: Forms) -> None:
    """Capture, refract into shared memory, ring the bell. Runs in its own
    process. Exceptions end the process; the eye restarts it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring = SharedRing.attach(ring_name)
    source = spec.source(**spec.source_kwargs)
    adapter = spec.adapter(**spec.adapter_kwargs)
    shape, dtype = out_spec_of(adapter, forms)
    fd = bell.fileno()
    os.set_blocking(fd, False)
    seen = sync.count if (spec.triggered and sync is not None) else 0
    try:
        while not stop.is_set():
            heartbeat.value = time.monotonic()
            if spec.triggered and sync is not None:
                tick = sync.wait_after(seen, timeout=0.2)
                if tick is None:
                    continue
                seen = tick
            raw = source.read()
            out = ring.begin(shape, dtype)
            try:
                adapter.refract(raw, out, forms)
            except Exception:
                ring.abort()
                raise
            del out
            ring.commit(raw.t_ns)
            _ring_bell(fd)
    finally:
        ring.close()
        if hasattr(source, "close"):
            source.close()


# ================================================================== model
class Percept:
    """What a model sees: the packet plus each stream's tensor as a read-only
    view into shared memory. Call :meth:`stale` after you are done to learn
    whether any slot was overwritten while you were reading it."""

    def __init__(self, packet: PerceptionPacket, frames: dict):
        self.packet = packet
        self._frames = frames
        self.tensors: dict[str, np.ndarray] = {s: f.array for s, f in frames.items()}

    def __getitem__(self, stream: str) -> np.ndarray:
        return self.tensors[stream]

    def __contains__(self, stream: str) -> bool:
        return stream in self.tensors

    def stale(self) -> list[str]:
        return [s for s, f in self._frames.items() if not f.valid()]


@runtime_checkable
class Model(Protocol):
    """Anything with ``perceive(percept) -> result``. Optional ``setup()``
    runs once in the worker before the first packet (load weights there, not
    in ``__init__``), and optional ``close()`` runs at shutdown. The result must
    pickle when the model runs in a worker process."""

    def perceive(self, percept: Percept) -> Any: ...


class SummaryModel:
    """A stand-in model that reports what it was given, in plain sentences."""

    def perceive(self, percept: Percept) -> str:
        parts = []
        for s in sorted(percept.tensors):
            t = percept.tensors[s]
            parts.append(f"{s} {'x'.join(map(str, t.shape))} mean {float(np.nanmean(t.astype(np.float32))):.2f}")
        return percept.packet.describe() + " Tensors: " + "; ".join(parts) + "."


class _Resolver:
    """Maps SlotRefs to shared memory views, attaching to each ring once."""

    def __init__(self):
        self.rings: dict[str, SharedRing] = {}

    def percept(self, packet: PerceptionPacket) -> tuple[Percept, list[str]]:
        frames, missing = {}, []
        for s, st in packet.streams.items():
            ring = self.rings.get(st.ref.ring)
            if ring is None:
                ring = self.rings[st.ref.ring] = SharedRing.attach(st.ref.ring)
            f = ring.resolve(st.ref)
            if f is None:
                missing.append(s)
            else:
                frames[s] = f
        return Percept(packet, frames), missing

    def close(self):
        for r in self.rings.values():
            r.close()
        self.rings.clear()


def _run_model(model, resolver: _Resolver, packet: PerceptionPacket):
    t0 = time.monotonic_ns()
    percept, missing = resolver.percept(packet)
    result = model.perceive(percept)
    stale = missing + percept.stale()
    del percept
    return result, stale, time.monotonic_ns() - t0


def model_worker(model_cls: type, model_kwargs: dict, conn, stop, heartbeat) -> None:
    """Receive packets, resolve them to views, run the model, send results."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    model = model_cls(**model_kwargs)
    if hasattr(model, "setup"):
        model.setup()
    resolver = _Resolver()
    conn.send(("ready",))
    try:
        while not stop.is_set():
            heartbeat.value = time.monotonic()
            if not conn.poll(0.2):
                continue
            try:
                packet = conn.recv()
            except EOFError:
                break
            try:
                result, stale, dt = _run_model(model, resolver, packet)
                conn.send(("result", packet.seq, result, stale, dt))
            except Exception:
                conn.send(("error", packet.seq, traceback.format_exc(), [], 0))
    finally:
        resolver.close()
        if hasattr(model, "close"):
            model.close()


# ================================================================ triggers
class Trigger:
    """Decides when the eye should look. ``check`` returns a reason or None;
    ``deadline`` returns the next instant (ns) the trigger wants to be
    checked even if nothing arrives, so the eye can sleep until then."""

    def check(self, now: int, new: set[str], events: list[Event]) -> str | None:
        return None

    def deadline(self, now: int) -> int | None:
        return None


@dataclass
class Periodic(Trigger):
    """Look at a fixed rate, whatever happens."""

    hz: float = 5.0
    _next: int = 0

    def check(self, now, new, events):
        if now < self._next:
            return None
        period = int(1e9 / self.hz)
        # Stay on the schedule, unless we fell more than a period behind.
        self._next = self._next + period if now - self._next < period else now + period
        return f"every {1 / self.hz:.2f} s"

    def deadline(self, now):
        return self._next


@dataclass
class OnStream(Trigger):
    """Look whenever a given stream has a new frame (e.g. every camera frame)."""

    stream: str = "camera"

    def check(self, now, new, events):
        return f"new {self.stream} frame" if self.stream in new else None


@dataclass
class OnEvent(Trigger):
    """Look when the cause form reports an event, optionally only for some
    streams. This is the salience trigger: the model wakes when something
    happens, and otherwise sleeps."""

    streams: tuple[str, ...] = ()

    def check(self, now, new, events):
        for e in events:
            if not self.streams or e.stream in self.streams:
                return f"event on {e.stream}"
        return None


# ===================================================================== eye
@dataclass
class _Worker:
    name: str
    target: Callable
    args: tuple
    heartbeat: Any
    proc: Any = None
    restarts: int = 0
    retry_at: float = 0.0
    failed: bool = False


@dataclass
class EyeStats:
    frames: dict[str, int] = field(default_factory=dict)
    apprehended: dict[str, int] = field(default_factory=dict)
    events: int = 0
    dispatched: int = 0
    results: int = 0
    errors: int = 0
    coalesced: int = 0
    stale: int = 0
    restarts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        fr = ", ".join(f"{k} {v}" for k, v in sorted(self.frames.items()))
        rs = ", ".join(f"{k} {v}" for k, v in sorted(self.restarts.items()) if v) or "none"
        return (
            f"Frames: {fr}. Events: {self.events}. Packets sent {self.dispatched}, "
            f"answered {self.results}, coalesced {self.coalesced}, errors {self.errors}, "
            f"stale {self.stale}. Restarts: {rs}."
        )


class Eye:
    """The executive runtime. Typical use::

        eye = Eye([SensorSpec.simulated("camera"), SensorSpec.simulated("imu")],
                  model=SummaryModel, triggers=[Periodic(5), OnEvent()])
        with eye:
            eye.run(seconds=3, on_result=print)
    """

    def __init__(
        self,
        sensors: list[SensorSpec],
        model: type = SummaryModel,
        model_kwargs: dict | None = None,
        forms: Forms | None = None,
        triggers: list[Trigger] | None = None,
        inline: bool = False,
        clock_hz: float | None = None,
        start_method: str = "spawn",
        heartbeat_timeout: float = 3.0,
        max_restarts: int = 5,
        supervise_every: float = 0.25,
    ):
        names = [s.name for s in sensors]
        if len(set(names)) != len(names):
            raise ValueError(f"sensor names must be unique: {names}")
        self.sensors = sensors
        self.model_cls = model
        self.model_kwargs = model_kwargs or {}
        self.forms = forms or Forms()
        self.lens = Lens(self.forms)
        self.triggers = triggers if triggers is not None else [Periodic(5.0), OnEvent()]
        self.inline = inline
        self.clock_hz = clock_hz
        self.ctx = mp.get_context(start_method)
        self.heartbeat_timeout = heartbeat_timeout
        self.max_restarts = max_restarts
        self.supervise_every = supervise_every
        self.run_id = secrets.token_hex(3)
        self.stats = EyeStats()
        self.rings: dict[str, SharedRing] = {}
        self._bells: dict[str, tuple] = {}
        self._workers: dict[str, _Worker] = {}
        self._stop = self.ctx.Event()
        self._sync: TickSync | None = None
        self._model_conn = None
        self._model = None
        self._resolver: _Resolver | None = None
        self._model_ready = False
        self._busy: PerceptionPacket | None = None
        self._pending: str | None = None
        self._events: list[Event] = []
        self._last_seen: dict[str, int] = {}
        self._started = False
        self._halt = threading.Event()

    # ---------------------------------------------------------- lifecycle
    def start(self) -> "Eye":
        if self._started:
            return self
        self._started = True
        needs_clock = any(s.triggered for s in self.sensors)
        if needs_clock:
            self._sync = TickSync(self.ctx)
        for spec in self.sensors:
            shape, dtype = out_spec_of(spec.adapter(**spec.adapter_kwargs), self.forms)
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ring = SharedRing.create(make_name(spec.name, self.run_id), spec.slots, nbytes)
            self.rings[spec.name] = ring
            r, w = self.ctx.Pipe(duplex=False)
            os.set_blocking(r.fileno(), False)
            self._bells[spec.name] = (r, w)
            self._last_seen[spec.name] = -1
            hb = self.ctx.Value("d", time.monotonic(), lock=False)
            self._add_worker(
                spec.name, sensor_daemon, (spec, ring.name, w, self._stop, hb, self._sync, self.forms), hb
            )
        if needs_clock:
            hz = self.clock_hz or 30.0
            hb = self.ctx.Value("d", time.monotonic(), lock=False)
            self._add_worker("clock", clock_daemon, (self._sync, hz, self._stop), hb, watch=False)
        if self.inline:
            self._model = self.model_cls(**self.model_kwargs)
            if hasattr(self._model, "setup"):
                self._model.setup()
            self._resolver = _Resolver()
            self._model_ready = True
        else:
            self._start_model_worker()
        return self

    def _add_worker(self, name, target, args, hb, watch=True):
        w = _Worker(name, target, args, hb if watch else None)
        self._workers[name] = w
        self.stats.restarts.setdefault(name, 0)
        self._spawn(w)

    def _spawn(self, w: _Worker):
        if w.heartbeat is not None:
            w.heartbeat.value = time.monotonic()
        w.proc = self.ctx.Process(target=w.target, args=w.args, name=f"noumena-{w.name}", daemon=True)
        w.proc.start()

    def _start_model_worker(self):
        parent, child = self.ctx.Pipe(duplex=True)
        self._model_conn = parent
        self._model_ready = False
        hb = self.ctx.Value("d", time.monotonic(), lock=False)
        args = (self.model_cls, self.model_kwargs, child, self._stop, hb)
        if "model" in self._workers:
            w = self._workers["model"]
            w.args, w.heartbeat = args, hb
            self._spawn(w)
        else:
            self._add_worker("model", model_worker, args, hb)

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._sync is not None:
            self._sync.fire()  # wake triggered sensors so they see the stop
        for w in self._workers.values():
            if w.proc is not None:
                w.proc.join(timeout=2.0)
                if w.proc.is_alive():
                    w.proc.terminate()
                    w.proc.join(timeout=1.0)
        if self._model is not None and hasattr(self._model, "close"):
            self._model.close()
        if self._resolver is not None:
            self._resolver.close()
        for r, w in self._bells.values():
            r.close()
            w.close()
        if self._model_conn is not None:
            self._model_conn.close()
        for ring in self.rings.values():
            ring.close()
            ring.unlink()
        self._started = False

    def __enter__(self) -> "Eye":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def halt(self) -> None:
        """Ask :meth:`run` to return. Safe to call from a signal handler or
        another thread."""
        self._halt.set()

    # -------------------------------------------------------------- loop
    def run(
        self,
        seconds: float | None = None,
        max_results: int | None = None,
        on_result: Callable[[PerceptionPacket, Any, list[str]], None] | None = None,
    ) -> EyeStats:
        """Run the event loop until ``seconds`` pass, ``max_results`` results
        arrive, or :meth:`halt` is called."""
        self.start()
        self._halt.clear()
        on_result = on_result or (lambda p, r, s: None)
        end = time.monotonic_ns() + int(seconds * 1e9) if seconds is not None else None
        next_supervise = 0
        bells = {self._bells[s][0]: s for s in self._bells}
        old = None
        if threading.current_thread() is threading.main_thread():
            old = {sig: signal.signal(sig, lambda *_: self.halt()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not self._halt.is_set():
                now = time.monotonic_ns()
                if end is not None and now >= end:
                    break
                if max_results is not None and self.stats.results >= max_results:
                    break
                wake = [d for d in (t.deadline(now) for t in self.triggers) if d is not None]
                wake += [next_supervise] + ([end] if end is not None else [])
                timeout = max(0.0, (min(wake) - now) / 1e9)
                conns = list(bells) + ([self._model_conn] if self._model_conn is not None else [])
                ready = wait(conns, timeout=min(timeout, 1.0))

                new: set[str] = set()
                for c in ready:
                    if c is self._model_conn:
                        self._drain_results(on_result)
                    else:
                        self._drain_bell(c)
                        new.add(bells[c])
                now = time.monotonic_ns()
                events = self._apprehend(new)
                for t in self.triggers:
                    reason = t.check(now, new, events)
                    if reason is not None and self._pending is None:
                        self._pending = reason
                    elif reason is not None:
                        self.stats.coalesced += 1
                self._maybe_dispatch(now, on_result)
                if now >= next_supervise:
                    self._supervise()
                    next_supervise = now + int(self.supervise_every * 1e9)
        finally:
            if old:
                for sig, h in old.items():
                    signal.signal(sig, h)
        return self.stats

    def _drain_bell(self, conn) -> None:
        try:
            while os.read(conn.fileno(), 4096):
                pass
        except BlockingIOError:
            pass

    def _apprehend(self, new: set[str]) -> list[Event]:
        events = []
        for s in new:
            ring = self.rings[s]
            self.stats.frames[s] = ring.commits
            f = ring.latest()
            if f is None or f.index == self._last_seen[s]:
                continue
            self._last_seen[s] = f.index
            ev = self.lens.apprehend(s, f.t_ns, f.array)
            valid = f.valid()
            del f
            self.stats.apprehended[s] = self.stats.apprehended.get(s, 0) + 1
            if ev is not None and valid:
                events.append(ev)
        self.stats.events += len(events)
        self._events.extend(events)
        return events

    def _maybe_dispatch(self, now: int, on_result) -> None:
        if self._pending is None or self._busy is not None or not self._model_ready:
            return
        candidates = {s: (ring.name, ring.recent()) for s, ring in self.rings.items()}
        packet = self.lens.synthesize(now, candidates, self._events, self._pending)
        self._events = []
        self._pending = None
        self.stats.dispatched += 1
        if self.inline:
            try:
                result, stale, _ = _run_model(self._model, self._resolver, packet)
                self._deliver(packet, result, stale, on_result)
            except Exception:
                self.stats.errors += 1
                traceback.print_exc()
            return
        self._busy = packet
        self._model_conn.send(packet)

    def _deliver(self, packet, result, stale, on_result):
        self.stats.results += 1
        if stale:
            self.stats.stale += 1
        on_result(packet, result, stale)

    def _drain_results(self, on_result) -> None:
        try:
            while self._model_conn.poll():
                msg = self._model_conn.recv()
                if msg[0] == "ready":
                    self._model_ready = True
                    continue
                kind, seq, payload, stale, _dt = msg
                packet, self._busy = self._busy, None
                if kind == "result" and packet is not None and packet.seq == seq:
                    self._deliver(packet, payload, stale, on_result)
                elif kind == "error":
                    self.stats.errors += 1
                    print(f"model error on packet {seq}:\n{payload}", file=sys.stderr)
        except (EOFError, OSError):
            # The worker died. Stop waiting on its pipe; supervision restarts it.
            self._model_conn.close()
            self._model_conn = None
            self._model_ready = False

    # -------------------------------------------------------- supervision
    def _supervise(self) -> None:
        now = time.monotonic()
        for w in self._workers.values():
            if w.failed or w.proc is None:
                continue
            alive = w.proc.is_alive()
            hung = alive and w.heartbeat is not None and now - w.heartbeat.value > self.heartbeat_timeout
            if alive and not hung:
                continue
            if w.retry_at == 0.0:
                if hung:
                    w.proc.terminate()
                    w.proc.join(timeout=1.0)
                why = "stopped responding" if hung else f"exited with code {w.proc.exitcode}"
                if w.restarts >= self.max_restarts:
                    w.failed = True
                    print(f"{w.name} {why}; giving up after {w.restarts} restarts", file=sys.stderr)
                    continue
                w.retry_at = now + min(0.1 * 2**w.restarts, 5.0)  # exponential backoff
                print(f"{w.name} {why}; restarting", file=sys.stderr)
            if now >= w.retry_at:
                w.retry_at = 0.0
                w.restarts += 1
                self.stats.restarts[w.name] = w.restarts
                if w.name == "model":
                    self._busy = None
                    if self._model_conn is not None:
                        self._model_conn.close()
                    self._start_model_worker()
                else:
                    self._spawn(w)


# ===================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Noumena perception stack on simulated sensors.")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--sensors", default="camera,imu,audio,distance,motion")
    p.add_argument("--hz", type=float, default=4.0, help="periodic trigger rate")
    p.add_argument("--inline", action="store_true", help="run the model in the eye process")
    p.add_argument("--sync", type=float, default=None, help="hardware-sync all sensors to a clock at this rate")
    p.add_argument("--quiet", action="store_true", help="print only the summary")
    a = p.parse_args(argv)

    specs = [SensorSpec.simulated(m.strip(), triggered=a.sync is not None) for m in a.sensors.split(",") if m.strip()]
    eye = Eye(specs, triggers=[Periodic(a.hz), OnEvent()], inline=a.inline, clock_hz=a.sync)

    def show(packet, result, stale):
        if not a.quiet:
            print(result + (f" Stale: {', '.join(stale)}." if stale else ""), flush=True)

    with eye:
        stats = eye.run(seconds=a.seconds, on_result=show)
    print(stats.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
