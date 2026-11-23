# Noumena
Noumena: Swapping lens for machine perception. An endrun around Kant in the pursuit of giving digital minds the purpose and drive that comes with sight.

## The Universal Lens

A hardware-agnostic perception layer for local AI models. Raw, chaotic sensor output (video, audio, IMU, distance, motion vectors) goes in. Clean, fixed-shape tensors stamped on one clock come out, and the model reads them straight from shared memory.

The stack has two layers:

- **`noumena/eye.py`, the locus of control.** It owns the shared memory rings, starts and supervises one daemon process per sensor, keeps sensors in step with a hardware clock, decides when to look, and hands packets to the model.
- **`noumena/lens.py`, the interpretable focus.** It is the content of perception. Per-modality adapters crop, rectify, normalise and structure raw data into dense tensors or token streams, and the lens bundles them into the packet the model reads. The model never sees anything else.

Underneath both, **`noumena/forms.py`** holds the a priori forms. Kant's claim was that space and time are not things we find in experience but the shape experience has to take, and that categories like substance and cause are how the understanding binds it into objects and events. Here that is literal: the forms exist before any sensor is plugged in, every adapter has to project into them, and each one is a swappable object.

| Form | Default | What it does |
|---|---|---|
| Time | `MonotonicTime` | One clock for every process. Aligns streams that tick at different rates onto one instant, and drops a stream from a packet when its newest frame is too old, so a dead sensor shows up as missing rather than silently stale. |
| Space | `BodyGrid` | One body frame. Sensor poses, ray casting, and a top-down grid where cells no sensor saw stay unknown, never assumed empty. |
| Substance | `RunningBaseline` | What persists. A running mean and variance per element, per stream, so the lens can say how far the present departs from what usually is. Gaussian noise scores about 1. |
| Cause | `RefractoryCause` | What follows what. Turns departures into events, gives each stream a refractory period after it fires, and links each event to the one just before it on another stream. |

Swap any of them with `Forms(time=..., space=..., substance=..., cause=...)`.

## Run it

Requires Python 3.10 or newer and NumPy. Nothing else.

    python -m noumena                       # 3 seconds, five simulated sensors, model in its own process
    python -m noumena --sync 30             # every sensor captures on the same 30 Hz clock tick
    python -m noumena --inline --quiet      # model in the eye's process, summary only
    python -m pytest                        # 43 tests, about 10 seconds

Each simulated sensor has one scripted moment, and the run should notice all of them: a bright object enters the camera at 1.0 s, a vibration burst hits the IMU at 1.3 s, a 1 kHz tone plays from 1.6 s, something approaches the distance sensor from 1.9 s, and the camera pans at 2.2 s with one object moving against the pan. A typical run's events read: camera; imu after camera; audio after imu; distance after audio; motion after distance.

## Execution flow

    sensor daemon (one per sensor)              eye (main process)                 model worker
    source.read()                               wait on doorbells + result pipe    recv(packet)
    adapter.refract(raw, out=ring slot)  ---->  ring.latest() -> lens.apprehend   resolve SlotRefs to views
    ring.commit(); ring doorbell         bell   triggers fire -> lens.synthesize   model.perceive(percept)
                                                send packet (header only)  ----->  send result back
            ^ optional tick                     supervise: heartbeats, restarts
       clock daemon or GPIO edge

1. **Capture and refract (sensor daemon).** Each sensor runs in its own process. It reads a raw sample, asks the ring for the next slot as a writable NumPy view, and the adapter writes its output straight into that view. Refraction runs in parallel across cores, and raw data never leaves the process that captured it.
2. **Commit and ring the bell.** The daemon publishes the slot (seqlock, below) and writes one byte to a non-blocking pipe. If the eye is behind and the pipe is full, the byte is dropped; the frame is still in the ring.
3. **Wake (eye).** The eye blocks in `multiprocessing.connection.wait` on every doorbell and on the model's result pipe, with a timeout that is exactly the time until the next periodic trigger or supervision check. There is no polling loop.
4. **Apprehend.** For each stream that rang, the eye reads the newest slot as a zero-copy view and folds it into the substance form (novelty) and the cause form (events).
5. **Trigger.** Triggers decide whether to look now: `Periodic(hz)`, `OnStream("camera")` for every new frame, `OnEvent()` when something happens. Combine them freely.
6. **Synthesize.** The time form picks, for every stream, the frame in its ring nearest the reference instant. The lens bundles those choices into a `PerceptionPacket`: a `SlotRef` (ring name, commit index, timestamp) per stream, its lag, its novelty, and the events since the last packet. No tensor data is in the packet.
7. **Dispatch.** At most one packet is in flight. If triggers fire while the model is busy, they are coalesced, and the freshest data goes the moment it is free. Sensors never wait for the model.
8. **Perceive (model worker).** The worker maps the same rings, resolves each `SlotRef` to a read-only view, and calls `model.perceive(percept)`. Afterwards it checks whether any slot was overwritten while it was reading, and reports those streams as stale with the result.
9. **Supervise.** Every daemon and the model worker write a heartbeat. A process that exits or stops beating is restarted with exponential backoff, up to `max_restarts`. The ring keeps its commit count across a restart, so readers never see the stream go backwards. On shutdown every process is joined and every ring unlinked.

## The ring buffer

`noumena/ring.py`. One POSIX shared memory segment per stream (`/dev/shm` on Linux, the kernel shm namespace on macOS), one writer, any number of readers.

    [ring header 64 B][slot header 64 B x N][payload x N, each 64 B aligned]

Commit `n` goes to slot `n % N` and is guarded by a seqlock: the writer sets the slot's sequence to `2n+1` (being written), fills it, then sets `2n+2` (readable). A reader checks the sequence before and after it reads; if they differ or are not `2n+2`, the frame is dropped as torn or overwritten. Readers never block the writer. A slow reader loses old frames, never new ones, and `RingReader` counts what it missed.

Views are true zero-copy: a model reading `percept["camera"]` is reading the pages the camera daemon wrote. `Frame.valid()` tells you whether the slot is still yours; `Frame.copy()` takes a private copy and fails loudly if the slot was reused while copying. Size rings so that `slots / fastest_rate` comfortably exceeds your model's latency (the default of 16 slots holds about half a second of a 30 fps camera).

On this machine (cloud x86, Python 3.13) a begin plus commit costs about 10 microseconds, a latest-frame read about 7, and refracting a 640 by 480 camera frame to 3 by 96 by 128 about 1.7 ms. Apple Silicon and a Pi 5 should be in the same order of magnitude; that is not yet measured.

Memory ordering: Python cannot issue CPU fences. On x86 the seqlock is sound as written. On ARM (Apple Silicon, Raspberry Pi), the interpreter's own work between the stores acts as a barrier in practice, and the double check on the sequence catches anything that slips through as a torn read. A writer moved into C or Rust needs a release fence before the final sequence store. macOS limits shm names to 31 bytes, so ring names are kept short.

## The adapters

`noumena/lens.py`. Each declares `out_spec() -> (shape, dtype)` so the eye can size its ring before the first frame, and implements `refract(raw, out, forms)`.

| Adapter | Input | Output |
|---|---|---|
| `CameraAdapter` | H x W x 3 or H x W, uint8 or float | (C, H, W) float32. Undistorts with two-term radial intrinsics, centre-crops to the output aspect or an explicit ROI, resizes bilinearly, normalises per channel. The sampling map is built once. |
| `AudioAdapter` | PCM chunk, int16 or float, mono or stereo | (frames, 40) log-mel float32, or `mode="mulaw"` for (samples,) uint8 tokens. |
| `IMUAdapter` | (N, 6) accel m/s^2 and gyro rad/s | 12 floats: roll, pitch, yaw from a complementary filter; mean angular rate; linear acceleration with gravity removed; per-axis vibration. Learns gyro bias from the first samples. |
| `DistanceAdapter` | (rows, cols) multizone time-of-flight, or (N,) scan | (2, H, W) top-down grid in the body frame: where rays hit, and everything rays saw. No return means unknown. |
| `MotionAdapter` | (H, W, 2) motion vectors, for example from a Pi's H.264 encoder | (3, gh, gw): dx, dy, magnitude, with camera ego-motion (the median vector) removed. |

`quantize_uniform`, `mulaw_encode` and `mulaw_decode` turn any of these into token streams for models that read tokens.

## Plugging in a model

A model is any class with `perceive(percept)`. Load weights in an optional `setup()`, which runs once inside the worker, not in `__init__`.

    import numpy as np
    from noumena import Eye, OnEvent, Periodic, SensorSpec

    class MyModel:
        def setup(self):
            self.net = load_my_weights()            # MLX, Core ML, ONNX Runtime, llama.cpp...

        def perceive(self, percept):
            frame = percept["camera"]               # (3, 96, 128) float32, read-only, zero-copy
            events = percept.packet.events
            return self.net(frame)

    eye = Eye([SensorSpec.simulated(m) for m in ("camera", "imu", "audio")],
              model=MyModel, triggers=[Periodic(5), OnEvent()])
    with eye:
        eye.run(seconds=10, on_result=lambda packet, result, stale: print(result))

Pass `inline=True` for frameworks that must stay in the main process. Results must pickle when the model runs in its own worker.

## Plugging in real hardware

A source is any class with `read() -> RawSample`, and optionally `close()`. Give the eye a `SensorSpec` with the class and its keyword arguments (classes, not instances, so the spec survives the `spawn` start method macOS uses). For example, with OpenCV installed:

    import time, cv2
    from noumena.lens import CameraAdapter, RawSample

    class OpenCVCamera:
        def __init__(self, index=0, stream="camera"):
            self.cap, self.stream = cv2.VideoCapture(index), stream
        def read(self):
            ok, bgr = self.cap.read()
            return RawSample(self.stream, time.monotonic_ns(), bgr[..., ::-1])
        def close(self):
            self.cap.release()

    SensorSpec("camera", OpenCVCamera, CameraAdapter, adapter_kwargs={"size": (96, 128)})

Set `triggered=True` on a spec to make it capture on clock ticks instead of its own schedule. `Eye(clock_hz=30)` runs a software strobe; on a Pi, `gpio_tick(sync, pin)` fires the same tick from a GPIO edge via gpiozero. That GPIO path, and every real-hardware path, is not yet tested on hardware. All data so far is simulated.

## Layout

    noumena/ring.py     zero-copy shared memory ring buffer
    noumena/forms.py    the a priori forms: time, space, substance, cause
    noumena/lens.py     modality adapters, Lens, PerceptionPacket
    noumena/eye.py      sensor daemons, hardware sync, triggers, supervision, dispatch, CLI
    noumena/sensors.py  simulated sources
    tests/              ring, forms, adapters, and end-to-end eye tests
