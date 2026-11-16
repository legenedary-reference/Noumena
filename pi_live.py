"""
Noumena on a Raspberry Pi: read depth frames in real time, run the attention lens,
and say what it notices.

Pipeline:  sensor -> depth grid -> Lens.look() -> Tracker -> print (and speak)

Sensors:
  --fake        replays the made-up loading dock from lens.py. Runs anywhere; use it to test.
  --vl53l5cx    Pimoroni VL53L5CX time-of-flight sensor: an 8 by 8 grid of distances, up to 15 per second.
                Each of its 64 zones becomes one lens cell. NOT YET TESTED ON HARDWARE.

Add --speak to have the Pi read events aloud with espeak-ng (sudo apt install espeak-ng).

Examples:
  python3 pi_live.py --fake
  python3 pi_live.py --vl53l5cx --speak
"""

import argparse
import subprocess
import sys
import time

import numpy as np

import lens as L


def fake_frames(rate):
    rng = np.random.default_rng(7)
    for img in L.make_scene(rng):
        yield img
        time.sleep(1 / rate)


def vl53l5cx_frames(rate):
    # Pimoroni library: pip install vl53l5cx-ctypes  (I2C must be enabled in raspi-config)
    from vl53l5cx_ctypes import VL53L5CX, RANGING_MODE_CONTINUOUS
    sensor = VL53L5CX()
    sensor.set_resolution(8 * 8)
    sensor.set_ranging_frequency_hz(int(rate))
    sensor.set_ranging_mode(RANGING_MODE_CONTINUOUS)
    sensor.start_ranging()
    while True:
        if sensor.data_ready():
            data = sensor.get_data()
            mm = np.array(data.distance_mm[0][:64], dtype=float).reshape(8, 8)
            status = np.array(data.target_status[0][:64]).reshape(8, 8)
            mm[(status != 5) & (status != 9)] = np.nan     # unreliable zones become unknown (ghosts)
            yield mm / 1000.0                               # metres
        time.sleep(0.005)


def speak(text):
    subprocess.Popen(["espeak-ng", "-s", "165", text],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description="Run the Noumena attention lens live.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="replay the fake loading dock")
    src.add_argument("--vl53l5cx", action="store_true", help="read a VL53L5CX 8x8 depth sensor")
    ap.add_argument("--rate", type=float, default=5, help="snapshots per second (default 5)")
    ap.add_argument("--speak", action="store_true", help="read events aloud with espeak-ng")
    args = ap.parse_args()

    if args.vl53l5cx:
        # one sensor zone per lens cell: an 8 by 8 grid
        L.GRID, L.CELL, L.SIZE, L.ROWS = 8, 1, 8, "ABCDEFGH"
        frames = vl53l5cx_frames(args.rate)
    else:
        frames = fake_frames(args.rate)

    lens, tracker = L.Lens(), L.Tracker()
    print(f"Noumena live. {L.GRID} by {L.GRID} grid at {args.rate:g} snapshots a second. "
          f"Press Control C to stop.", flush=True)
    try:
        for t, img in enumerate(frames):
            events, weight = lens.look(img, t)
            figures = [L.describe(f) for f in L.group(weight > L.THRESHOLD)]
            events += tracker.update(figures, t)
            for e in events:
                print(f"[{time.strftime('%H:%M:%S')}] {e}", flush=True)
                if args.speak:
                    speak(e)
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    sys.exit(main())
