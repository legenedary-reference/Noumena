"""
Noumena: attention lens prototype.

A depth sensor watches a scene. The lens splits each snapshot into a 7 by 7 grid
and stays quiet unless a cell earns attention. Each cell gets a weight:

    weight = change x importance x surprise

    change      how much the depth in that cell moved since the last snapshot
    importance  how much that cell matters (stations high, walls low); learned from feedback
    surprise    how unusual that change is for that cell, compared with its own history

A slow-drift check catches things that change too gradually for frame-to-frame change,
like a pile growing. Cells the sensor can't read become ghosts: unknown, not empty.

Gestalt then groups neighbouring hot cells into figures, so five hot cells become
one object, and follows each figure from snapshot to snapshot.

Everything here is fake data. Run:  python lens.py
Output is plain text, written for a screen reader.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

GRID = 7
CELL = 10                      # pixels per cell side; image is 70 x 70
SIZE = GRID * CELL
FRAMES = 60
THRESHOLD = 0.3                # attention weight needed to speak
DRIFT_WINDOW = 15              # frames back for the slow-drift check
DRIFT_LIMIT = 0.35             # metres of steady change that counts as drift
DRIFT_STEADY = 0.9             # how straight the trend must be (R squared): a pile grows steadily, a cart passing does not
ROWS = "ABCDEFG"


def cell_name(r, c):
    return f"{ROWS[r]}{c + 1}"


# ---------------------------------------------------------------- fake scene
def make_scene(rng):
    """A loading dock seen from above. Depth in metres from the sensor."""
    floor = np.full((SIZE, SIZE), 4.0)
    floor[:, :8] = 1.0                        # wall on the left (close to sensor)
    floor[10:60, 52:62] = 2.5                 # shelving rack on the right

    frames = []
    for t in range(FRAMES):
        img = floor.copy()

        # a cart crossing the dock left to right, frames 10 to 24
        if 10 <= t <= 24:
            x = 10 + int((t - 10) * 2.6)
            img[30:38, x:x + 8] = 2.4

        # a pallet stack slowly growing in cell E3 (taller = closer to sensor)
        h = min(1.2, 0.03 * t)
        img[40:50, 20:30] = 4.0 - h

        # a flickering light in cell A6: throws off the whole cell every frame, means nothing
        img[0:10, 50:60] += rng.normal(0, 0.4)

        # sensor dropout over cells F6 and G6 from frame 35 on
        if t >= 35:
            img[50:70, 50:60] = np.nan

        # a person steps into the doorway at B2, frames 44 to 47
        if 44 <= t <= 47:
            img[10:20, 10:20] = 2.2

        img += rng.normal(0, 0.01, img.shape)   # ordinary sensor noise
        frames.append(img)
    return frames


# ---------------------------------------------------------------- the lens
def cell_view(img):
    """Mean depth per cell, and how much of each cell is unreadable."""
    blocks = img.reshape(GRID, CELL, GRID, CELL).swapaxes(1, 2)
    unknown = np.isnan(blocks).mean(axis=(2, 3))
    with np.errstate(all="ignore"):
        depth = np.nanmean(blocks, axis=(2, 3))
    return depth, unknown


def starting_importance():
    imp = np.full((GRID, GRID), 0.6)
    imp[:, 0] = 0.2                     # the wall
    imp[1:6, 5] = 1.0                   # the shelving rack
    imp[3:5, 1:5] = 1.0                 # the dock floor where carts and pallets move
    return imp


class Lens:
    def __init__(self):
        self.importance = starting_importance()
        self.mean = np.full((GRID, GRID), 0.05)       # each cell's usual change
        self.var = np.full((GRID, GRID), 0.0)         # and how much that usual change swings
        self.history = []
        self.ghost_since = {}
        self.drifting = set()

    def look(self, img, t):
        depth, unknown = cell_view(img)
        events = []

        # ghosts: cells the sensor can't read are unknown, not empty
        for r in range(GRID):
            for c in range(GRID):
                name = cell_name(r, c)
                if unknown[r, c] > 0.5 and name not in self.ghost_since:
                    self.ghost_since[name] = t
                    events.append(f"{name} went dark. Holding its last known state as a ghost.")
                elif unknown[r, c] <= 0.5 and name in self.ghost_since:
                    gone = t - self.ghost_since.pop(name)
                    events.append(f"{name} is visible again after {gone} snapshots.")

        if not self.history:
            self.history.append(depth)
            return events, np.zeros((GRID, GRID))

        prev = self.history[-1]
        change = np.nan_to_num(np.abs(depth - prev), nan=0.0)
        surprise = change / (self.mean + 2 * np.sqrt(self.var) + 0.02)
        weight = change * self.importance * np.minimum(surprise, 10)

        # slow drift: a steady trend over the last DRIFT_WINDOW snapshots.
        # A straight line means something is building up; a bump means something passed by.
        if len(self.history) >= DRIFT_WINDOW and t % 5 == 0:
            y = np.array(self.history[-DRIFT_WINDOW:])            # window x 7 x 7
            x = np.arange(DRIFT_WINDOW)[:, None, None] - (DRIFT_WINDOW - 1) / 2
            dy = y - y.mean(axis=0)
            slope = (x * dy).sum(axis=0) / (x ** 2).sum()
            r2 = (x * dy).sum(axis=0) ** 2 / ((x ** 2).sum() * (dy ** 2).sum(axis=0) + 1e-9)
            total = np.abs(slope) * (DRIFT_WINDOW - 1)
            steady = (total > DRIFT_LIMIT) & (r2 > DRIFT_STEADY)
            now = {cell_name(r, c) for r, c in zip(*np.nonzero(steady))}
            for r, c in zip(*np.nonzero(steady)):
                if cell_name(r, c) not in self.drifting:
                    way = "rising toward the sensor" if slope[r, c] < 0 else "sinking away"
                    events.append(f"Slow drift at {cell_name(r, c)}: {total[r, c]:.2f} m over "
                                  f"{DRIFT_WINDOW} snapshots, steadily {way}. Something is building up.")
            for name in sorted(self.drifting - now):
                events.append(f"{name} has stopped building up.")
            self.drifting = now

        # learn each cell's normal amount of change and its normal swing,
        # so routine noise stops surprising
        self.var = 0.9 * self.var + 0.1 * (change - self.mean) ** 2
        self.mean = 0.9 * self.mean + 0.1 * change
        self.history.append(np.where(np.isnan(depth), prev, depth))
        return events, weight

    def feedback(self, cell, real):
        """The person says whether a zoom was worth it. Importance learns."""
        r, c = ROWS.index(cell[0]), int(cell[1:]) - 1
        self.importance[r, c] = min(1.5, self.importance[r, c] * (1.3 if real else 0.6))


# ---------------------------------------------------------------- gestalt
def group(hot):
    """Join neighbouring hot cells into figures (wholes before parts)."""
    seen, figures = set(), []
    for r, c in zip(*np.nonzero(hot)):
        if (r, c) in seen:
            continue
        stack, members = [(r, c)], []
        seen.add((r, c))
        while stack:
            a, b = stack.pop()
            members.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (a + da, b + db)
                if 0 <= n[0] < GRID and 0 <= n[1] < GRID and hot[n] and n not in seen:
                    seen.add(n)
                    stack.append(n)
        figures.append(sorted(members))
    return figures


def describe(figure):
    cells = ", ".join(cell_name(r, c) for r, c in figure)
    cr = np.mean([r for r, _ in figure])
    cc = np.mean([c for _, c in figure])
    return cells, (cr, cc)


class Tracker:
    """Follows figures over time. Speaks when one appears and when it stops, quiet in between."""
    GAP = 4          # snapshots a figure can go unseen before its track ends

    def __init__(self):
        self.tracks = []

    def update(self, figures, t):
        events = []
        for cells, (cr, cc) in figures:
            live = [k for k in self.tracks if t - k["last"] <= self.GAP]
            near = min(live, key=lambda k: abs(k["pos"][0] - cr) + abs(k["pos"][1] - cc), default=None)
            if near and abs(near["pos"][0] - cr) + abs(near["pos"][1] - cc) <= 1.5:
                near.update(pos=(cr, cc), last=t)
                near["cells"].append(cells)
            else:
                self.tracks.append({"start": t, "last": t, "first": (cr, cc), "pos": (cr, cc),
                                    "cells": [cells]})
                events.append(f"New figure at {cells}.")
        for k in [k for k in self.tracks if t - k["last"] == self.GAP + 1]:
            events.append(self.story(k))
        return events

    @staticmethod
    def story(k):
        dc = k["pos"][1] - k["first"][1]
        dr = k["pos"][0] - k["first"][0]
        start, end = k["cells"][0].split(", ")[0], k["cells"][-1].split(", ")[-1]
        span = f"snapshots {k['start']} to {k['last']}"
        if abs(dc) < 1 and abs(dr) < 1:
            return f"Figure at {start} has settled. It was active {span} without moving."
        way = ("right" if dc > 0 else "left") if abs(dc) >= abs(dr) else ("down" if dr > 0 else "up")
        return f"Figure that started at {start} moved {way} to {end} ({span}) and has stopped."


# ---------------------------------------------------------------- 3D space
# The lens makes the space. Cameras sit at the corners of cubes laid end to end
# along a process line (1 cube = 8 cameras, 3 cubes = 16, 15 cubes = 64).
# Every voxel becomes full, empty, or unknown. Unknown means fewer than two
# cameras could see it, so it is held as a ghost, never assumed empty.

class Camera:
    def __init__(self, cid, pos, look_at, fov_deg=120, reach=6.0):
        self.cid = cid
        self.pos = np.array(pos, float)
        d = np.array(look_at, float) - self.pos
        self.dir = d / np.linalg.norm(d)
        self.cos_half = np.cos(np.radians(fov_deg / 2))
        self.reach = reach


def corridor_cameras(cubes, edge=4):
    """Cameras at every cube corner, each looking at the centre of the nearest cube section."""
    cams, n = [], 0
    for i in range(cubes + 1):
        x = i * edge
        cx = min(max(x, edge / 2), cubes * edge - edge / 2)   # end posts look into their own cube
        for y in (0, edge):
            for z in (0, edge):
                n += 1
                cams.append(Camera(n, (x, y, z), (cx, edge / 2, edge / 2)))
    return cams


class Space:
    MIN_VIEWS = 2      # a voxel needs this many cameras to count as known

    def __init__(self, shape, cameras, edge=4):
        self.shape, self.cameras, self.edge = shape, cameras, edge
        idx = np.indices(shape).reshape(3, -1).T            # every voxel (x, y, z)
        centres = idx + 0.5
        self.view = {}
        for cam in cameras:
            v = centres - cam.pos
            dist = np.linalg.norm(v, axis=1)
            inside = (dist <= cam.reach) & ((v @ cam.dir) / dist >= cam.cos_half)
            self.view[cam.cid] = idx[inside]
        self.coverage = np.zeros(shape, int)

    def observe(self, occupied, online, rng=None, flip=0.0):
        """Fuse what every online camera sees into one voxel map: 1 full, 0 empty, -1 unknown."""
        seen = np.zeros(self.shape, int)
        full = np.zeros(self.shape, int)
        steps = np.linspace(0.08, 0.92, 14)[None, :, None]
        for cam in self.cameras:
            if cam.cid not in online:
                continue
            vox = self.view[cam.cid]
            # occlusion: walk the line from the camera to each voxel; anything full in between blocks it
            pts = cam.pos[None, None, :] + steps * ((vox + 0.5) - cam.pos)[:, None, :]
            cell = np.floor(pts).astype(int)
            ok = np.all((cell >= 0) & (cell < np.array(self.shape)), axis=2)
            same = np.all(cell == vox[:, None, :], axis=2)
            cl = np.clip(cell, 0, np.array(self.shape) - 1)
            blocked = (occupied[cl[..., 0], cl[..., 1], cl[..., 2]] & ok & ~same).any(axis=1)
            vis = vox[~blocked]
            reading = occupied[vis[:, 0], vis[:, 1], vis[:, 2]].astype(int)
            if rng is not None and flip:
                reading ^= (rng.random(len(reading)) < flip).astype(int)
            np.add.at(seen, (vis[:, 0], vis[:, 1], vis[:, 2]), 1)
            np.add.at(full, (vis[:, 0], vis[:, 1], vis[:, 2]), reading)
        state = np.where(full * 2 > seen, 1, 0)
        state[seen < self.MIN_VIEWS] = -1
        self.coverage = seen
        return state

    def address(self, x, y, z):
        return f"Cube {int(x) // self.edge + 1}"


# ---------------------------------------------------------------- run
def main():
    rng = np.random.default_rng(7)
    frames = make_scene(rng)
    lens = Lens()
    tracker = Tracker()
    flicker_alerts = []
    dismissals = 0
    spoke = 0

    print("Noumena attention lens. 7 by 7 grid, rows A to G top to bottom, columns 1 to 7 left to right.")
    print(f"{FRAMES} snapshots of a loading dock. The lens speaks only when something earns attention.\n")

    for t, img in enumerate(frames):
        events, weight = lens.look(img, t)
        hot = weight > THRESHOLD
        figures = [describe(f) for f in group(hot)]

        events += tracker.update(figures, t)
        if cell_name(0, 5) in " ".join(c for c, _ in figures):
            flicker_alerts.append(t)

        if events:
            spoke += 1
            print(f"Snapshot {t}:")
            for e in events:
                print("  " + e)

        # simulated administrator: dismisses every new alert at A6 as a false alarm
        if any(e == f"New figure at {cell_name(0, 5)}." for e in events):
            lens.feedback("A6", real=False)
            dismissals += 1
            print(f"  Feedback: A6 is a false alarm (dismissal {dismissals}). A6 importance goes down.")

    quiet = FRAMES - spoke
    print(f"\nSummary: the lens spoke on {spoke} of {FRAMES} snapshots and stayed quiet on {quiet}.")
    flicker = cell_name(0, 5)
    when = ", ".join(map(str, flicker_alerts)) or "never"
    last = flicker_alerts[-1] if flicker_alerts else None
    print(f"The flickering light at {flicker} changed every snapshot. It was dismissed {dismissals} times; "
          f"its last alert was snapshot {last}, and the lens stayed quiet about it for the "
          f"{FRAMES - 1 - last} snapshots after that.")
    print("Ghost cells still dark at the end: " + (", ".join(sorted(lens.ghost_since)) or "none") + ".")


if __name__ == "__main__":
    main()
