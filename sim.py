"""
A fake process line to test the lens and Gestalt together.

Three cubes in a row (16 cameras at the corners). Each cube is a station with a
machine against the back wall. Totes roll along the floor, stop at each station,
then move on. Two things go wrong:
  - one tote jams at the middle station, and the totes behind it queue up
  - while it is jammed, the two middle camera posts (8 of 16 cameras) lose power

Gestalt is not told any of this. It only gets the voxel map from the lens each snapshot.
At the end, its findings are checked against what really happened.

Run:  python sim.py     (writes occasions.csv)
"""

import numpy as np

import lens as L
from gestalt import Gestalt

EDGE, CUBES = 4, 3
SHAPE = (EDGE * CUBES, 4, 4)
STATION_NAMES = {1: "Receive", 2: "Process", 3: "Pack"}
DWELL = {1: 5, 2: 7, 3: 4}            # normal snapshots spent at each station
JAM = (3, 2, 26)                      # tote 3 sits 26 snapshots at station 2
TOTES, FIRST, EVERY, FRAMES = 7, 10, 12, 160
FLIP = 0.01                           # each camera misreads 1% of voxels


def station_front(k):
    return (k - 1) * EDGE + 2         # a stopped tote fills x = s+1 and s+2 of its cube


def run():
    rng = np.random.default_rng(11)
    cams = L.corridor_cameras(CUBES, EDGE)
    space = L.Space(SHAPE, cams, EDGE)
    g = Gestalt(space)

    machines = np.zeros(SHAPE, bool)
    for k in range(CUBES):
        machines[k * EDGE + 1:k * EDGE + 3, 3, 0:2] = True

    totes = [{"n": i + 1, "front": None, "enter": FIRST + i * EVERY, "next": 1, "dwell": 0, "done": False}
             for i in range(TOTES)]
    truth = {i + 1: {} for i in range(TOTES)}       # tote -> station -> [arrive, leave]
    outage = None
    middle_posts = {c.cid for c in cams if 0 < c.pos[0] < SHAPE[0]}
    every_cam = {c.cid for c in cams}
    log = []

    for t in range(FRAMES):
        # ---- move the totes (front-most first so queues behave)
        for tt in sorted(totes, key=lambda x: -(x["front"] if x["front"] is not None else -99)):
            if tt["done"] or t < tt["enter"]:
                continue
            if tt["front"] is None:
                backs = [o["front"] - 1 for o in totes if o is not tt and not o["done"] and o["front"] is not None]
                if backs and min(backs) <= 1:            # entrance blocked: wait outside the line
                    continue
                tt["front"] = 0
                continue
            if tt["dwell"] > 0:
                tt["dwell"] -= 1
                if tt["dwell"] == 0:
                    truth[tt["n"]][tt["next"] - 1][1] = t
                continue
            nxt = tt["front"] + 1
            ahead = [o["front"] - 1 for o in totes if o is not tt and not o["done"]
                     and o["front"] is not None and o["front"] > tt["front"]]
            if ahead and min(ahead) <= nxt + 1:          # keep a one-voxel gap; queue up
                continue
            tt["front"] = nxt
            if tt["next"] <= CUBES and nxt == station_front(tt["next"]):
                k = tt["next"]
                stay = JAM[2] if (tt["n"], k) == JAM[:2] else DWELL[k]
                tt["dwell"], tt["next"] = stay, k + 1
                truth[tt["n"]][k] = [t, None]
                if (tt["n"], k) == JAM[:2]:
                    outage = (t + 6, t + 18)
            if tt["front"] - 1 >= SHAPE[0]:
                tt["done"] = True

        # ---- what is really there
        occ = machines.copy()
        for tt in totes:
            if tt["front"] is not None and not tt["done"]:
                for x in (tt["front"] - 1, tt["front"]):
                    if 0 <= x < SHAPE[0]:
                        occ[x, 1, 0] = True

        online = every_cam - middle_posts if outage and outage[0] <= t < outage[1] else every_cam
        if outage and t == outage[0]:
            log.append((t, f"Power lost to the two middle camera posts: {len(middle_posts)} of {len(every_cam)} cameras dark."))
        if outage and t == outage[1]:
            log.append((t, "Middle camera posts back online."))

        state = space.observe(occ, online, rng, FLIP)
        for row in g.fill(state, t):
            log.append((row["t"], row))
    return g, truth, log, outage


def main():
    g, truth, log, outage = run()
    stations, dwell, waits = g.flow()
    g.save("occasions.csv")

    print("Gestalt: 3 cubes, 16 cameras, 160 snapshots of a process line.")
    print("Gestalt was not told where stations are or what moves. Everything below it worked out itself.\n")

    print("What Gestalt found:")
    objects = sorted({r["object"] for r in g.occasions}, key=lambda s: int(s.split()[1]))
    print(f"  {len(objects)} objects passed along the line. The machines never moved, so they became ground.")
    names = [f"x = {s}" for s in stations]
    print(f"  Objects routinely stop at {len(stations)} spots ({', '.join(names)}). It treats those as stations.")
    for i, s in enumerate(stations, 1):
        d = sorted(dwell[s], key=lambda v: v[0])
        typical = int(np.median([v[0] for v in d]))
        worst = d[-1]
        line = f"  Station {i}: typical stop {typical} snapshots."
        if worst[0] >= 2 * typical:
            line += f" {worst[1]} stopped {worst[0]}, more than {worst[0] // typical} times typical. That's a jam."
        print(line)
    if waits:
        total = sum(w[0] for w in waits)
        who = sorted({w[1] for w in waits}, key=lambda s: int(s.split()[1]))
        print(f"  Stops away from any station: {len(waits)}, totalling {total} snapshots of waiting, "
              f"by {', '.join(who)}. They were queued behind the jam.")

    print("\nKey moments:")
    for t, e in log:
        if isinstance(e, str):
            print(f"  Snapshot {t}: {e}")
        elif e["occasion"] in ("went out of sight", "seen again", "vanished"):
            print(f"  Snapshot {t}: {e['object']} {e['occasion']} in Cube {e['cube']}: {e['detail']}.")

    print("\nCheck against what really happened:")
    real = {k: [] for k in STATION_NAMES}
    for tote in truth.values():
        for k, (a, b) in tote.items():
            if b is not None:
                real[k].append(b - a)
    for i, s in enumerate(stations, 1):
        cube = int(s) // EDGE + 1
        found = sorted(v[0] for v in dwell[s])
        print(f"  Station {i} is in cube {cube}, which is really {STATION_NAMES[cube]}. "
              f"Gestalt measured stops of {found}; the true dwells were {sorted(real[cube])}.")
    print(f"  Totes in the simulation: {len(truth)}. Objects Gestalt found: {len(objects)}.")
    print(f"\nWrote {len(g.occasions)} occasions to occasions.csv. Each one points to the occasion before it.")


if __name__ == "__main__":
    main()
