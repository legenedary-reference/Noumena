"""
Gestalt: fills the lens's 3D space with objects.

The lens says, voxel by voxel, what is full, empty, or unknown. Gestalt does the rest,
in the spirit of Whitehead's process philosophy: the world is made of occasions
(things happening), and an object is a pattern of occasions that keeps going.

  1. Ground   learn what never moves (machines, walls), so only figures are left.
  2. Figures  join touching full voxels into wholes.
  3. Objects  follow each whole from snapshot to snapshot and give it an identity.
              If its voxels go unknown, it becomes a ghost: held where it was last seen.
  4. Occasions  every change in an object's life becomes one row, and each row
              points to the occasion before it (caused_by), so every event carries its past.
  5. Flow     from the occasions alone, find where objects routinely stop (stations),
              how long they stay, and where they wait when they shouldn't.

Gestalt is never told where the stations are or what the objects are. It works that out.
"""

import csv
from collections import defaultdict

import numpy as np

NEIGHBOURS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def components(mask):
    """Groups of touching voxels: wholes before parts."""
    seen, out = np.zeros(mask.shape, bool), []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            v = stack.pop()
            members.append(v)
            for d in NEIGHBOURS:
                n = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
                if all(0 <= n[i] < mask.shape[i] for i in range(3)) and mask[n] and not seen[n]:
                    seen[n] = True
                    stack.append(n)
        out.append(np.array(members))
    return out


class Gestalt:
    def __init__(self, space, ground_frames=8, match_dist=2.5, still_after=2, move_step=0.6):
        self.space = space
        self.ground_frames, self.match_dist = ground_frames, match_dist
        self.still_after, self.move_step = still_after, move_step
        self.ground_count = np.zeros(space.shape, int)
        self.frames_seen = 0
        self.ground = None
        self.objects = {}
        self.occasions = []
        self.next_id = 1
        self.tentative = []        # wholes seen once; real things persist, noise does not

    # ------------------------------------------------------------ occasions
    def _occur(self, t, obj, what, detail=""):
        x, y, z = obj["pos"]
        row = {"id": len(self.occasions) + 1, "t": t, "object": f"Object {obj['id']}", "occasion": what,
               "cube": self.space.address(x, y, z).split()[-1], "x": round(x, 1), "y": round(y, 1),
               "z": round(z, 1), "detail": detail, "caused_by": obj.get("last_occ") or ""}
        obj["last_occ"] = row["id"]
        self.occasions.append(row)
        return row

    # ------------------------------------------------------------ one snapshot
    def fill(self, state, t):
        if self.ground is None:                          # still learning what never moves
            self.ground_count += (state == 1)
            self.frames_seen += 1
            if self.frames_seen == self.ground_frames:
                self.ground = self.ground_count >= 0.9 * self.ground_frames
            return []

        new = []
        figures = [(vox, vox.mean(axis=0) + 0.5) for vox in components((state == 1) & ~self.ground)]
        live = [o for o in self.objects.values() if o["status"] in ("visible", "ghost")]

        # match each figure to the nearest known object (greedy, closest pairs first)
        pairs = sorted(((np.linalg.norm(c - o["pos"]), i, o["id"])
                        for i, (_, c) in enumerate(figures) for o in live), key=lambda p: p[0])
        used_f, used_o = set(), set()
        for d, i, oid in pairs:
            if d > self.match_dist or i in used_f or oid in used_o:
                continue
            used_f.add(i)
            used_o.add(oid)
            new += self._update(self.objects[oid], figures[i], t)

        # wholes nobody knows: believe them only if they were also there last snapshot
        keep = []
        for i, (vox, c) in enumerate(figures):
            if i in used_f:
                continue
            prior = [p for p in self.tentative if np.linalg.norm(c - p[1]) <= self.match_dist]
            if prior:
                obj = {"id": self.next_id, "pos": c, "vox": vox, "status": "visible", "moving": True,
                       "still": 0, "cube": self.space.address(*c), "last_seen": t}
                self.objects[self.next_id] = obj
                self.next_id += 1
                new.append(self._occur(t - 1, obj, "appeared", f"in {obj['cube']}"))
            else:
                keep.append((vox, c))
        self.tentative = keep

        for o in live:                                   # known objects not found this time
            if o["id"] in used_o or o["status"] != "visible":
                continue
            v = o["vox"]
            last = state[v[:, 0], v[:, 1], v[:, 2]]
            if (last == -1).mean() >= 0.5:
                o["status"] = "ghost"
                new.append(self._occur(t, o, "went out of sight",
                                       "its voxels went unknown; holding it as a ghost where it was last seen"))
            else:
                o["status"] = "gone"
                at_end = o["pos"][0] >= self.space.shape[0] - 1.5
                new.append(self._occur(t, o, "left the line" if at_end else "vanished",
                                       "" if at_end else "it was in plain sight and is now gone"))
        return new

    def _update(self, o, figure, t):
        vox, c = figure
        out = []
        if o["status"] == "ghost":
            gone = t - o["last_seen"]
            moved = np.linalg.norm(c - o["pos"])
            o["status"] = "visible"
            out.append(self._occur(t, o, "seen again", f"after {gone} snapshots out of sight, "
                                   + ("in the same place" if moved < self.move_step else f"{moved:.1f} voxels away")))
        step = c[0] - o["pos"][0]
        o["pos"], o["vox"], o["last_seen"] = c, vox, t
        cube = self.space.address(*c)
        if cube != o["cube"]:
            o["cube"] = cube
            out.append(self._occur(t, o, "entered", cube))
        if abs(step) >= self.move_step:
            if not o["moving"]:
                still = t - o.get("stop_t", t - o["still"])
                out.append(self._occur(t, o, "started moving", f"after {still} snapshots still"))
            o["moving"], o["still"] = True, 0
        else:
            o["still"] += 1
            if o["moving"] and o["still"] == self.still_after:
                o["moving"] = False
                o["stop_t"] = t - self.still_after + 1
                out.append(self._occur(o["stop_t"], o, "stopped", ""))
        return out

    # ------------------------------------------------------------ flow
    def flow(self):
        """Find stations and waits from the occasions alone."""
        stops = defaultdict(list)          # object -> list of (stop row, start row)
        open_stop = {}
        for r in self.occasions:
            if r["occasion"] == "stopped":
                open_stop[r["object"]] = r
            elif r["occasion"] in ("started moving", "left the line") and r["object"] in open_stop:
                s = open_stop.pop(r["object"])
                stops[r["object"]].append((s, r["t"] - s["t"]))
        spot_objects = defaultdict(set)
        for obj, lst in stops.items():
            for s, _ in lst:
                spot_objects[round(s["x"])].add(obj)
        n_obj = len({r["object"] for r in self.occasions})
        stations = sorted(x for x, objs in spot_objects.items() if len(objs) >= 0.6 * n_obj)
        dwell, waits = defaultdict(list), []
        for obj, lst in stops.items():
            for s, dur in lst:
                spot = round(s["x"])
                if spot in stations:
                    dwell[spot].append((dur, obj, s))
                else:
                    waits.append((dur, obj, s))
        return stations, dwell, waits

    def save(self, path):
        cols = ["id", "t", "object", "occasion", "cube", "x", "y", "z", "detail", "caused_by"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.occasions)
