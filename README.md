# Noumena
Noumena: Swapping lens for machine perception. An endrun around Kant in the pursuit of giving digital minds the purpose and drive that comes with sight.

## What's here

**lens.py**: the attention lens. A depth sensor watches a scene, and the lens splits each snapshot into a grid. Each cell gets a weight, and the lens only speaks when a cell earns it:

    weight = change x importance x surprise

- **change**: how much the depth in a cell moved since the last snapshot
- **importance**: how much the cell matters; it learns from feedback ("that was a false alarm")
- **surprise**: how unusual the change is for that cell, compared with its own history

On top of that:

- **Gestalt**: neighbouring hot cells are grouped into one figure, then followed over time. The lens says when a figure appears and when it stops, and stays quiet in between.
- **Slow drift**: a steady trend over 15 snapshots means something is building up, like a pile growing. A bump means something just passed by.
- **Ghosts**: cells the sensor can't read are unknown, not empty. The lens holds their last known state.

Run it: `python lens.py`. It plays 60 snapshots of a fake loading dock (a cart crossing, a pallet stack growing, a flickering light, a sensor dropout, a person in a doorway) and prints what it notices in plain text.

**pi_live.py**: runs the same lens in real time on a Raspberry Pi.

    python3 pi_live.py --fake               # test anywhere with the fake dock
    python3 pi_live.py --vl53l5cx --speak   # 8x8 time-of-flight sensor, events read aloud

The VL53L5CX path is not yet tested on hardware.

## Gestalt: the lens makes the space, Gestalt fills it with things

In the spirit of Whitehead's process philosophy: the world is made of occasions (things happening), and an object is a pattern of occasions that keeps going.

**lens.py (3D):** cameras sit at the corners of cubes laid end to end along a process line. A row of cubes has 4 x (cubes + 1) corners, so 1 cube uses 8 cameras, 3 cubes use 16, 7 cubes use 32, and 15 cubes use 64. Every voxel is fused into full, empty, or unknown. Unknown means fewer than two cameras can see it, and it is never assumed empty.

**gestalt.py:** fills that space with objects.
1. **Ground:** learns what never moves (machines, walls).
2. **Figures:** joins touching voxels into wholes. A new whole only counts once it's been seen twice, so sensor noise doesn't become objects.
3. **Objects:** follows each whole over time. If it goes unknown, it becomes a ghost held where it was last seen.
4. **Occasions:** every change in an object's life is one row, and each row points to the occasion before it (`caused_by`).
5. **Flow:** from the occasions alone, finds the stations (where objects routinely stop), how long they stay, jams, and queues.

**sim.py:** a 3-cube, 16-camera process line with 7 totes, a jam at the middle station, and 8 cameras losing power mid-jam. Gestalt is told none of it.

    python sim.py      # prints what Gestalt found, checks it against the truth, writes occasions.csv

Current result: 7 of 7 objects, all 3 stations found, every dwell time exact, the jam flagged, and the jammed tote held as a ghost through the outage and re-identified in the same place.

Query the occasions with SQL (DuckDB):

    SELECT t, occasion, cube, detail FROM 'occasions.csv' WHERE object = 'Object 3' ORDER BY t;

Requires Python 3 and NumPy. All data so far is fake.
