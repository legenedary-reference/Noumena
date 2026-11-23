"""
Zero-copy shared memory ring buffer.

One writer process, any number of reader processes. Each ring lives in a POSIX
shared memory segment (``/dev/shm/<name>`` on Linux, the kernel's shm namespace
on macOS), so the bytes a sensor daemon writes are the same physical pages a
model worker reads. Nothing is pickled or copied on the way.

Layout of one segment::

    +--------------------+  offset 0
    | ring header (64 B) |  magic, version, slot count, slot capacity, commits
    +--------------------+  offset 64
    | slot header x N    |  64 B each: seqlock, commit index, timestamp,
    |                    |  dtype, shape, byte count
    +--------------------+  offset 64 + 64*N, rounded up to 64
    | payload x N        |  slot_bytes each, every slot starts 64 B aligned
    +--------------------+

Every write is commit number ``n``; it lands in slot ``n % N``. Each slot is
guarded by a seqlock:

    writer:  seq = 2n+1  (odd: being written)
             write payload and metadata
             seq = 2n+2  (even: commit n is readable)
             ring.commits = n+1

    reader:  s1 = seq;  read;  s2 = seq;  the read is good only if s1 == s2 == 2n+2

A reader never blocks the writer. If the writer laps a slow reader, the reader
sees a newer sequence number, drops the stale frame, and counts the overrun.

A note on memory ordering: Python cannot issue CPU fences. On x86 stores are
not reordered with other stores, so the seqlock is sound. On ARM (Apple
Silicon, Raspberry Pi) CPython's own interpreter work between the stores
almost always acts as a barrier, and the double check on ``seq`` catches the
rest as a torn read, which the reader drops. If you move the writer into C or
Rust, put a release fence before the final ``seq`` store.
"""

from __future__ import annotations

import os
import secrets
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

MAGIC = 0x414E4D4E  # "NMNA"
VERSION = 1
ALIGN = 64
MAX_DIMS = 4
# macOS limits POSIX shm names to 31 bytes including the leading slash.
MAX_NAME = 30

RING_DT = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u2"),
        ("flags", "<u2"),
        ("slot_count", "<u4"),
        ("slot_bytes", "<u4"),
        ("payload_offset", "<u8"),
        ("commits", "<u8"),
        ("writer_pid", "<u4"),
        ("_pad", "V28"),
    ]
)
SLOT_DT = np.dtype(
    [
        ("seq", "<u8"),
        ("index", "<u8"),
        ("t_ns", "<i8"),
        ("dtype", "<u1"),
        ("ndim", "<u1"),
        ("flags", "<u2"),
        ("nbytes", "<u4"),
        ("shape", "<u4", (MAX_DIMS,)),
        ("_pad", "V16"),
    ]
)
assert RING_DT.itemsize == ALIGN and SLOT_DT.itemsize == ALIGN

DTYPES = [
    np.dtype(t)
    for t in ("u1", "i1", "<u2", "<i2", "<u4", "<i4", "<i8", "<f2", "<f4", "<f8", "?")
]
DTYPE_CODE = {dt: i for i, dt in enumerate(DTYPES)}


class RingError(RuntimeError):
    pass


def _round_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def make_name(stream: str, run: str | None = None) -> str:
    """A short, unique, macOS-safe shared memory name for one stream."""
    run = run or secrets.token_hex(3)
    safe = "".join(c for c in stream if c.isalnum())[:14]
    name = f"nm{run}_{safe}"
    return name[:MAX_NAME]


def _open_shm(name: str, create: bool, size: int = 0) -> shared_memory.SharedMemory:
    if sys.version_info >= (3, 13):
        # track=False: only the creator unlinks. Readers must not.
        return shared_memory.SharedMemory(name=name, create=create, size=size, track=create)
    shm = shared_memory.SharedMemory(name=name, create=create, size=size)
    if not create:
        # Before 3.13 every attach registers with the resource tracker, which
        # would unlink the segment when this reader exits.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
    return shm


@dataclass(frozen=True)
class SlotRef:
    """A tiny, picklable pointer to one commit in one ring. This is what crosses
    process boundaries instead of the data itself."""

    ring: str
    index: int
    t_ns: int


@dataclass
class Frame:
    """One committed slot as seen by a reader. ``array`` is either a zero-copy
    view into shared memory or a private copy, depending on how it was read."""

    ring: "SharedRing"
    index: int
    t_ns: int
    array: np.ndarray
    is_view: bool

    @property
    def ref(self) -> SlotRef:
        return SlotRef(self.ring.name, self.index, self.t_ns)

    def valid(self) -> bool:
        """True while the writer has not yet reused this slot. Check it after
        you finish with a view; if it is False the view may hold newer data."""
        if not self.is_view:
            return True
        return self.ring._seq_of(self.index) == 2 * self.index + 2

    def copy(self) -> np.ndarray:
        arr = np.array(self.array, copy=True)
        if not self.valid():
            raise RingError(f"commit {self.index} overwritten while copying")
        return arr


class SharedRing:
    """A fixed-size ring of fixed-capacity slots in one shared memory segment."""

    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self._shm = shm
        self.owner = owner
        self.name = shm.name.lstrip("/")
        buf = shm.buf
        self._head = np.ndarray((1,), RING_DT, buffer=buf, offset=0)
        h = self._head[0]
        if int(h["magic"]) != MAGIC:
            raise RingError(f"{self.name} is not a Noumena ring")
        if int(h["version"]) != VERSION:
            raise RingError(f"{self.name} has ring version {int(h['version'])}, expected {VERSION}")
        self.slot_count = int(h["slot_count"])
        self.slot_bytes = int(h["slot_bytes"])
        self._payload_offset = int(h["payload_offset"])
        self._slots = np.ndarray((self.slot_count,), SLOT_DT, buffer=buf, offset=ALIGN)
        self._payload = np.ndarray(
            (self.slot_count, self.slot_bytes), np.uint8, buffer=buf, offset=self._payload_offset
        )
        self._writing: int | None = None

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, name: str, slot_count: int, slot_bytes: int) -> "SharedRing":
        if len(name) > MAX_NAME:
            raise RingError(f"ring name {name!r} is longer than {MAX_NAME} characters")
        if slot_count < 2:
            raise RingError("a ring needs at least 2 slots")
        slot_bytes = _round_up(max(slot_bytes, 1))
        payload_offset = _round_up(ALIGN + SLOT_DT.itemsize * slot_count)
        size = payload_offset + slot_count * slot_bytes
        shm = _open_shm(name, create=True, size=size)
        try:
            shm.buf[:payload_offset] = bytes(payload_offset)  # zero every header
            head = np.ndarray((1,), RING_DT, buffer=shm.buf, offset=0)
            head[0]["slot_count"] = slot_count
            head[0]["slot_bytes"] = slot_bytes
            head[0]["payload_offset"] = payload_offset
            head[0]["version"] = VERSION
            head[0]["magic"] = MAGIC  # last: readers trust the ring once magic is set
            del head
            return cls(shm, owner=True)
        except BaseException:
            shm.close()
            shm.unlink()
            raise

    @classmethod
    def attach(cls, name: str, retries: int = 50, wait: float = 0.02) -> "SharedRing":
        last: Exception | None = None
        for _ in range(retries):
            try:
                return cls(_open_shm(name, create=False), owner=False)
            except (FileNotFoundError, RingError) as e:  # not created or not initialised yet
                last = e
                time.sleep(wait)
        raise RingError(f"could not attach to ring {name!r}: {last}")

    def close(self) -> None:
        # Drop our numpy views first, or SharedMemory.close() raises BufferError.
        self._head = self._slots = self._payload = None  # type: ignore[assignment]
        try:
            self._shm.close()
        except BufferError:
            # Someone still holds a Frame view. The mapping is freed when it goes.
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "SharedRing":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        if self.owner:
            self.unlink()

    # ------------------------------------------------------------ writer
    @property
    def commits(self) -> int:
        return int(self._head[0]["commits"])

    def begin(self, shape: tuple[int, ...], dtype) -> np.ndarray:
        """Start commit ``n`` and return a writable view of its slot, shaped and
        typed as asked. Fill it in place, then call :meth:`commit`. This is the
        zero-copy path: the producer writes straight into shared memory."""
        if self._writing is not None:
            raise RingError("begin() called twice without commit()")
        dtype = np.dtype(dtype)
        if dtype not in DTYPE_CODE:
            raise RingError(f"unsupported dtype {dtype}")
        if len(shape) > MAX_DIMS:
            raise RingError(f"at most {MAX_DIMS} dimensions")
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes > self.slot_bytes:
            raise RingError(f"{nbytes} bytes does not fit a {self.slot_bytes} byte slot")
        n = self.commits
        slot = self._slots[n % self.slot_count]
        slot["seq"] = 2 * n + 1
        slot["dtype"] = DTYPE_CODE[dtype]
        slot["ndim"] = len(shape)
        slot["nbytes"] = nbytes
        sh = np.zeros(MAX_DIMS, np.uint32)
        sh[: len(shape)] = shape
        slot["shape"] = sh
        self._writing = n
        return self._payload[n % self.slot_count, :nbytes].view(dtype).reshape(shape)

    def commit(self, t_ns: int | None = None, flags: int = 0) -> int:
        """Publish the slot started by :meth:`begin`. Returns its commit index."""
        n = self._writing
        if n is None:
            raise RingError("commit() without begin()")
        slot = self._slots[n % self.slot_count]
        slot["index"] = n
        slot["t_ns"] = time.monotonic_ns() if t_ns is None else t_ns
        slot["flags"] = flags
        slot["seq"] = 2 * n + 2
        self._head[0]["commits"] = n + 1
        self._head[0]["writer_pid"] = os.getpid()
        self._writing = None
        return n

    def abort(self) -> None:
        """Abandon a begun write. The slot stays odd (unreadable) until reused."""
        self._writing = None

    def write(self, array: np.ndarray, t_ns: int | None = None, flags: int = 0) -> int:
        """Convenience: one copy from a private array into the ring."""
        out = self.begin(array.shape, array.dtype)
        np.copyto(out, array)
        return self.commit(t_ns, flags)

    # ------------------------------------------------------------ reader
    def _seq_of(self, index: int) -> int:
        return int(self._slots[index % self.slot_count]["seq"])

    def t_of(self, index: int) -> int | None:
        """Timestamp of commit ``index`` if it is still in the ring, else None."""
        slot = self._slots[index % self.slot_count]
        s1 = int(slot["seq"])
        t = int(slot["t_ns"])
        if s1 != 2 * index + 2 or int(slot["seq"]) != s1:
            return None
        return t

    def read(self, index: int, copy: bool = False) -> Frame | None:
        """Read commit ``index``. Returns None if it is not written yet, being
        written, or already overwritten. With ``copy=False`` the frame's array is
        a read-only view straight into shared memory."""
        slot = self._slots[index % self.slot_count]
        s1 = int(slot["seq"])
        if s1 != 2 * index + 2:
            return None
        dtype = DTYPES[int(slot["dtype"])]
        ndim = int(slot["ndim"])
        shape = tuple(int(x) for x in slot["shape"][:ndim])
        nbytes = int(slot["nbytes"])
        t_ns = int(slot["t_ns"])
        view = self._payload[index % self.slot_count, :nbytes].view(dtype).reshape(shape)
        view.flags.writeable = False
        arr = np.array(view, copy=True) if copy else view
        if int(slot["seq"]) != s1:
            return None  # torn: the writer came round while we were looking
        return Frame(self, index, t_ns, arr, is_view=not copy)

    def resolve(self, ref: SlotRef, copy: bool = False) -> Frame | None:
        return self.read(ref.index, copy=copy)

    def latest(self, copy: bool = False) -> Frame | None:
        """The newest committed frame, or None if nothing has been written."""
        for _ in range(4):
            n = self.commits
            if n == 0:
                return None
            f = self.read(n - 1, copy=copy)
            if f is not None:
                return f
        return None

    def recent(self, depth: int | None = None) -> list[tuple[int, int]]:
        """(index, t_ns) for the commits still readable, oldest first. Used by
        the time form to find the frame nearest a reference instant."""
        n = self.commits
        depth = min(depth or self.slot_count - 1, self.slot_count - 1, n)
        out = []
        for i in range(n - depth, n):
            t = self.t_of(i)
            if t is not None:
                out.append((i, t))
        return out


class RingReader:
    """A reader with its own cursor, for consumers that want every frame in
    order (or to know how many they missed)."""

    def __init__(self, ring: SharedRing, start: str = "latest"):
        self.ring = ring
        self.cursor = ring.commits if start == "latest" else 0
        self.overruns = 0

    def next(self, copy: bool = False) -> Frame | None:
        """The next unread frame, skipping ahead if the writer lapped us."""
        n = self.ring.commits
        if self.cursor >= n:
            return None
        oldest = n - (self.ring.slot_count - 1)
        if self.cursor < oldest:
            self.overruns += oldest - self.cursor
            self.cursor = oldest
        while self.cursor < self.ring.commits:
            f = self.ring.read(self.cursor, copy=copy)
            self.cursor += 1
            if f is not None:
                return f
            self.overruns += 1
        return None

    def drain(self, copy: bool = False) -> list[Frame]:
        out = []
        while (f := self.next(copy=copy)) is not None:
            out.append(f)
        return out
