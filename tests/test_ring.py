import multiprocessing as mp

import numpy as np
import pytest

from noumena.ring import RingError, RingReader, SharedRing, make_name


@pytest.fixture
def ring():
    r = SharedRing.create(make_name("test"), slot_count=4, slot_bytes=1024)
    yield r
    r.close()
    r.unlink()


def test_write_then_read_roundtrip(ring):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    n = ring.write(a, t_ns=123)
    f = ring.read(n)
    assert f.t_ns == 123 and f.index == 0
    assert f.array.shape == (3, 4) and f.array.dtype == np.float32
    np.testing.assert_array_equal(f.array, a)
    assert f.is_view and not f.array.flags.writeable


def test_begin_commit_writes_in_place(ring):
    out = ring.begin((2, 2), np.int16)
    out[...] = [[1, 2], [3, 4]]
    assert ring.read(0) is None  # not readable until committed
    ring.commit(t_ns=5)
    np.testing.assert_array_equal(ring.latest().array, [[1, 2], [3, 4]])


def test_view_is_zero_copy_and_detects_overwrite(ring):
    ring.write(np.full(4, 1.0, np.float32))
    f = ring.read(0)
    assert f.valid()
    for _ in range(4):  # lap the ring; slot 0 is reused
        ring.write(np.full(4, 9.0, np.float32))
    assert not f.valid()
    assert f.array[0] == 9.0  # same physical memory: it changed under us
    with pytest.raises(RingError):
        f.copy()
    assert ring.read(0) is None


def test_copy_survives_overwrite(ring):
    ring.write(np.full(4, 1.0, np.float32))
    f = ring.read(0, copy=True)
    for _ in range(4):
        ring.write(np.full(4, 9.0, np.float32))
    assert f.valid() and f.array[0] == 1.0


def test_reader_counts_overruns(ring):
    reader = RingReader(ring, start="oldest")
    for i in range(10):
        ring.write(np.array([i], np.int32))
    got = [int(f.array[0]) for f in reader.drain()]
    assert got == [7, 8, 9]  # slot_count - 1 frames survive
    assert reader.overruns == 7
    ring.write(np.array([10], np.int32))
    assert int(reader.next().array[0]) == 10 and reader.next() is None


def test_recent_lists_readable_commits(ring):
    for i in range(6):
        ring.write(np.array([i], np.int32), t_ns=100 + i)
    assert ring.recent() == [(3, 103), (4, 104), (5, 105)]


def test_rejects_oversize_and_bad_dtype(ring):
    with pytest.raises(RingError):
        ring.begin((1024,), np.float32)
    with pytest.raises(RingError):
        ring.begin((2,), np.complex64)


def test_rejects_attach_to_foreign_segment():
    from multiprocessing import shared_memory

    name = make_name("foreign")
    shm = shared_memory.SharedMemory(name=name, create=True, size=4096)
    try:
        with pytest.raises(RingError):
            SharedRing.attach(name, retries=2)
    finally:
        shm.close()
        shm.unlink()


def _writer(name, count):
    ring = SharedRing.attach(name)
    for i in range(count):
        out = ring.begin((256,), np.float32)
        out[...] = i
        ring.commit(t_ns=i)
    ring.close()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_cross_process_writer(method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} not available")
    ring = SharedRing.create(make_name("xproc"), slot_count=8, slot_bytes=1024)
    try:
        p = mp.get_context(method).Process(target=_writer, args=(ring.name, 20))
        p.start()
        p.join(20)
        assert p.exitcode == 0
        assert ring.commits == 20
        f = ring.latest()
        assert f.index == 19 and f.t_ns == 19 and np.all(f.array == 19)
    finally:
        ring.close()
        ring.unlink()
