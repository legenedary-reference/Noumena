import numpy as np

from noumena.forms import BodyGrid, MonotonicTime, Pose, RefractoryCause, RunningBaseline


def test_time_aligns_to_nearest_past_frame():
    t = MonotonicTime(tolerance_ns=50)
    got = t.align(1000, {"a": [(0, 900), (1, 980), (2, 1010)], "b": [(5, 800)], "c": []})
    assert got == {"a": (1, 20)}  # b is too old, c is empty, a's future frame is ignored


def test_time_falls_back_to_future_when_no_past():
    t = MonotonicTime(tolerance_ns=50)
    assert t.align(1000, {"a": [(3, 1020)]}) == {"a": (3, -20)}


def test_time_per_stream_tolerance():
    t = MonotonicTime(tolerance_ns=10, per_stream={"slow": 500})
    assert set(t.align(1000, {"fast": [(0, 900)], "slow": [(0, 600)]})) == {"slow"}


def test_pose_rotates_then_translates():
    p = Pose.from_euler(yaw=np.pi / 2, xyz=(1, 0, 0))
    np.testing.assert_allclose(p.apply(np.array([[1.0, 0, 0]])), [[1, 1, 0]], atol=1e-12)


def test_grid_carves_hits_and_leaves_unknown():
    g = BodyGrid(extent=(0, 4, -2, 2), resolution=0.5)
    out = np.zeros((2, *g.grid_shape), np.float32)
    hits = np.array([[3.0, 0.1, 0], [2.0, 0.0, 0]])
    g.carve(np.zeros(3), hits, np.array([True, False]), out)
    r, c, _ = g.cells(hits[:1])
    assert out[0, r[0], c[0]] == 1  # the hit
    assert out[0].sum() == 1  # the invalid ray marks nothing
    assert out[1].sum() >= 6  # cells along the ray were observed
    assert out[1].sum() < out[1].size  # most of the grid stays unknown


def test_baseline_novelty_is_about_one_for_noise_and_large_for_change():
    b = RunningBaseline()
    rng = np.random.default_rng(0)
    scores = [b.observe("s", rng.normal(size=2000)) for _ in range(60)]
    assert scores[0] == 0.0
    assert 0.7 < np.mean(scores[20:]) < 1.5
    x = rng.normal(size=2000)
    x[:20] += 15  # a small thing changes a lot
    assert b.observe("s", x) > 5


def test_baseline_ignores_unknown_elements():
    b = RunningBaseline(warmup=1)
    for _ in range(5):
        b.observe("s", np.zeros(100))
    x = np.zeros(100)
    x[:50] = np.nan
    assert b.observe("s", x) < 1


def test_cause_refractory_and_links():
    c = RefractoryCause(threshold=3, refractory_ns=100, window_ns=1000)
    a = c.observe("cam", 0, 5)
    assert a is not None and a.caused_by is None
    assert c.observe("cam", 50, 9) is None  # refractory
    assert c.observe("mic", 60, 1) is None  # below threshold
    b = c.observe("mic", 70, 4)
    assert b.caused_by == a.id
    assert c.observe("cam", 200, 4).caused_by == b.id
    assert c.observe("imu", 5000, 4).caused_by is None  # outside the window
