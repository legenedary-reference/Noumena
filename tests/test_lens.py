import numpy as np
import pytest

from noumena.forms import Forms, Pose
from noumena.lens import (
    ADAPTERS,
    AudioAdapter,
    CameraAdapter,
    DistanceAdapter,
    IMUAdapter,
    Intrinsics,
    Lens,
    MotionAdapter,
    RawSample,
    bilinear_sample,
    block_mean,
    mel_filterbank,
    mulaw_decode,
    mulaw_encode,
    out_spec_of,
    quantize_uniform,
)
from noumena.sensors import SIMULATORS


def run(adapter, data, forms=None):
    forms = forms or Forms()
    shape, dtype = out_spec_of(adapter, forms)
    out = np.full(shape, np.nan if dtype.kind == "f" else 0, dtype)
    adapter.refract(RawSample("s", 0, data), out, forms)
    return out


@pytest.mark.parametrize("modality", sorted(ADAPTERS))
def test_every_simulator_fits_its_adapter(modality):
    forms = Forms()
    src = SIMULATORS[modality](paced=False)
    adapter = ADAPTERS[modality]()
    shape, dtype = out_spec_of(adapter, forms)
    out = np.zeros(shape, dtype)
    for _ in range(3):
        adapter.refract(src.read(), out, forms)
    assert np.isfinite(out.astype(np.float64)).all()


def test_bilinear_sample_interpolates():
    img = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert bilinear_sample(img, np.array([0.5]), np.array([0.5]))[0] == pytest.approx(15)


def test_block_mean_ignores_nan():
    a = np.ones((4, 4))
    a[0, 0] = np.nan
    np.testing.assert_allclose(block_mean(a, (2, 2)), np.ones((2, 2)))


def test_camera_crops_resizes_normalises():
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, 160:] = 255
    out = run(CameraAdapter(size=(48, 64), mean=(0.0,), std=(1.0,)), img)
    assert out.shape == (3, 48, 64)
    assert out[:, :, 0].max() == pytest.approx(0) and out[:, :, -1].min() == pytest.approx(1)


def test_camera_grayscale_and_rectify_identity():
    img = np.random.default_rng(0).integers(0, 255, (60, 80), np.uint8)
    plain = run(CameraAdapter(size=(60, 80), grayscale=True, mean=(0,), std=(1,)), img)
    k = Intrinsics(fx=80, fy=80, cx=39.5, cy=29.5)  # no distortion: rectify is a no-op
    rect = run(CameraAdapter(size=(60, 80), grayscale=True, mean=(0,), std=(1,), intrinsics=k), img)
    np.testing.assert_allclose(plain, rect, atol=1e-5)
    np.testing.assert_allclose(plain[0], img / 255.0, atol=1e-5)


def test_camera_rectify_moves_pixels_outward_for_barrel():
    img = np.zeros((101, 101), np.uint8)
    img[50, 90] = 255  # a point near the right edge
    k = Intrinsics(fx=100, fy=100, cx=50, cy=50, k1=-0.3)
    out = run(CameraAdapter(size=(101, 101), grayscale=True, mean=(0,), std=(1,), intrinsics=k), img)
    assert np.argmax(out[0, 50]) > 90


def test_audio_logmel_finds_tone_band():
    sr = 16000
    t = np.arange(1600) / sr
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
    a = AudioAdapter()
    out = run(a, tone)
    assert out.shape == (a.frames, a.n_mels)
    fb = mel_filterbank(sr, a.n_fft, a.n_mels)
    peak_bin = int(round(1000 * a.n_fft / sr))
    assert np.argmax(out.mean(axis=0)) == np.argmax(fb[:, peak_bin])


def test_audio_mulaw_tokens_roundtrip():
    x = np.linspace(-1, 1, 1600).astype(np.float32)
    tok = run(AudioAdapter(mode="mulaw"), x)
    assert tok.dtype == np.uint8 and tok.shape == (1600,)
    assert np.abs(mulaw_decode(tok) - x).max() < 0.05
    np.testing.assert_array_equal(mulaw_encode(x), tok)


def test_quantize_uniform_bounds():
    q = quantize_uniform(np.array([-5.0, 0.0, 0.5, 1.0, 9.0]), 0.0, 1.0)
    assert q.tolist() == [0, 0, 128, 255, 255]


def test_imu_still_device_reads_level_and_learns_bias():
    rng = np.random.default_rng(0)
    a = IMUAdapter(rate=100, calibrate=50)
    forms = Forms()
    out = np.zeros(12, np.float32)
    for _ in range(10):
        s = np.zeros((10, 6))
        s[:, 2] = 9.80665
        s[:, 3:] = 0.02  # gyro bias
        s += rng.normal(0, 0.001, s.shape)
        a.refract(RawSample("imu", 0, s), out, forms)
    roll, pitch, yaw = out[:3]
    assert abs(roll) < 0.01 and abs(pitch) < 0.01 and abs(yaw) < 0.01
    np.testing.assert_allclose(out[6:9], 0, atol=0.05)  # gravity removed


def test_imu_tilt_is_recovered():
    a = IMUAdapter(blend=0.0)  # trust the accelerometer fully
    roll = 0.3
    s = np.zeros((5, 6))
    s[:, 1] = 9.80665 * np.sin(roll)
    s[:, 2] = 9.80665 * np.cos(roll)
    out = run(a, s)
    assert out[0] == pytest.approx(roll, abs=1e-3)
    np.testing.assert_allclose(out[6:9], 0, atol=1e-3)


def test_distance_marks_hits_and_keeps_no_return_unknown():
    forms = Forms()
    a = DistanceAdapter(pose=Pose.from_euler(xyz=(0, 0, 0)))
    r = np.full((8, 8), 2.0)
    out = run(a, r, forms)
    assert out[0].sum() > 0
    none = run(a, np.full((8, 8), np.nan), forms)
    assert none.sum() == 0  # nothing returned: everything unknown, not empty


def test_motion_removes_ego_motion():
    v = np.zeros((24, 32, 2), np.float32)
    v[..., 0] = 5.0
    v[0:4, 0:4, 0] = 13.0
    m = MotionAdapter(grid=(6, 8), max_speed=8.0)
    out = run(m, v)
    assert out[2, 0, 0] == pytest.approx(1.0)
    assert out[2, 1:, 1:].max() == pytest.approx(0.0)
    np.testing.assert_allclose(m.last_ego, [5.0, 0.0])


def test_lens_synthesize_aligns_and_carries_events():
    lens = Lens()
    lens.forms.substance.warmup = 1
    for i in range(3):
        lens.apprehend("cam", i, np.zeros(10))
    ev = lens.forms.cause.observe("cam", 10, 99.0)
    pkt = lens.synthesize(
        1_000_000_000,
        {"cam": ("ringA", [(0, 900_000_000), (1, 990_000_000)]), "old": ("ringB", [(0, 1)])},
        [ev],
        "test",
    )
    assert set(pkt.streams) == {"cam"}  # "old" is beyond tolerance
    assert pkt.streams["cam"].ref.index == 1 and pkt.streams["cam"].lag_ns == 10_000_000
    assert pkt.events == [ev] and "cam" in pkt.describe()
