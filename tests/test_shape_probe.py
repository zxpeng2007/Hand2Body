"""Tests for the scripts/shape_probe.py trajectory builder (no model, cache, or GPU needed)."""
import importlib.util
import os

import numpy as np

from h2b.representations import frames as F

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "shape_probe.py")
_SPEC = importlib.util.spec_from_file_location("shape_probe", _PATH)
SP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SP)

RNG = np.random.default_rng(0)


def _fake_hand_frames(n=4000, wrists=1):
    """Synthetic cached hand frames: wrist 0 in a box wider in y than x, plausible vel/rot."""
    blocks = []
    for k in range(wrists):
        pos = np.stack([RNG.uniform(-1.9, -1.4, n),          # x span 0.5
                        RNG.uniform(-0.4, 0.5, n),           # y span 0.9  (wider -> u_axis)
                        RNG.uniform(1.0, 1.3, n)], 1)        # z
        vel = RNG.normal(0.0, 0.5, (n, 3))
        rot = np.repeat(np.array([[1.0, 0, 0, 0, 1.0, 0]]), n, 0)
        blocks.append(np.concatenate([pos, vel, rot], 1))
    return np.concatenate(blocks, 1).astype(np.float32)


def test_workspace_picks_wider_horizontal_axis():
    ws = SP.wrist_workspace(_fake_hand_frames(), 0)
    assert ws["u_axis"] == 1 and ws["held_axis"] == 0
    assert np.all(np.abs(ws["center"] - [-1.65, 0.05, 1.15]) < 0.05)
    assert ws["rot6d"].shape == (6,)
    assert ws["speed_p90"] > 0


def test_build_hand_anchor_and_conventions():
    ws = [SP.wrist_workspace(_fake_hand_frames(), 0)]
    ts = 120
    for name in SP.SHAPE_NAMES:
        h = SP.build_hand(name, ws, ts)
        assert h.shape == (ts + SP.LEAD, 12) and np.isfinite(h).all()
        # the lead-in starts every probe AT the workspace center: this pins the canonicalization
        # anchor to the data-typical height (otherwise the model infers a shifted floor)
        assert np.allclose(h[0, 0:3], ws[0]["center"], atol=1e-6)
        # velocity block is exactly the training convention over the position block
        assert np.allclose(h[:, 3:6], F.finite_diff_velocity(h[:, 0:3].astype(np.float64), SP.FPS),
                           atol=1e-4)
        # orientation held constant; held horizontal axis does not move; box respected
        assert np.allclose(h[:, 6:12], h[0:1, 6:12])
        held = ws[0]["held_axis"]
        assert np.ptp(h[:, held]) < 1e-6
        u = ws[0]["u_axis"]
        assert np.abs(h[:, u] - ws[0]["center"][u]).max() <= ws[0]["half"][u] + 1e-6
        assert np.abs(h[:, 2] - ws[0]["center"][2]).max() <= ws[0]["half"][2] + 1e-6


def test_build_hand_two_wrists_block_order_and_idle():
    H = _fake_hand_frames(wrists=2)
    ws = [SP.wrist_workspace(H, 0), SP.wrist_workspace(H, 12)]
    h = SP.build_hand("circle", ws, 120)
    assert h.shape == (120 + SP.LEAD, 24)
    # [wrist0 | wrist1] block order; wrist 1 idles within a few cm of its own center
    drift1 = np.linalg.norm(h[:, 12:15] - ws[1]["center"][None], axis=1)
    assert drift1.max() < 0.05
    # wrist 0 actually draws (moves far more than the idle hand)
    drift0 = np.linalg.norm(h[:, 0:3] - ws[0]["center"][None], axis=1)
    assert drift0.max() > 5 * drift1.max()
