"""v1 two-wrist (24D) generalization — reps, extraction, bimanual consistency loss, model, train."""

import numpy as np
import pytest

from h2b.representations import frames as F

torch = pytest.importorskip("torch")

from h2b.representations import body as B
from h2b.models import fk_torch as FKt
from h2b.data import smpl_fk as SF
from h2b import losses as L
from h2b import training as TR


def _bimanual_hand_from_body(body):
    """Build a self-consistent 24D hand = FK(both wrists of `body`) + finite-diff velocity."""
    p, r = FKt.wrists_pose(torch.tensor(body)[None], F.WRIST_JOINTS)     # (1,T,2,3),(1,T,2,6)
    p, r = p[0].numpy(), r[0].numpy()
    T = body.shape[0]
    hand = np.zeros((T, F.HAND24_DIM), np.float32)
    for k in range(2):
        hand[:, 12 * k:12 * k + 3] = p[:, k]
        hand[:, 12 * k + 3:12 * k + 6] = F.finite_diff_velocity(p[:, k], 30.0)
        hand[:, 12 * k + 6:12 * k + 12] = r[:, k]
    return hand


def test_frames_24d_pack_canon_roundtrip():
    assert F.hand_dim_for(2) == 24 and F.wrist_count_of(24) == 2 and F.wrist_count_of(12) == 1
    assert F.hand_pos_slices(24) == [slice(0, 3), slice(12, 15)]
    l12 = np.arange(12.0); r12 = np.arange(12.0) + 100
    h = F.pack_hand24(l12, r12)
    assert h.shape == (24,)
    a, b = F.unpack_hand24(h)
    assert np.allclose(a, l12) and np.allclose(b, r12)
    # canonicalize shifts BOTH position blocks by ONE anchor; velocity/orientation untouched
    seq = np.random.randn(7, 24)
    c, anchor = F.canonicalize_hand(seq)
    assert np.allclose(anchor, seq[0, 0:3])
    assert np.allclose(c[:, 0:3], seq[:, 0:3] - anchor)
    assert np.allclose(c[:, 12:15], seq[:, 12:15] - anchor)
    assert np.allclose(c[:, 3:6], seq[:, 3:6]) and np.allclose(c[:, 18:24], seq[:, 18:24])
    assert np.allclose(F.decanonicalize_hand(c, anchor), seq)


def test_fk_right_wrist_and_wrists_pose():
    _, body = TR.synthetic_clips(n_clips=1, T=16, seed=0)[0]
    rp, rr = FKt.right_wrist_pose(torch.tensor(body)[None])
    p, r = FKt.wrists_pose(torch.tensor(body)[None], F.WRIST_JOINTS)
    assert p.shape == (1, 16, 2, 3) and r.shape == (1, 16, 2, 6)
    assert torch.allclose(p[:, :, 1], rp) and torch.allclose(r[:, :, 1], rr)   # index 1 == right


def test_bimanual_hand_consistency_cycle():
    _, body = TR.synthetic_clips(n_clips=1, T=24, seed=1)[0]
    hand24 = _bimanual_hand_from_body(body)
    bt, ht = torch.tensor(body)[None], torch.tensor(hand24)[None]
    _, parts = L.compute_losses(bt, bt, ht, {"hand_consistency": 1.0})
    assert parts["hand_consistency"] < 1e-6                    # FK(both wrists) == the 24D input
    # a wrong body must break it
    _, parts2 = L.compute_losses(bt + 0.3, bt, ht, {"hand_consistency": 1.0})
    assert parts2["hand_consistency"] > 1e-3


def test_extract_hand_bilateral_shape_and_left_block():
    T = 12
    poses = np.zeros((T, 72), np.float32); poses[:, 0] = np.linspace(0, 0.5, T)
    trans = np.cumsum(np.ones((T, 3), np.float32) * 0.01, 0)
    betas = np.zeros(10, np.float32)
    h24 = SF.extract_hand_bilateral(poses, trans, betas, SF.synthetic_joints_fn, fps=30.0)
    h12 = SF.extract_hand12(poses, trans, betas, SF.synthetic_joints_fn, fps=30.0, joint=F.LEFT_WRIST)
    assert h24.shape == (T, 24)
    assert np.allclose(h24[:, :12], h12)                      # left block == the 1-wrist signal


def test_dit_denoiser_hand24_forward():
    from h2b.models.diffusion import DiTDenoiser
    m = DiTDenoiser(hidden=64, n_layers=2, hand_dim=24).eval()
    x = torch.randn(2, 10, B.MOTION_DIM); t = torch.randint(0, 1000, (2,)); hand = torch.randn(2, 10, 24)
    out = m(x, t, hand)
    assert out.shape == (2, 10, B.MOTION_DIM) and torch.isfinite(out).all()


def test_train_diffusion_24d_smoke():
    clips = []
    for s in range(2):
        _, body = TR.synthetic_clips(n_clips=1, T=40, seed=s)[0]
        clips.append((_bimanual_hand_from_body(body), body))
    model, diff, hist = TR.train_diffusion(clips, length=20, steps=20, batch_size=8, device="cpu",
                                           hidden=64, n_layers=2, num_steps=100, hand_dim=24,
                                           log_every=10)
    assert np.isfinite(hist[-1]["total"])


def test_streamer_block_24d():
    # 24D online streaming: push_block must accept (B, 24) (it used to hard-reshape to 12 cols)
    from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
    from h2b.models.streaming import DiffusionStreamer
    diff = GaussianDiffusion(num_steps=50)
    m = DiTDenoiser(hidden=64, n_layers=2, hand_dim=24).eval()
    s = DiffusionStreamer(m, diff, window=5, block=4, sample_steps=2)
    out = s.push_block(np.zeros((4, 24), np.float32))
    assert out.shape == (4, B.MOTION_DIM) and np.isfinite(out).all()
    single = s.push_block(np.ones(24, np.float32))             # a single flat frame also works
    assert single.shape == (1, B.MOTION_DIM)


def test_generate_stream_infers_model_device():
    # device=None must resolve to the model's own device instead of crashing on cpu/cuda mismatch
    from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
    from h2b import inference as INF
    m = DiTDenoiser(hidden=64, n_layers=2, hand_dim=12).eval()
    hand = np.zeros((8, 12), np.float32)
    out = INF.generate_stream(m, hand, GaussianDiffusion(num_steps=50), window=5, block=4,
                              sample_steps=2)
    assert out.shape == (8, B.MOTION_DIM) and np.isfinite(out).all()
