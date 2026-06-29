"""Tests for the train.pkl loader using a synthetic joblib pkl in the real format
(dict of lists-of-sequences; poses [T,66] = 22-joint axis-angle; trans [T,3])."""

import numpy as np
import pytest

joblib = pytest.importorskip("joblib")

from h2wb.data import pkl_loader as PL
from h2wb.representations import body as B
from h2wb.representations import frames as F
from h2wb.data import smpl_fk as FK


def _make_pkl(tmp_path, n_seq=3, seed=0):
    rng = np.random.default_rng(seed)
    poses, trans, balls = [], [], []
    for i in range(n_seq):
        T = 20 + i * 5
        po = np.cumsum(rng.standard_normal((T, 66)) * 0.05, axis=0).astype(np.float32)  # 22 joints
        tr = (np.cumsum(rng.standard_normal((T, 3)) * 0.01, axis=0) + [0, 0, 1.0]).astype(np.float32)
        poses.append(po); trans.append(tr); balls.append(rng.standard_normal((T, 3)).astype(np.float32))
    payload = {"poses": poses, "trans": trans, "ball_pos": balls}
    p = tmp_path / "train.pkl"
    joblib.dump(payload, str(p))
    return str(p)


def test_load_and_iter_sequences(tmp_path):
    path = _make_pkl(tmp_path, n_seq=3)
    payload = PL.load_smpl_pkl(path)
    seqs = list(PL.iter_sequences(payload))
    assert len(seqs) == 3
    assert seqs[0]["poses"].shape[1] == 66 and seqs[0]["trans"].shape[1] == 3


def test_poses_padding_66_to_72():
    poses66 = np.ones((5, 66))
    p72 = PL._poses_to_72(poses66)
    assert p72.shape == (5, 72)
    assert np.allclose(p72[:, 66:], 0.0)                  # hand joints zeroed
    assert np.allclose(p72[:, :66], poses66)


def test_pkl_to_clips_shapes_and_consistency(tmp_path):
    path = _make_pkl(tmp_path, n_seq=2)
    clips = PL.pkl_to_clips(path)
    assert len(clips) == 2
    hand, body = clips[0]
    T = hand.shape[0]
    assert hand.shape == (T, F.HAND12_DIM)
    assert body.shape == (T, B.MOTION_DIM)
    # the derived 12D's 6D must decode to the true global left-wrist rotation
    poses72 = PL._poses_to_72(np.cumsum(np.zeros((T, 66)), axis=0))  # placeholder, recompute below
    Rfrom12 = F.matrix_from_hand12_rot(hand)
    # recompute global wrist rotation from the same poses used to build the clip
    payload = PL.load_smpl_pkl(path)
    seq0 = list(PL.iter_sequences(payload))[0]
    Rg = FK.global_joint_rotations(PL._poses_to_72(seq0["poses"]))[:, F.PADDLE_HAND_JOINT]
    assert np.allclose(Rfrom12, Rg, atol=1e-4)


def test_72dim_poses_also_supported(tmp_path):
    rng = np.random.default_rng(1)
    T = 16
    payload = {"poses": [np.zeros((T, 72), np.float32)],
               "trans": [np.tile([0, 0, 1.0], (T, 1)).astype(np.float32)]}
    p = tmp_path / "t72.pkl"
    joblib.dump(payload, str(p))
    clips = PL.pkl_to_clips(str(p))
    assert clips[0][1].shape == (T, B.MOTION_DIM)
