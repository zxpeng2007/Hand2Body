"""Tests for the AMASS-style SMPL export (Stage-2 → Stage-3 handoff)."""

import numpy as np

from h2wb.export import to_amass_npz as EX
from h2wb.representations import frames as F
from h2wb.representations import rotations as R


def test_motion6d_to_aa_shapes_and_roundtrip():
    rng = np.random.default_rng(0)
    T = 12
    root_R = R.axis_angle_to_matrix(rng.standard_normal((T, 3)) * 0.3)
    body_R = R.axis_angle_to_matrix(rng.standard_normal((T, 23, 3)) * 0.3)
    # encode with the project convention so it matches motion6d_to_aa's decode
    root6d = R.matrix_to_rotation_6d(root_R, convention=F.PROJECT_R6D)
    body6d = R.matrix_to_rotation_6d(body_R, convention=F.PROJECT_R6D)
    poses = EX.motion6d_to_aa(root6d, body6d)
    assert poses.shape == (T, 72)
    # re-encode and compare rotations (axis-angle is not unique, matrices are)
    assert np.allclose(R.axis_angle_to_matrix(poses[:, 0:3]), root_R, atol=1e-7)
    body_back = R.axis_angle_to_matrix(poses[:, 3:72].reshape(T, 23, 3))
    assert np.allclose(body_back, body_R, atol=1e-7)


def test_npz_written_with_contract_keys(tmp_path):
    T = 8
    poses = np.zeros((T, 72))
    trans = np.tile([0.0, 0.0, 1.0], (T, 1))
    betas = np.zeros(10)
    out = EX.smpl_motion_to_amass_npz(
        str(tmp_path / "clip.npz"), poses, trans, betas, fps=30, gender="neutral",
        contacts=np.zeros((T, 4)),
    )
    d = np.load(out, allow_pickle=True)
    assert set(["poses", "trans", "betas", "gender", "mocap_frame_rate", "contacts"]).issubset(d.files)
    assert d["poses"].shape == (T, 72)
    assert d["trans"].shape == (T, 3)
    assert int(d["mocap_frame_rate"]) == 30
    assert str(d["gender"]) == "neutral"
