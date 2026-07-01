"""Differentiable SMPL forward kinematics (torch) for the training losses.

Given the model's 135-D motion output, recover (a) global joint rotations and (b) global
joint positions, so we can:
  * supervise 3D joint positions (fk_joint loss), and
  * recover the GLOBAL left-wrist position + orientation to match the input 12D signal
    (hand-consistency loss — the term that forces the body to honor the hand).

Joint POSITIONS need rest-pose joint offsets, which depend on the SMPL model (betas).
We inject them via `rest_joints` (default = the model-free approximation from
h2b.data.smpl_fk; swap in the real model's J once the SMPL build ships). Global
ROTATIONS need only the pose, so they are exact regardless.
"""

from __future__ import annotations

import numpy as np

from ..representations import body as B
from ..representations import rotations_torch as RT

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def default_rest_joints():
    """(22, 3) torch tensor of approximate SMPL rest joints (placeholder until real J)."""
    from ..data.smpl_fk import _approx_rest_joints
    return torch.tensor(_approx_rest_joints()[: B.NUM_BODY_JOINTS], dtype=torch.float32)


def forward_kinematics(local_R, trans, rest_joints, parents=B.BODY_PARENTS):
    """local_R (..., J, 3, 3), trans (..., 3) -> (global_R (...,J,3,3), pos (...,J,3)).

    Joint 0 rotation is treated as global (global_orient); joints 1..J-1 are local.
    """
    J = local_R.shape[-3]
    rest = rest_joints.to(local_R.dtype).to(local_R.device)
    gR = [local_R[..., 0, :, :]]
    for j in range(1, J):
        gR.append(gR[parents[j]] @ local_R[..., j, :, :])
    pos = [trans + rest[0]]
    for j in range(1, J):
        offset = (rest[j] - rest[parents[j]]).reshape(3, 1)
        rotated = (gR[parents[j]] @ offset).squeeze(-1)
        pos.append(pos[parents[j]] + rotated)
    return torch.stack(gR, dim=-3), torch.stack(pos, dim=-2)


def motion_to_joints(motion, rest_joints=None, parents=B.BODY_PARENTS):
    """(..., 135) -> (global_R (...,22,3,3), pos (...,22,3))."""
    if rest_joints is None:
        rest_joints = default_rest_joints()
    trans = motion[..., B.B_TRANS]
    rot6d = motion[..., B.B_ROT6D].reshape(*motion.shape[:-1], B.NUM_BODY_JOINTS, 6)
    local_R = RT.rotation_6d_to_matrix(rot6d)
    return forward_kinematics(local_R, trans, rest_joints, parents)


def wrist_pose(motion, joint, rest_joints=None):
    """(..., 135) -> (pos (...,3), rot6d (...,6)) of the GLOBAL wrist `joint` (20=left, 21=right)."""
    gR, pos = motion_to_joints(motion, rest_joints)
    return pos[..., joint, :], RT.matrix_to_rotation_6d(gR[..., joint, :, :])


def left_wrist_pose(motion, rest_joints=None):
    """GLOBAL left wrist (joint 20) — compared against the input 12D in the hand-consistency loss."""
    from ..representations.frames import LEFT_WRIST     # == 20, inside the 22-joint subtree
    return wrist_pose(motion, LEFT_WRIST, rest_joints)


def right_wrist_pose(motion, rest_joints=None):
    """GLOBAL right wrist (joint 21) — the 2-wrist (bimanual) consistency target."""
    from ..representations.frames import RIGHT_WRIST
    return wrist_pose(motion, RIGHT_WRIST, rest_joints)


def wrists_pose(motion, joints, rest_joints=None):
    """N-wrist consistency helper: (..., 135), joints=(20,) or (20,21) ->
    (pos (...,K,3), rot6d (...,K,6)) for the K tracked wrists, one FK pass."""
    gR, pos = motion_to_joints(motion, rest_joints)
    p = torch.stack([pos[..., j, :] for j in joints], dim=-2)
    r = torch.stack([RT.matrix_to_rotation_6d(gR[..., j, :, :]) for j in joints], dim=-2)
    return p, r
