"""Training objective for Hand2WholeBody (weights in configs/default.yaml `loss`).

Terms (see research synthesis / CONTRACT):
  * recon         — diffusion x0 reconstruction (or flow-matching target).
  * rot6d         — geodesic-ish L2 on the 6D joint rotations.
  * fk_joint      — decode SMPL, supervise 3D joint positions (FK consistency).
  * velocity      — first/second difference penalty for smoothness (low jitter → trackable).
  * hand_consistency — THE key term: run FK on the predicted SMPL pose, recover the GLOBAL
                    left-wrist position+orientation, and force it to match the input 12D.
                    This closes the loop so the model actually honors its conditioning
                    rather than hallucinating a plausible-but-unconditioned body.
  * foot_contact  — light no-slide penalty on predicted-contact feet (keep small; the
                    HoloMotion RL policy handles balance — do not over-constrain).

torch-guarded so importing the package never requires torch. The geometric terms are
implemented; the FK-dependent terms need a differentiable SMPL/FK (torch port of
h2wb.data.smpl_fk) wired in M2/M4.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as Fn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def rot6d_loss(pred6d, gt6d):
    """L2 on 6D rotation channels. pred/gt: (..., J*6)."""
    return Fn.mse_loss(pred6d, gt6d)


def velocity_loss(pred_motion):
    """Penalize frame-to-frame jerk for smoothness. pred_motion: (B, T, M)."""
    v = pred_motion[:, 1:] - pred_motion[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    return v.pow(2).mean() + a.pow(2).mean()


def hand_consistency_loss(pred_wrist_pos, pred_wrist_rot6d, input_hand12):
    """Match the FK-derived global left-wrist pose to the conditioning 12D signal.

    pred_wrist_pos:   (B, T, 3)  global wrist position from FK on the predicted pose
    pred_wrist_rot6d: (B, T, 6)  global wrist orientation (6D) from FK on the predicted pose
    input_hand12:     (B, T, 12) the conditioning signal [pos|vel|rot6d]
    """
    tgt_pos = input_hand12[..., 0:3]
    tgt_rot = input_hand12[..., 6:12]
    return Fn.mse_loss(pred_wrist_pos, tgt_pos) + Fn.mse_loss(pred_wrist_rot6d, tgt_rot)


def fk_joint_loss(pred_joints, gt_joints):
    """3D joint-position loss after FK. (B, T, J, 3)."""
    return Fn.mse_loss(pred_joints, gt_joints)


def foot_contact_loss(pred_foot_vel, contact_prob):
    """No-slide: contact-weighted foot velocity. pred_foot_vel (B,T,Nf,3), contact (B,T,Nf)."""
    return (contact_prob[..., None] * pred_foot_vel.pow(2)).mean()


def total_loss(parts: dict, weights: dict):
    """Weighted sum of available terms; missing terms are skipped."""
    return sum(weights.get(k, 0.0) * v for k, v in parts.items())
