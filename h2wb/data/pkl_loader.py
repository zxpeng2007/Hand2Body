"""Loader for the coworker's whole-body SMPL data (train.pkl) -> training pairs.

Format (see memory / vis_smpl22_aitviewer.py): a JOBLIB pickle of a dict whose values are
LISTS (or object-arrays) of per-sequence arrays:
  * trans : [T, 3] root translation (world, z-up).
  * poses : [T, 66] axis-angle = root(3) + 22 body joints(63)  -> the smpl_22 layout
            (joints 0..21, hand joints dropped) — matches h2wb.representations.body.
  * ball_pos, nmr_joints, left/right_hand_pose : optional extras (ignored for training).

We pad poses 66 -> 72 (zero hand joints) so the existing 24-joint FK works, derive the
left-wrist 12D via h2wb.data.smpl_fk, and pack the 135-D body via h2wb.representations.body.
"""

from __future__ import annotations

import sys

import numpy as np

from . import smpl_fk as FK
from ..representations import body as B


def load_smpl_pkl(path: str):
    """joblib.load with the numpy._core compat shim newer pickles need. Returns the dict."""
    import joblib
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") != "numpy._core":
            raise
        import numpy.core as np_core
        sys.modules.setdefault("numpy._core", np_core)
        if hasattr(np_core, "multiarray"):
            sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        return joblib.load(path)


def _to_seq_list(value, key):
    if isinstance(value, list):
        return [np.asarray(v) for v in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [np.asarray(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):                 # a single stacked sequence
        return [value]
    raise ValueError(f"`{key}` must be a list/obj-array/ndarray of sequences, got {type(value)}")


def iter_sequences(payload: dict):
    """Yield dicts {poses (T,66+), trans (T,3), ...} for each sequence in the payload."""
    trans_list = _to_seq_list(payload["trans"], "trans")
    poses_list = _to_seq_list(payload["poses"], "poses")
    if len(trans_list) != len(poses_list):
        raise ValueError(f"trans/poses length mismatch: {len(trans_list)} vs {len(poses_list)}")
    for i, (tr, po) in enumerate(zip(trans_list, poses_list)):
        yield {"idx": i, "trans": np.asarray(tr, np.float64), "poses": np.asarray(po, np.float64)}


def _poses_to_72(poses: np.ndarray) -> np.ndarray:
    """Pad/truncate axis-angle poses to the 72-dim (24-joint) layout for the FK helpers."""
    T, D = poses.shape
    if D >= 72:
        return poses[:, :72]
    out = np.zeros((T, 72), np.float64)
    out[:, :D] = poses                                # 66 -> 72 (joints 22,23 = 0)
    return out


def sequence_to_pair(seq: dict, fps: float = 30.0, joints_fn=None, betas=None):
    """One sequence -> (hand12 (T,12), body (T,135)). FK-extract the left-wrist 12D."""
    poses72 = _poses_to_72(seq["poses"])
    trans = seq["trans"]
    betas = np.zeros(10) if betas is None else np.asarray(betas)
    joints_fn = joints_fn or FK.synthetic_joints_fn
    hand = FK.extract_hand12(poses72, trans, betas, joints_fn, fps=fps)
    body = B.smpl72_to_motion(poses72, trans)
    return hand.astype(np.float32), body.astype(np.float32)


def pkl_to_clips(path: str, fps: float = 30.0, joints_fn=None, min_frames: int = 2):
    """Load train.pkl and return a list of (hand12, body) clips ready for training/dataset."""
    payload = load_smpl_pkl(path)
    clips = []
    for seq in iter_sequences(payload):
        if seq["poses"].shape[0] < min_frames:
            continue
        clips.append(sequence_to_pair(seq, fps=fps, joints_fn=joints_fn))
    return clips
