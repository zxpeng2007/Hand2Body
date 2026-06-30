"""Export the 3-way foot-sliding comparison on one held-out clip, each carrying the GT wrist.

  baseline      : diffusion_full output (no foot fix)
  baseline + A  : baseline, then post-process foot-lock IK
  B             : diffusion_footloss output (trained with the foot-contact loss)

Each .npz has poses (T,72), trans (T,3), wrist (T,3) = the input-12D / ground-truth left wrist,
so scripts/render_aitviewer.py --ghost-wrist overlays the target wrist on every panel.
Prints per-model wrist deviation (mm vs the GT wrist) and foot-skate (mm/s).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from h2b.data.cache import load_pairs_cache, clip_wrist_activity
from h2b.eval import split_clips
from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
from h2b.models import fk_torch as FKt
from h2b import inference as INF
from h2b.export.footlock import footlock, foot_skate
from h2b.representations import body as B
from h2b.representations import rotations_torch as RT


def _gen_wrist(motion, rest):
    """Generated left-wrist (T,3) world position + (T,3,3) world rotation matrix."""
    wp, wr = FKt.left_wrist_pose(torch.tensor(np.asarray(motion, np.float32))[None],
                                 torch.as_tensor(rest, dtype=torch.float32))
    return wp[0].numpy(), RT.rotation_6d_to_matrix(wr[0]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--baseline-ckpt", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--b-ckpt", default="checkpoints/diffusion_footloss.pt")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    target = int(args.seconds * 30)
    clips, rest = load_pairs_cache(args.cache)
    _, val = split_clips(clips, val_frac=0.1, seed=0)
    lens = np.array([len(c[0]) for c in val])
    acts = np.array([clip_wrist_activity(c) for c in val])
    elig = np.where(lens >= target)[0]
    idx = int(elig[np.argmax(acts[elig])]) if len(elig) else int(np.argmax(lens))
    hand = np.asarray(val[idx][0][:target], np.float32)
    wrist_gt = hand[:, 0:3]
    wrist_gt_R = RT.rotation_6d_to_matrix(torch.from_numpy(hand[:, 6:12])).numpy()  # GT wrist orient
    print(f"clip {idx}: {len(hand)} frames")

    diff = GaussianDiffusion(device=dev)

    def load(ckpt):
        m = DiTDenoiser(hidden=256, n_layers=4).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev))
        return m

    base = INF.generate_long(load(args.baseline_ckpt), hand, arch="diffusion",
                             diffusion=diff, sample_steps=8, device=dev)
    baseA, _ = footlock(base, hand=hand, rest_joints=rest, iters=args.iters, device=dev)
    bmot = INF.generate_long(load(args.b_ckpt), hand, arch="diffusion",
                             diffusion=diff, sample_steps=8, device=dev)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'panel':16s} {'wrist dev (mm)':>14s} {'foot-skate (mm/s)':>18s}")
    for name, m in (("baseline", base), ("baselineA", baseA), ("B", bmot)):
        poses, trans = B.motion_to_smpl72(m)
        gen_pos, gen_R = _gen_wrist(m, rest)
        np.savez(os.path.join(args.out_dir, f"tw_{name}.npz"),
                 poses=np.asarray(poses, np.float32), trans=np.asarray(trans, np.float32),
                 wrist=wrist_gt, wrist_R=wrist_gt_R, gen_wrist=gen_pos, gen_wrist_R=gen_R)
        dev = float(np.linalg.norm(gen_pos - wrist_gt, axis=-1).mean() * 1000.0)
        print(f"{name:16s} {dev:14.1f} {1000*foot_skate(m, rest_joints=rest):18.1f}")
    print(f"\nwrote tw_baseline.npz / tw_baselineA.npz / tw_B.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
