"""Generalization probe: synthetic wrist trajectories (big Z, circle, figure-8, high-low) -> body.

The shapes are placed inside the training wrist workspace (calibrated from the pairs cache:
position box, speeds, a typical orientation) but are geometrically unlike anything in the data.
Metrics answer "does the wrist still track?" and "what does the body prior invent?".

Every trajectory starts with a 1 s lead-in ease from the workspace-center height. This is load-
bearing, not cosmetic: canonicalization (CONTRACT §5) anchors positions to the FIRST wrist frame,
so absolute height is unobservable and the model reconstructs the floor as if that frame sat at
the data-typical wrist height. A probe starting at the top of the Z floats the whole body by
(start_z - typical_z); the lead-in pins the anchor frame to the typical height instead.

    # 1-wrist table-tennis model
    python scripts/shape_probe.py --cache data/cache/pairs_full.npz \
        --checkpoint checkpoints/diffusion_full.pt
    # 2-wrist ARCTIC model, with mesh videos (drawing hand = wrist overlays, idle hand = ghosts)
    python scripts/shape_probe.py --cache data/cache/arctic_bimanual.npz \
        --checkpoint checkpoints/arctic_bimanual_30k.pt --wrist-count 2 \
        --render --smpl-models <smpl_models_dir> --out-dir probe_out
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h2b.representations import frames as F
from h2b.representations import body as B
from h2b.representations import rotations as R

FPS = 30
LEAD = 30                       # 1 s anchor lead-in (see module docstring)
SHAPE_NAMES = ("zshape", "circle", "fig8", "highlow")


# ---------------------------------------------------------------- workspace calibration

def wrist_workspace(hand_frames: np.ndarray, off: int) -> dict:
    """Per-wrist workspace stats from cached hand frames (N, 12*K): position box (p5..p95),
    the drawing plane (vertical z + the wider horizontal axis), speed p90, a typical rot6d."""
    pos = hand_frames[:, off:off + 3]
    vel = hand_frames[:, off + 3:off + 6]
    lo, hi = np.percentile(pos, 5, axis=0), np.percentile(pos, 95, axis=0)
    u = 0 if (hi - lo)[0] >= (hi - lo)[1] else 1           # wider horizontal axis: x or y
    speed = np.linalg.norm(vel, axis=1)
    typical = hand_frames[int(np.argmin(np.abs(speed - np.median(speed))))]
    return {
        "center": 0.5 * (lo + hi), "half": 0.5 * (hi - lo),
        "u_axis": u, "held_axis": 1 - u,
        "speed_p90": float(np.percentile(speed, 90)),
        "rot6d": typical[off + 6:off + 12].astype(np.float32),
    }


# ---------------------------------------------------------------- shapes (unit coords, |u|,|v| <= 1)

def _ease(a, b, n):
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    return a[None] + s[:, None] * (b - a)[None]


def _hold(p, n):
    return np.repeat(p[None], n, 0)


def _zshape(ts):
    p1, p2 = np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    p3, p4 = np.array([-1.0, -1.0]), np.array([1.0, -1.0])
    n1, n2 = int(0.25 * ts), int(0.29 * ts)
    h = int(0.03 * ts)
    used = 2 * h + 2 * n1 + n2 + int(0.05 * ts)
    return np.concatenate([
        _hold(p1, int(0.05 * ts)), _ease(p1, p2, n1), _hold(p2, h),
        _ease(p2, p3, n2), _hold(p3, h), _ease(p3, p4, n1), _hold(p4, ts - used),
    ])


def _circle(ts):
    th = np.linspace(0.0, 2 * np.pi * 2, ts)
    return np.stack([np.sin(th), np.cos(th)], 1)


def _fig8(ts):
    w = 2 * np.pi * 1.5 / ts
    t = np.arange(ts)
    return np.stack([np.sin(w * t), np.sin(2 * w * t)], 1)


def _highlow(ts):
    w = 2 * np.pi * 2 / ts
    t = np.arange(ts)
    return np.stack([0.3 * np.sin(0.5 * w * t), np.sin(w * t - np.pi / 2)], 1)


_SHAPES = {"zshape": _zshape, "circle": _circle, "fig8": _fig8, "highlow": _highlow}


def build_hand(name: str, ws: list[dict], ts: int, fill: float = 0.85) -> np.ndarray:
    """(ts+LEAD, 12*len(ws)) probe signal. Wrist 0 draws `name` in its (u, z) plane scaled to
    `fill` of the workspace box; further wrists idle near their centers with a gentle sway.
    Velocities use the training convention (finite_diff_velocity); orientation is held typical."""
    blocks = []
    for k, w in enumerate(ws):
        t_all = ts + LEAD
        if k == 0:
            uv = _SHAPES[name](ts) * fill
            uv = np.concatenate([_ease(np.zeros(2), uv[0], LEAD), uv])
            pos = np.repeat(w["center"][None], t_all, 0).astype(np.float64)
            pos[:, w["u_axis"]] += uv[:, 0] * w["half"][w["u_axis"]]
            pos[:, 2] += uv[:, 1] * w["half"][2]
        else:
            sway = 0.02 * np.sin(2 * np.pi * 0.15 * np.arange(t_all) / FPS)
            pos = np.repeat(w["center"][None], t_all, 0).astype(np.float64)
            pos[:, w["u_axis"]] += sway
            pos[:, w["held_axis"]] += 0.5 * sway
        blocks.append(np.concatenate(
            [pos, F.finite_diff_velocity(pos, FPS), np.repeat(w["rot6d"][None], t_all, 0)], 1))
    return np.concatenate(blocks, 1).astype(np.float32)


# ---------------------------------------------------------------- probe + metrics

def run_probe(args):
    import torch
    from h2b.data.cache import load_pairs_cache
    from h2b.models.diffusion import DiTDenoiser, GaussianDiffusion
    from h2b.models import fk_torch as FKt
    from h2b import inference as INF

    clips, rest = load_pairs_cache(args.cache)
    hand_dim = 12 * args.wrist_count
    H = np.concatenate([np.asarray(h) for h, _ in clips[:400]], 0)
    if H.shape[-1] != hand_dim:
        raise SystemExit(f"cache hand_dim {H.shape[-1]} != 12*wrist_count {hand_dim}")
    ws = [wrist_workspace(H, 12 * k) for k in range(args.wrist_count)]

    dev = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = DiTDenoiser(hidden=256, n_layers=4, hand_dim=hand_dim).to(dev).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    diff = GaussianDiffusion(device=dev)
    rest_t = torch.as_tensor(rest, dtype=torch.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"rest": rest}
    print(f"{'shape':8s} {'w0 err mm mean/max':>19s} {'w0 deg':>6s} "
          f"{'others mm':>9s} {'root xy m':>9s} {'ankle z range':>15s} {'jitter':>7s}")
    for name in args.shapes:
        hand = build_hand(name, ws, args.shape_frames)
        motion = INF.generate(model, hand, arch="diffusion", diffusion=diff,
                              sample_steps=args.sample_steps, device=dev)
        mt = torch.as_tensor(motion, dtype=torch.float32)
        gR, jpos = FKt.motion_to_joints(mt, rest_t)
        p0 = jpos[:, F.WRIST_JOINTS[0]].numpy()
        R0 = gR[:, F.WRIST_JOINTS[0]].numpy()
        cmd_R = R.rotation_6d_to_matrix(hand[:, 6:12], convention=F.PROJECT_R6D)
        err0 = np.linalg.norm(p0 - hand[:, 0:3], axis=1) * 1000.0
        tr = np.clip((np.einsum("tij,tij->t", cmd_R, R0) - 1.0) * 0.5, -1.0, 1.0)
        deg0 = np.degrees(np.arccos(tr))
        errs = [np.linalg.norm(jpos[:, F.WRIST_JOINTS[k]].numpy()
                               - hand[:, 12 * k:12 * k + 3], axis=1).mean() * 1000.0
                for k in range(1, args.wrist_count)]
        root = motion[:, 0:3]
        travel = float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum())
        az = jpos[:, [7, 8], 2].numpy()                     # ankles
        acc = np.diff(jpos.numpy(), n=2, axis=0) * FPS * FPS
        jitter = float(np.linalg.norm(acc, axis=-1).mean())
        others = f"{np.mean(errs):8.1f}" if errs else "       -"
        print(f"{name:8s} {err0.mean():11.1f}/{err0.max():6.1f} {deg0.mean():6.1f} "
              f"{others} {travel:9.2f} [{az.min():+.3f},{az.max():+.3f}] {jitter:7.2f}")
        out[f"{name}_hand"] = hand
        out[f"{name}_motion"] = motion
    npz = os.path.join(args.out_dir, "shape_probe.npz")
    np.savez(npz, **out)
    print("saved", npz)

    if args.render:
        if not args.smpl_models:
            raise SystemExit("--render needs --smpl-models")
        for name in args.shapes:                            # fresh process per video: clean GL context
            subprocess.run([sys.executable, os.path.abspath(__file__), "--render-one", name,
                            "--npz", npz, "--smpl-models", args.smpl_models,
                            "--out-dir", args.out_dir] + (["--table"] if args.table else []),
                           check=True)


# ---------------------------------------------------------------- rendering (one shape per process)

def render_one(args):
    import torch
    from h2b.models import fk_torch as FKt
    d = np.load(args.npz)
    hand, motion, rest = d[f"{args.render_one}_hand"], d[f"{args.render_one}_motion"], d["rest"]
    outp = os.path.join(args.out_dir, f"probe_{args.render_one}.mp4")

    from aitviewer.configuration import CONFIG as C
    C.update_conf({"playback_fps": float(FPS), "smplx_models": args.smpl_models,
                   "window_width": 1600, "window_height": 1200})
    from aitviewer.headless import HeadlessRenderer
    from h2b.export.aitviewer_vis import (_smpl_sequence, _table_mesh, _net_mesh,
                                          _wrist_ghost, wrist_overlays)

    mt = torch.as_tensor(motion, dtype=torch.float32)
    gR, jpos = FKt.motion_to_joints(mt, torch.as_tensor(rest, dtype=torch.float32))
    p0 = jpos[:, F.WRIST_JOINTS[0]].numpy()
    R0 = gR[:, F.WRIST_JOINTS[0]].numpy()
    cmd_R = R.rotation_6d_to_matrix(hand[:, 6:12], convention=F.PROJECT_R6D)

    poses, trans = B.motion_to_smpl72(motion)
    r = HeadlessRenderer()
    r.scene.add(_smpl_sequence(poses, trans, "neutral", "smpl"))
    if args.table:
        r.scene.add(_table_mesh())
        r.scene.add(_net_mesh())
    for o in wrist_overlays(gt_pos=hand[:, 0:3], gt_R=cmd_R, gen_pos=p0, gen_R=R0):
        r.scene.add(o)
    for k in range(1, hand.shape[-1] // 12):                # idle wrists: command + generated ghosts
        r.scene.add(_wrist_ghost(hand[:, 12 * k:12 * k + 3], radius=0.035,
                                 color=(0.2, 0.4, 0.95, 0.75)))
        r.scene.add(_wrist_ghost(jpos[:, F.WRIST_JOINTS[k]].numpy(), radius=0.030,
                                 color=(0.2, 0.85, 0.4, 0.75)))
    c = jpos.reshape(-1, 3).mean(0).numpy()
    tgt = np.array([c[0], c[2], -c[1]], np.float32)         # viewer frame: (x, z, -y)
    cam = r.scene.camera
    cam.target = tgt
    cam.position = tgt + np.array([1.7, 0.55, 1.9], np.float32)
    cam.up = np.array([0.0, 1.0, 0.0], np.float32)
    for a in ("playback_fps", "_playback_fps"):
        if hasattr(r, a):
            try:
                setattr(r, a, FPS)
            except Exception:
                pass
    base = os.path.splitext(outp)[0]
    for f_ in glob.glob(base + "_*.mp4") + glob.glob(outp):  # save_video appends _N, never overwrites
        try:
            os.remove(f_)
        except OSError:
            pass
    r.save_video(video_dir=outp, output_fps=FPS)
    produced = sorted(glob.glob(base + "_*.mp4")) or [outp]
    if produced[0] != outp and os.path.exists(produced[0]):
        os.replace(produced[0], outp)
    print("wrote", outp, f"({len(motion)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/cache/pairs_full.npz")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--wrist-count", type=int, default=1, choices=(1, 2))
    ap.add_argument("--shapes", nargs="+", default=list(SHAPE_NAMES), choices=SHAPE_NAMES)
    ap.add_argument("--shape-frames", type=int, default=220,
                    help="frames per shape; +LEAD(30) must stay within the ~256-frame budget")
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--out-dir", default="probe_out")
    ap.add_argument("--render", action="store_true", help="also write aitviewer mesh videos")
    ap.add_argument("--smpl-models", default="", help="SMPL model dir (for --render)")
    ap.add_argument("--table", action="store_true", help="draw the table-tennis scene (12D model)")
    ap.add_argument("--render-one", default="", help=argparse.SUPPRESS)   # internal subprocess mode
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.shape_frames + LEAD > 250:
        raise SystemExit(f"--shape-frames {args.shape_frames} + lead-in {LEAD} exceeds the "
                         "single-shot positional budget (250)")
    if args.render_one:
        render_one(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
