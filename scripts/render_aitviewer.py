"""Headless aitviewer render -> mp4 of the SMPL MESH (proper body, not a skeleton).

Requires the licensed SMPL model files on disk (download from smpl.is.tue.mpg.de /
smpl-x.is.tue.mpg.de). Point aitviewer at them with --smpl-models DIR (a folder containing
`smpl/` and/or `smplx/` with the model .pkl/.npz). aitviewer's headless GL renderer works
in this environment; only the model files are missing.

    # render our exported clip as a mesh video:
    python scripts/render_aitviewer.py --input Downloads\h2wb_demo.npz \
        --smpl-models C:\path\to\body_models --out Downloads\h2wb_mesh.mp4

    # or generate a fresh held-out clip from a checkpoint, then render:
    python scripts/render_aitviewer.py --cache data/cache/pairs_full.npz \
        --checkpoint checkpoints/diffusion_full.pt --smpl-models C:\path\to\body_models \
        --out Downloads\h2wb_mesh.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="our AMASS-style .npz (poses (T,72), trans)")
    ap.add_argument("--cache", default="", help="generate from a held-out cache clip instead")
    ap.add_argument("--checkpoint", default="checkpoints/diffusion_full.pt")
    ap.add_argument("--arch", default="diffusion", choices=["diffusion", "regressor"])
    ap.add_argument("--smpl-models", default="", help="dir with SMPL/SMPL-X model files (sets aitviewer config)")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--model-type", default="smpl", choices=["smpl", "smplx"])
    ap.add_argument("--out", default="h2wb_mesh.mp4")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    args.max_frames = min(args.max_frames, 256)            # model positional budget (single-shot)

    if args.smpl_models:
        from aitviewer.configuration import CONFIG as C
        try:
            C.update_conf({"smplx_models": args.smpl_models})
        except Exception:
            C.smplx_models = args.smpl_models

    # obtain (poses (T,72), trans (T,3))
    if args.input:
        d = np.load(args.input, allow_pickle=True)
        poses, trans = np.asarray(d["poses"]), np.asarray(d["trans"])
    elif args.cache:
        import torch
        from h2wb.data.cache import load_pairs_cache, clip_wrist_activity
        from h2wb.eval import split_clips
        from h2wb.representations import body as B
        from h2wb import inference as INF
        from h2wb.models.diffusion import DiTDenoiser, GaussianDiffusion
        from h2wb.models.regressor import RegressorHand2Body
        clips, _ = load_pairs_cache(args.cache)
        _, val = split_clips(clips, val_frac=0.1, seed=0)
        lens = np.minimum([len(c[0]) for c in val], args.max_frames)
        idx = int(np.argmax(np.array([clip_wrist_activity(c) for c in val]) * lens))
        hand = val[idx][0][:args.max_frames]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if args.arch == "diffusion":
            model = DiTDenoiser(hidden=256, n_layers=4).to(dev); diff = GaussianDiffusion(device=dev)
        else:
            model = RegressorHand2Body(hidden=256, n_layers=4).to(dev); diff = None
        model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
        motion = INF.generate(model, hand, arch=args.arch, diffusion=diff, sample_steps=8, device=dev)
        poses, trans = B.motion_to_smpl72(motion)
    else:
        raise SystemExit("pass --input <npz> or --cache <npz> + --checkpoint")

    from aitviewer.headless import HeadlessRenderer
    from h2wb.export.aitviewer_vis import _table_mesh, _net_mesh, _smpl_sequence
    seq = _smpl_sequence(poses[:args.max_frames], trans[:args.max_frames], args.gender, args.model_type)
    r = HeadlessRenderer()
    r.scene.add(_table_mesh()); r.scene.add(_net_mesh()); r.scene.add(seq)
    # aitviewer's default camera auto-frames the scene; orbit interactively for other angles.
    r.save_video(video_dir=args.out, output_fps=args.fps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
