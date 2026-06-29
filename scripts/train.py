"""Training entrypoint (M2/M4). Stub: wires config -> dataset -> model -> loop.

    python scripts/train.py --config configs/default.yaml --pairs data/pairs

The training loop body is implemented once torch is in the venv and paired data exists.
Kept minimal here so the CLI/plumbing is reviewable now.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import yaml


def load_pairs(pairs_dir: str):
    clips = []
    for fp in sorted(glob.glob(os.path.join(pairs_dir, "*.npz"))):
        d = np.load(fp)
        clips.append((d["hand12"], d["body"]))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    clips = load_pairs(args.pairs)
    print(f"loaded {len(clips)} clips; arch={cfg['model']['arch']}")

    # TODO(M2/M4):
    #   from h2wb.data.dataset import Hand2BodyDataset
    #   from h2wb.models.diffmlp import DiffMLP   (or models.regressor for M2)
    #   build dataset with cfg['window'], train with h2wb.losses.* and cfg['loss'] weights.
    raise SystemExit("training loop not yet wired — see TODO(M2/M4) in scripts/train.py")


if __name__ == "__main__":
    main()
