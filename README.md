# Hand2WholeBody

Generate **whole-body SMPL motion** from a **single left-hand 12D signal**, for the
table-tennis humanoid pipeline:

```mermaid
flowchart LR
    subgraph UP["Stage 1 · Upstream (coworker)"]
        HG["Table-tennis<br/>hand generator"]
    end
    subgraph REPO["Stage 2 · Hand2WholeBody (this repo)"]
        H2B["Causal diffusion model<br/>12D hand ➜ whole-body SMPL"]
    end
    subgraph DOWN["Stage 3 · Retarget + track (Linux / robot)"]
        GMR["GMR<br/>SMPL ➜ Unitree G1"]
        HM["HoloMotion<br/>whole-body tracker"]
        G1(["Unitree G1<br/>29-DoF humanoid"])
    end
    HG -->|"12D / frame<br/>pos · vel · 6D rot<br/>30 Hz · world frame"| H2B
    H2B -->|"SMPL-X .npz<br/>30 Hz"| GMR
    GMR -->|"G1 dof + root<br/>.pkl"| HM
    HM -->|"joint targets<br/>50 Hz PD"| G1

    classDef up fill:#fff3e0,stroke:#fb8c00,color:#111;
    classDef repo fill:#e3f2fd,stroke:#1e88e5,color:#111,stroke-width:2px;
    classDef down fill:#e8f5e9,stroke:#43a047,color:#111;
    class HG up
    class H2B repo
    class GMR,HM,G1 down
```

- **Input** (per frame, left wrist = SMPL joint 20): `[pos(3), lin_vel(3), rot6D(6)]`,
  world frame, **global** wrist orientation. Forehand/backhand is encoded by that
  orientation.
- **Output**: AMASS-style SMPL `.npz` at 30 Hz (plain SMPL — rigid wrist, no fingers).
- **Causal / streaming** (real-time on the robot).

👉 **Read [`docs/CONTRACT.md`](docs/CONTRACT.md) first** — it pins the world frame, the 12D
semantics, the SMPL output format, and the open questions for the coworker. All code
constants come from [`configs/default.yaml`](configs/default.yaml).

## Model & I/O

```mermaid
flowchart TB
    IN["12D left-hand signal · per frame<br/>position (3) · velocity (3) · 6D rotation (6)<br/>world frame · global wrist orientation"]
    WIN["Causal window<br/>last K frames · anchor on hand-start<br/>orientation kept global"]
    DENOISE["Conditional diffusion · DiT denoiser<br/>causal transformer · DDIM sampling<br/>predicts clean body x0"]
    MOTION["135-D body / frame<br/>root translation (3) + 22 joints × 6D (132)"]
    SMPL["SMPL pose (72) + translation"]
    EXP["Export<br/>AMASS .npz · GMR-ready SMPL-X .npz"]
    LOSS["Training losses<br/>recon · 6D rotation · FK joints<br/>velocity · hand-consistency"]
    IN --> WIN --> DENOISE --> MOTION --> SMPL --> EXP
    LOSS -.->|"supervise x0"| DENOISE

    classDef io fill:#e3f2fd,stroke:#1e88e5,color:#111;
    classDef core fill:#ede7f6,stroke:#5e35b1,color:#111,stroke-width:2px;
    classDef aux fill:#fafafa,stroke:#9e9e9e,color:#111;
    class IN,MOTION,SMPL,EXP io
    class WIN,DENOISE core
    class LOSS aux
```

## Training & data flow

```mermaid
flowchart LR
    PKL[("train.pkl<br/>7753 seqs · ~4.1M frames<br/>SMPL 22-joint + real joints")]
    FK["FK extract<br/>left-wrist 12D + 135-D body<br/>calibrate rest skeleton"]
    CACHE[("pairs cache<br/>2.3 GB · mmap")]
    SPLIT["windowed dataset<br/>train / held-out val split"]
    TRAIN["train<br/>diffusion / regressor"]
    CKPT["checkpoint"]
    EVAL["held-out eval<br/>MPJPE · wrist · jitter"]
    VIZ["render<br/>skeleton + aitviewer mesh"]
    PKL --> FK --> CACHE --> SPLIT --> TRAIN --> CKPT
    CKPT --> EVAL
    CKPT --> VIZ

    classDef data fill:#fff8e1,stroke:#f9a825,color:#111;
    classDef proc fill:#e3f2fd,stroke:#1e88e5,color:#111;
    classDef out fill:#e8f5e9,stroke:#43a047,color:#111;
    class PKL,CACHE data
    class FK,SPLIT,TRAIN proc
    class CKPT,EVAL,VIZ out
```

## Status (2026-06-30)

| Milestone | State |
|-----------|-------|
| **M0** repo + env + frozen contract | ✅ scaffold, world frame, configs; torch 2.11+cu128 on GPU |
| **M1** SMPL→12D FK extractor + cycle-consistency | ✅ `h2wb/data/smpl_fk.py`, verified on synthetic data |
| **M2** deterministic regressor baseline | ✅ causal transformer + FK losses + training loop, verified on GPU |
| **M3** train on real data (train.pkl) | ✅ diffusion: **20.7 mm** held-out MPJPE, wrist ~8 mm / <1° — see [results.md](docs/results.md) |
| **M4** streaming diffusion (primary) | ✅ causal DiT denoiser + DDIM + streaming, verified on GPU; SAGE/distillation = polish |
| **M5** close the loop: SMPL→GMR→MuJoCo→HoloMotion | ◐ GMR-ready SMPL-X export + verified [runbook](docs/stage3_runbook.md); robot run is on user's Linux/G1 |
| **M6** domain-gap hardening | ⬜ |

6D rotation convention is **confirmed** = Zhou-2019 columns (`frames.PROJECT_R6D`). The
models train hand[1..L]→body[1..L] causally; run `python scripts/train.py --synthetic` to
smoke-test the loop without data.

**Real data** (`train.pkl`, joblib, SMPL 22-joint `poses [T,66]` + `trans`):
```
python scripts/train.py --pkl train.pkl --arch diffusion --steps 20000   # FK-extracts the 12D internally
python -m h2wb.export.aitviewer_vis --input train.pkl --seq_idx 0         # view raw data (aitviewer)
python scripts/generate.py --arch diffusion --checkpoint checkpoints/diffusion.pt --hand H.npy --out out.npz --viz out.png
python -m h2wb.export.aitviewer_vis --input out.npz                       # view a generated clip
```
**Mesh visualization (aitviewer).** One-time setup for the SMPL body-mesh render: download
the official SMPL models (`smpl.is.tue.mpg.de`), then convert + render:
```
python -m uv pip install --python .venv --no-build-isolation chumpy   # one-time, for the conversion
python scripts/clean_smpl_models.py --src .../SMPL_python_v.1.1.0/smpl/models --out .../smpl_models
python scripts/render_aitviewer.py --cache data/cache/pairs_full.npz \
    --checkpoint checkpoints/diffusion_full.pt --smpl-models .../smpl_models --out mesh.mp4
```
`clean_smpl_models.py` converts the chumpy/numpy-1 release into the `SMPL_{GENDER}.pkl` layout
smplx/aitviewer expect (and that works under numpy 2.x). The matplotlib `h2wb.export.visualize`
+ `scripts/render_video.py` are a schematic, dependency-light headless fallback (no models needed).

The representation core, FK extractor, the regressor + conditional diffusion models, losses,
dataset, caching, evaluation, and export are implemented and **tested** (`pytest` — 75 passing,
incl. overfit, generative, and FK-parity tests). Trained on real data (see [results.md](docs/results.md));
remaining is **M6** (domain-gap hardening) and the on-robot Stage-3 run on the user's Linux/G1.

## Layout

```
assets/urdf/        ball / table / g1_29dof_rev_1_0_pingpong  (world frame source of truth)
configs/            default.yaml
docs/CONTRACT.md    inter-stage data contract  ← the important doc
h2wb/
  representations/  rotations.py (6D/aa/quat, pluggable convention), frames.py (world, SMPL, 12D)
  data/             smpl_fk.py (SMPL→12D), dataset.py (windowed causal — stub)
  models/           diffmlp.py / streaming.py / regressor.py (interfaces)
  losses.py         recon + 6D + FK + jitter + hand-consistency + foot-contact
  export/           to_amass_npz.py (SMPL→AMASS .npz for GMR/HoloMotion)
scripts/            setup_env.ps1, extract_amass.py, train.py
tests/              representation + FK tests (run with pytest)
```

## Setup

The system Python is 3.14 (no torch wheels yet). Use the project venv (Python 3.12, via
`uv`) — already created at `.venv`:

```powershell
# core (done): numpy, pyyaml, pytest, editable install
# heavy ML stack (Blackwell GPU → cu128 torch):
python -m uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 torch
python -m uv pip install --python .venv -e ".[train,dev]"   # smplx, trimesh, tqdm, ...

# run tests
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m pytest -q
```

> Note: GMR / HoloMotion themselves run best under Linux/WSL2. Hand2WholeBody training
> and the SMPL export are platform-independent; the Stage-3 retarget happens downstream.
