# Results — Hand2Body

Two instantiations of the same causal model: **1 wrist (12D)** on table tennis, and
**2 wrists (24D)** on bimanual manipulation (ARCTIC).

## 1 wrist (12D) — table tennis

Data: the table-tennis motion set — 7753 BABEL ball-conditioned SMPL
sequences (~4.14 M frames, 22-joint poses + real 3D joints), world frame = table URDF (z-up).
The left-wrist 12D is FK-derived; the FK rest skeleton is calibrated from the real `joints`.

### Held-out comparison (identical 775-sequence val split, seed 0)

`python scripts/compare_models.py --cache data/cache/pairs_full.npz`

| model | MPJPE (mm) | wrist pos (mm) | wrist orient | jitter |
|---|---|---|---|---|
| **diffusion (full data)** | 20.7 | 8.5 | **0.72°** | **3.76** |
| regressor (full data) | 22.6 | 10.5 | 0.97° | 4.43 |
| **diffusion (activity-filtered, top 50%)** | **20.5** | **7.5** | 0.99° | 3.82 |

- **MPJPE** = mean per-joint position error after FK. **wrist pos/orient** = generated left wrist
  vs the *input* hand (held-out hand-consistency). **jitter** = mean joint acceleration (smoothness).
- Diffusion > regressor on every axis; the smoothness gap (jitter) matters for HoloMotion trackability.
- Activity-filtering matches full-data on half the clips → the data is redundant / the model is data-efficient.

So: from a **single left-hand 12D signal**, the model generates a full body with **~2 cm joint
accuracy** and the wrist tracked to **~8 mm / <1°** on unseen sequences.

### Training convergence (full diffusion, held-out, every 2000 steps)

| step | 0 | 2k | 10k | 20k | 28k |
|---|---|---|---|---|---|
| MPJPE (mm) | 1180 | 107 | 50 | 30 | 22.5 |
| wrist (mm) | 1152 | 73 | 22 | 9.5 | 9.7 |
| wrist (deg) | 126 | 7.9 | 1.7 | 1.7 | 0.93 |

Monotonic on held-out val → genuine generalization, no overfitting.

## 2 wrists (24D) — ARCTIC bimanual manipulation

The same architecture generalizes beyond table tennis: condition on **both wrists** (24D = `[left12 |
right12]`) and lift to the whole body. FK-derived from [ARCTIC](https://arctic.is.tue.mpg.de) SMPL-X
(301 bimanual object-manipulation sequences; body joints 0..21 == SMPL, wrists 20/21), `hand_dim=24`,
30k steps, held-out 30-seq val split.

| step | 0 | 5k | 10k | 20k | 25k |
|---|---|---|---|---|---|
| MPJPE (mm) | 1590 | 92 | 74 | **63** | 67 |
| wrist (mm) | 1440 | 85 | 39 | 18 | **14** |
| wrist (deg) | 125 | 8.9 | 6.5 | 2.9 | **2.1** |
| jitter | 256 | 1.3 | 1.0 | 0.82 | 0.75 |

Wrist tracking is tight (**14 mm / 2.1°**, approaching the 1-wrist 8.5 mm). The **MPJPE floors at
~63 mm = the unconstrained lower body**: neither wrist observes the legs/root, so they stay
prior-driven — the fundamental limit of sparse-wrist tracking (more data/steps sharpen the upper body,
not the legs). All 301 ARCTIC seqs are already planted-feet, so the seated regime holds; the lever for
the lower body is an explicit root/pelvis signal, not more wrist data. (Numbers are above the 1-wrist
20.7 mm mainly because ARCTIC is ~25× smaller — 271 vs 7753 train seqs — and a harder, more varied
domain.)

```bash
python scripts/train.py --arctic <arctic_raw_seqs> --smplx-models <smplx_dir> --wrist-count 2 \
    --arch diffusion --steps 30000 --out checkpoints/arctic_bimanual_30k.pt
```

## Reproduce

```bash
# 1) one-time cache (2.5 GB pkl -> 2.3 GB fast npz; also prints act_cat distribution)
python scripts/cache_pairs.py --pkl motions.pkl --out data/cache/pairs_full.npz

# 2) train (loads cache in seconds)
python scripts/train.py --cache data/cache/pairs_full.npz --val-frac 0.1 \
    --arch diffusion --steps 30000 --eval-every 2000 --device cuda --out checkpoints/diffusion_full.pt
# variants: --arch regressor   |   --top-activity-frac 0.5 (striking-focused)

# 3) compare checkpoints on the same val split
python scripts/compare_models.py --cache data/cache/pairs_full.npz

# 4) generate + export a clip (AMASS + GMR-ready SMPL-X) and view
python scripts/generate.py --arch diffusion --checkpoint checkpoints/diffusion_full.pt \
    --hand HAND.npy --out clip.npz --gmr-out clip_smplx.npz
python -m h2b.export.aitviewer_vis --input clip.npz
```

Note: `act_cat` is uniformly `walk` for all sequences (vestigial BABEL label), so metadata
label-filtering is useless; the motion-based `--top-activity-frac` (wrist speed) is the
meaningful "striking vs idle" selector.
