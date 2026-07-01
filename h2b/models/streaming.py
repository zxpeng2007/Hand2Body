"""Causal online generation wrapper around the M4 diffusion model.

Maintains a short sliding window of the most recent hand frames (world frame). Each step it
canonicalizes the window by its first-frame hand position (the inference-time anchor, matching
training -- see dataset.canonicalize_window), DDIM-samples the body over the window, and emits
the newest body frame(s) de-canonicalized back to world. The causal denoiser guarantees the
latest frame depends only on frames <= t, so this is valid online.

Two modes:
  * push(hand)            -> emit the single newest body frame (1 DDIM sample / output frame).
  * push_block(hand_blk)  -> append a block of B frames, DDIM-sample the window ONCE, emit the
                             newest B body frames. The sample is amortized over the block, so
                             per-frame cost drops ~B x -- the headroom for the downstream
                             GMR + HoloMotion stages.

Window/latency/quality (measured, RTX 5080, trained model):
  * Cost is launch-bound: the per-sample time is ~flat for window 5..32 (~5 ms at ddim=2 warm),
    so a LARGER window is effectively free. With block=4 that is ~1.2 ms/output-frame (~4% of
    the 133 ms 4-frame budget at 30 fps).
  * Quality, though, depends on the window: a very short window is jerky at the block seams
    (jitter ~12 at w=5 vs ~6 at w=16 vs offline ~3.6); wrist tracking stays tight (~8 mm) until
    the window grows long enough to re-introduce anchor drift (>~20). Sweet spot ~window=16.
  * Output smoothing was tried and rejected: a 1-Euro filter on the whole body lags the extended
    wrist badly (8 mm -> 100 mm+). Leg-ONLY smoothing (tried 2026-06-30) cuts leg-rotation jitter
    ~80% with the wrist left exact, BUT it strips the micro-corrections that keep the feet planted
    -> bad FOOT SLIDING, so it was reverted too. The fix is the window size, not a post-filter.

Speed pass (investigated 2026-06-30, RTX 5080 laptop / torch 2.11 / Windows):
  * The streamer is LAUNCH-bound -- per-sample time is ~flat for window 5..32 -- so a classic
    KV-cache (which cuts attention FLOPs) would NOT help. The realized win is the block emission
    above (one DDIM sample serves B frames -> ~B x fewer samples).
  * The launch-overhead killers are blocked on THIS stack: torch.compile(reduce-overhead) needs
    Triton (absent on Windows), and CUDA-graph capture of nn.TransformerEncoder is rejected
    (cudaErrorStreamCaptureInvalidated, even a single forward). Both are viable on a Linux deploy
    (the robot target), so they stay as deploy-time options, not code here.
  * ddim_sample was made host-sync-free (no int(t) per step) so it pipelines and is graph/compile
    -capturable where those work. On this box each push is already ~5 ms (ddim=2) = ~4% of the
    133 ms block budget, i.e. ~96% headroom -- at the practical hardware floor.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..representations import frames as F
from ..representations import body as B


class DiffusionStreamer:
    """Online hand->body generator. push() one frame, or push_block() a block of frames."""

    def __init__(self, model, diffusion, window: int = 16, block: int = 4,
                 sample_steps: int = 2, device="cpu"):
        self.model = model
        self.diff = diffusion
        self.window = window
        self.block = block
        self.sample_steps = sample_steps
        self.device = device
        self._hand = deque(maxlen=window)

    def reset(self):
        self._hand.clear()

    def _sample_window(self):
        """DDIM-sample the buffered window -> (L,135) world body for the whole window."""
        import torch
        hand = np.stack(self._hand)[None]                       # (1, L, 12*N)
        anchor = hand[:, 0:1, 0:3].copy()                       # (1,1,3) first-wrist anchor
        hand_c = hand.copy()
        for s in F.hand_pos_slices(hand.shape[-1]):             # shift every wrist's position block
            hand_c[..., s] -= anchor
        ht = torch.from_numpy(hand_c).to(self.device)
        body = self.diff.ddim_sample(self.model, (1, ht.shape[1], B.MOTION_DIM), ht,
                                     steps=self.sample_steps, device=self.device).cpu().numpy()[0]
        body[:, B.B_TRANS] += anchor[0, 0]                      # de-canonicalize trans -> world
        return body                                            # (L, 135)

    def push_block(self, hand_block):
        """Append a block of B world-frame 12D samples, DDIM-sample the window ONCE, and return
        the newest min(B, buffered) body frames (B,135) world. One sample serves the whole block
        -> ~B x lower per-frame cost. Returns None only if nothing is buffered."""
        hb = np.asarray(hand_block, np.float32).reshape(-1, 12)
        for h in hb:
            self._hand.append(h)
        if not self._hand:
            return None
        body = self._sample_window()
        k = min(hb.shape[0], body.shape[0])
        return body[-k:].copy()                                # (k, 135)

    def push(self, hand12_world):
        """Single-frame online step: append one 12D frame, re-sample, emit the latest body
        frame (135,). Returns None until >=2 frames are buffered (needs a little context)."""
        self._hand.append(np.asarray(hand12_world, np.float32))
        if len(self._hand) < 2:
            return None
        return self._sample_window()[-1].copy()
