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
                             per-frame cost drops ~B x -- this is what leaves headroom for the
                             downstream GMR + HoloMotion stages.

Defaults (window=5, block=4) keep the per-block sample tiny; bump `window` for more overlap
(smoother block seams) or drop `sample_steps` for lower latency. The window is re-sampled each
step (simple + correct); a KV-cache / block-streaming cache is a later speed pass -- TODO(perf).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..representations import frames as F
from ..representations import body as B


class DiffusionStreamer:
    """Online hand->body generator. push() one frame, or push_block() a block of frames."""

    def __init__(self, model, diffusion, window: int = 5, block: int = 4,
                 sample_steps: int = 4, device="cpu"):
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
        hand = np.stack(self._hand)[None]                       # (1, L, 12)
        anchor = hand[:, 0:1, F.HAND12_POS].copy()              # (1,1,3) inference anchor
        hand_c = hand.copy()
        hand_c[..., F.HAND12_POS] -= anchor
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
