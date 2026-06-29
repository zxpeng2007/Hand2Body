"""Causal streaming rollout wrapper (DART-style) around a per-step denoiser.

Turns the windowed DiffMLP denoiser into a real-time generator: maintain ring buffers
of the last K hand frames and last Kb body frames; each step, sample the next P body
frames via a few DDIM/flow steps, emit them, and shift the buffers. This is what makes
Stage 2 causal/real-time per CONTRACT §4.

Interface only — the sampling loop is wired once the diffusion schedule (diffusion.py)
and a trained DiffMLP checkpoint exist (M4).
"""

from __future__ import annotations

from collections import deque


class StreamingHand2Body:
    """Online generator: push one 12D hand frame, pop body frames as they are produced."""

    def __init__(self, denoiser, k_hand: int, k_body: int, p_predict: int, sample_steps: int = 8):
        self.denoiser = denoiser
        self.k_hand = k_hand
        self.k_body = k_body
        self.p_predict = p_predict
        self.sample_steps = sample_steps
        self._hand = deque(maxlen=k_hand)
        self._body = deque(maxlen=k_body)
        self._anchor = None

    def reset(self, anchor_xyz=None):
        self._hand.clear()
        self._body.clear()
        self._anchor = anchor_xyz

    def push_hand(self, hand12):
        """Append one canonicalized 12D hand frame to the conditioning buffer."""
        self._hand.append(hand12)

    def step(self):
        """Sample and return the next `p_predict` body frames (world frame).

        TODO(M4): DDIM/flow sampling loop over self.denoiser using the current buffers,
        then de-canonicalize positions with self._anchor and append to self._body.
        """
        raise NotImplementedError("wire DDIM/flow sampling once DiffMLP is trained (M4)")
