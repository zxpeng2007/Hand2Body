"""Deterministic transformer regressor — Fallback 1 / the M2 baseline.

AvatarPoser/AvatarJLM-style: a temporal transformer over the past-hand window that
directly regresses the next body frame(s) of SMPL 6D pose + root. Fast, stable, lowest
latency; it averages in ambiguous phases (the known single-hand failure mode), so it is
the baseline and latency floor, not the final model. Share the loss terms in losses.py.

Interface stub — implement the encoder once torch is in the venv (M2).
"""

from __future__ import annotations


class RegressorHand2Body:
    """Past-hand window -> next body frame(s). Deterministic baseline."""

    def __init__(self, motion_dim: int = 141, hand_dim: int = 12, k_hand: int = 20,
                 p_predict: int = 8, hidden: int = 512, n_layers: int = 6, n_heads: int = 8):
        self.cfg = dict(motion_dim=motion_dim, hand_dim=hand_dim, k_hand=k_hand,
                        p_predict=p_predict, hidden=hidden, n_layers=n_layers, n_heads=n_heads)

    def build(self):
        """Construct the torch module (TransformerEncoder over the hand window). TODO(M2)."""
        raise NotImplementedError("implement TransformerEncoder baseline once torch lands (M2)")
