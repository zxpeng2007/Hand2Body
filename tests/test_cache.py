"""Tests for the pairs cache (save/load round-trip + label filtering)."""

import numpy as np

from h2wb.data.cache import save_pairs_cache, load_pairs_cache


def _clips(seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((T, 12)).astype(np.float32),
             rng.standard_normal((T, 135)).astype(np.float32)) for T in (10, 15, 8)]


def test_cache_roundtrip_preserves_clips_and_rest(tmp_path):
    clips = _clips()
    rest = np.arange(66, dtype=np.float32).reshape(22, 3)
    path = save_pairs_cache(str(tmp_path / "c.npz"), clips, ["a", "b,c", "a"], rest)
    out, rest2 = load_pairs_cache(path)
    assert len(out) == 3
    for (h, b), (h0, b0) in zip(out, clips):
        assert h.shape == h0.shape and np.allclose(h, h0)
        assert b.shape == b0.shape and np.allclose(b, b0)
    assert np.allclose(rest2, rest)


def test_label_keep_filter(tmp_path):
    clips = _clips()
    path = save_pairs_cache(str(tmp_path / "c.npz"), clips, ["walk", "strike,backhand", "walk"], None)
    kept, _ = load_pairs_cache(path, keep_labels=["strike"])
    assert len(kept) == 1                      # only the 'strike,backhand' sequence
    assert kept[0][0].shape[0] == 15


def test_label_drop_filter(tmp_path):
    clips = _clips()
    path = save_pairs_cache(str(tmp_path / "c.npz"), clips, ["walk", "strike", "walk"], None)
    kept, _ = load_pairs_cache(path, drop_labels=["walk"])
    assert len(kept) == 1
    assert kept[0][0].shape[0] == 15


def test_no_rest_roundtrips_to_none(tmp_path):
    path = save_pairs_cache(str(tmp_path / "c.npz"), _clips(), ["a", "a", "a"], None)
    _, rest = load_pairs_cache(path)
    assert rest is None
