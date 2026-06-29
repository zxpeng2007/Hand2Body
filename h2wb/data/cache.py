"""Fast on-disk cache of extracted (hand12, body) training pairs + labels + rest skeleton.

Building the pairs from train.pkl costs a 2.5 GB joblib load + FK over ~4M frames. Cache the
result once as flat numpy arrays so subsequent runs load in seconds (np.load, optionally mmap)
and support post-hoc label filtering without re-extraction.

Layout (np.savez): hand (N,12) f32, body (N,135) f32, offsets (n_seq+1,) int64 (sequence
boundaries), labels (object: per-seq comma-joined act_cat set), names (object), rest (22,3) f32.
"""

from __future__ import annotations

import numpy as np


def save_pairs_cache(path, clips, labels, rest, names=None):
    """clips: list of (hand (T,12), body (T,135)); labels: list[str]; rest: (22,3) or None."""
    hand = np.concatenate([c[0] for c in clips]).astype(np.float32)
    body = np.concatenate([c[1] for c in clips]).astype(np.float32)
    offsets = np.zeros(len(clips) + 1, np.int64)
    offsets[1:] = np.cumsum([len(c[0]) for c in clips])
    np.savez(
        path,
        hand=hand, body=body, offsets=offsets,
        labels=np.array(labels, dtype=object),
        names=np.array(names if names is not None else [""] * len(clips), dtype=object),
        rest=(np.zeros((0, 3), np.float32) if rest is None else np.asarray(rest, np.float32)),
    )
    return path


def load_pairs_cache(path, keep_labels=None, drop_labels=None, mmap=True):
    """Return (clips, rest). Optionally keep/drop sequences whose act_cat set matches."""
    d = np.load(path, allow_pickle=True, mmap_mode=("r" if mmap else None))
    hand, body, offsets = d["hand"], d["body"], d["offsets"]
    labels = d["labels"]
    rest = d["rest"]
    rest = None if getattr(rest, "shape", (0,))[0] == 0 else np.asarray(rest)
    keep = set(keep_labels) if keep_labels else None
    drop = set(drop_labels) if drop_labels else None
    clips = []
    for i in range(len(offsets) - 1):
        lab = set(str(labels[i]).split(",")) if labels[i] else set()
        if keep is not None and not (lab & keep):
            continue
        if drop is not None and (lab & drop):
            continue
        s, e = int(offsets[i]), int(offsets[i + 1])
        clips.append((np.asarray(hand[s:e]), np.asarray(body[s:e])))
    return clips, rest
