"""Loading of the trajectories and of the per-frame ground-truth labels.

An episode is stored as a dict:
    {"1": {"label": str, "points": [[x, y], ...]}, "2": {...}, ...,
     "fine_labels": [...], "coarse_labels": [...], "episode_idx": int}
with one entry per tracked object (2D centroid of its SAM2 mask in image coordinates).
"""

import glob
import json
import os

import numpy as np

META_KEYS = ("fine_labels", "coarse_labels", "episode_idx")


def load_trajectories(data_dir, task, split):
    """split: "expert" (nominal demonstrations) or "anomaly" (test episodes)."""
    with open(os.path.join(data_dir, task, f"{split}.json"), "r") as f:
        return json.load(f)


def load_labels(data_dir, task):
    """Per-frame ground truth of the test episodes (1 = failure), in the order of anomaly.json."""
    files = sorted(glob.glob(os.path.join(data_dir, task, "labels", "*_labels.csv")))
    return [np.loadtxt(f, delimiter=",", dtype=int).reshape(-1) for f in files]


def _pad_repeat_axis0(a, target_len):
    """Pad an array (T, D) by repeating its last row until target_len."""
    T = a.shape[0]
    if T >= target_len:
        return a[:target_len]
    last = a[-1:]
    pad = np.repeat(last, target_len - T, axis=0)
    return np.concatenate([a, pad], axis=0)


def episodes_from_data_dict(data_dict, return_labels=False):
    """Convert a list of episode dicts into a list of arrays (T, C, 2).

    Objects are ordered as in episode["fine_labels"]; shorter object trajectories are padded by
    repeating their last point and a missing object is replaced by a NaN trajectory.
    Returns the episodes and, if return_labels, the list of object labels of each episode.
    """
    episodes = []
    labels_all = []

    for ep in data_dict:
        if "fine_labels" in ep:
            keys = [str(i) for i in range(1, len(ep["fine_labels"]) + 1)]
        else:
            keys = sorted((k for k in ep.keys() if k not in META_KEYS), key=lambda x: int(x))

        node_trajs = []
        node_labels = []
        max_T = 0
        for k in keys:
            if k not in ep or ep[k].get("points") is None:
                arr = np.full((1, 2), np.nan, dtype=float)
                label = ep[k].get("label") if k in ep else None
            else:
                arr = np.asarray(ep[k]["points"], dtype=np.float64)
                if arr.ndim == 1 and arr.size == 2:
                    arr = arr.reshape(1, 2)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(f"Trajectory of object {k} must have shape (T, 2), got {arr.shape}")
                label = ep[k].get("label")
            node_trajs.append(arr)
            node_labels.append(label)
            max_T = max(max_T, arr.shape[0])

        padded = [_pad_repeat_axis0(arr, max_T) for arr in node_trajs]
        episodes.append(np.stack(padded, axis=1))  # (T, C, 2)
        labels_all.append(node_labels)

    if return_labels:
        return episodes, labels_all
    return episodes
