"""Streaming evaluation of the non-parametric scorers: an episode is replayed as growing prefixes."""

import numpy as np
import torch
from torch.utils.data import Dataset

from robot_watchdog.dataset.io import META_KEYS


class PrefixDataset(Dataset):
    """Item t of an episode is the prefix of length t + 1 of every object trajectory.

    The dataset length is the length of the longest object trajectory; shorter trajectories are
    padded by repeating their last point. Item t is a dict {i: {"label": str, "points": (t + 1, 2)}}.
    """

    def __init__(self, episode_dict, dtype=torch.float32):
        super().__init__()
        keys = sorted((k for k in episode_dict.keys() if k not in META_KEYS), key=lambda x: int(x))

        self.trajs = []
        self.labels = []
        for k in keys:
            v = episode_dict[k]
            pts = torch.tensor(v["points"], dtype=dtype)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(f"Trajectory of object {k} must have shape (T, 2), got {pts.shape}")
            self.trajs.append(pts)
            self.labels.append(v["label"])

        self._len = max(t.shape[0] for t in self.trajs) if self.trajs else 0

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        t = idx + 1
        prefixes = {}
        for i, (traj, label) in enumerate(zip(self.trajs, self.labels)):
            if traj.shape[0] >= t:
                prefix = traj[:t].clone()
            else:
                pad = traj[-1].unsqueeze(0).repeat(t - traj.shape[0], 1)
                prefix = torch.cat([traj, pad], dim=0)
            prefixes[i] = {"label": label, "points": prefix}
        return prefixes


def identity_collate(batch):
    """batch_size = 1: return the sample unchanged."""
    return batch[0]


def extract_scalar(x):
    """Convert a scalar tensor / array to a Python float (non-scalars are returned as arrays)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.numpy()
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x
    return x


def _scores_to_numpy(scores):
    """Scores may be a (CUDA) tensor, a scalar or a list mixing tensors and floats."""
    if torch.is_tensor(scores):
        return scores.detach().cpu().numpy()
    if isinstance(scores, (list, tuple)):
        return np.array([float(x.detach().cpu()) if torch.is_tensor(x) else float(x) for x in scores])
    return np.asarray(scores)


def sum_over_objects(object_scores):
    """{i: {"scores": s_i}, ...} -> sum of the object scores (NaN ignored), as a float32 tensor."""
    score_lists = [_scores_to_numpy(v["scores"]) for v in object_scores.values()]
    return torch.tensor(np.nansum(np.array(score_lists), axis=0), dtype=torch.float32)


def episodes_sum_over_objects(episodes_object_scores):
    """List of {i: {"scores": [s_i(t)]_t}} -> tensor [N, T_max] of the per-step sum over objects.

    Episodes shorter than T_max are padded with NaN.
    """
    all_series = []
    max_T = 0
    for d in episodes_object_scores:
        score_lists = [np.array(v["scores"]) for v in d.values()]
        max_len = max(len(s) for s in score_lists)
        max_T = max(max_T, max_len)
        padded = [np.pad(s, (0, max_len - len(s)), constant_values=np.nan) for s in score_lists]
        all_series.append(np.nansum(np.stack(padded), axis=0))

    all_series = [np.pad(s, (0, max_T - len(s)), constant_values=np.nan) for s in all_series]
    return torch.tensor(np.stack(all_series), dtype=torch.float32)


def list_to_tensor(list_of_lists):
    """List of N score sequences (floats) -> tensor [N, T_max], padded with NaN."""
    T_max = max(len(seq) for seq in list_of_lists)
    result = torch.full((len(list_of_lists), T_max), torch.nan)
    for n, seq in enumerate(list_of_lists):
        for t, value in enumerate(seq):
            result[n, t] = float(value)
    return result
