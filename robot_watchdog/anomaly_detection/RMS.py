"""AP-RMS: arc-length prefix RMS distance to the expert trajectories (trajectory matching baseline)."""

from collections import defaultdict

import numpy as np
import torch
from scipy.interpolate import interp1d

from robot_watchdog.dataset.io import META_KEYS


class RMSTrajectoryAnomalyScorer:
    """The observed trajectory and the expert trajectories of the same label are resampled by arc
    length to the length T of the observation; the score is the temporal mean over t of the minimum
    (over experts) RMS distance between the prefixes of length t.
    """

    def __init__(self, expert_episodes_list, eps=1e-8):
        self.eps = eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # label -> list of expert trajectories (T_i, 2)
        self.experts_by_label = defaultdict(list)
        for ep in expert_episodes_list:
            for k, v in ep.items():
                if k in META_KEYS:
                    continue
                pts = np.array(v["points"], dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 2:
                    raise ValueError("Trajectories must have shape (T, 2)")
                self.experts_by_label[v["label"].split("_")[0]].append(pts)

        # (label, target_len) -> tensor (n_experts, target_len, 2)
        self._resampled_cache = {}

    def _to_numpy(self, pts):
        if isinstance(pts, torch.Tensor):
            return pts.detach().cpu().numpy().astype(np.float64)
        return np.array(pts, dtype=np.float64)

    def _to_tensor(self, pts):
        if isinstance(pts, torch.Tensor):
            return pts.detach().to(self.device).float()
        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    def _resample_by_arclength(self, curve, target_len):
        """Resample a 2D curve to target_len points equally spaced in arc length.

        Static segments are removed first; degenerate (single point) curves are repeated.
        """
        eps = self.eps
        curve = self._to_numpy(curve)
        if curve is None or len(curve) == 0:
            return self._to_tensor(np.zeros((target_len, 2), dtype=np.float64))
        if curve.ndim == 1:
            if curve.shape[0] != 2:
                raise ValueError(f"Invalid curve shape {curve.shape}: expected (N, 2) or (2,)")
            curve = curve[None, :]
        if curve.shape[0] == 1:
            return self._to_tensor(np.tile(curve[0], (target_len, 1)))

        seg_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        clean_curve = curve[np.concatenate(([True], seg_lengths > eps))]
        if clean_curve.shape[0] < 2:
            return self._to_tensor(np.tile(clean_curve[0], (target_len, 1)))

        seg_lengths = np.linalg.norm(np.diff(clean_curve, axis=0), axis=1)
        s = np.concatenate(([0], np.cumsum(seg_lengths)))
        total_len = s[-1]
        if total_len < eps:
            return self._to_tensor(np.tile(clean_curve[0], (target_len, 1)))
        s /= total_len

        f = interp1d(s, clean_curve, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
        return self._to_tensor(f(np.linspace(0.0, 1.0, target_len)))

    def _resampled_experts(self, label, target_len):
        key = (label, target_len)
        if key not in self._resampled_cache:
            res = [self._resample_by_arclength(e, target_len) for e in self.experts_by_label.get(label, [])]
            self._resampled_cache[key] = torch.stack(res)
        return self._resampled_cache[key]

    def score_trajectory_against_label(self, test_points, label):
        """Returns the score and the minimum prefix RMS distance at every step (T,)."""
        test = self._to_tensor(test_points)
        T = test.shape[0]
        if T < 1:
            empty = torch.empty(0, device=self.device)
            return empty, empty

        expert_tensor = self._resampled_experts(label, T)  # (N, T, 2)
        if len(expert_tensor) == 0:
            raw = torch.full((T,), 1e6, device=self.device)
            return raw / (1.0 + self.eps), raw

        test_res = self._resample_by_arclength(test, T)

        raw_min = []
        for t in range(1, T + 1):
            diffs = test_res[:t].unsqueeze(0) - expert_tensor[:, :t]  # (N, t, 2)
            dists = torch.sqrt(torch.mean(torch.sum(diffs**2, dim=2), dim=1))  # (N,)
            raw_min.append(torch.min(dists))
        raw_min = torch.stack(raw_min)  # (T,)

        return torch.sum(raw_min) / len(raw_min), raw_min

    def score_inference(self, inf_episode_dict):
        """inf_episode_dict: {i: {"label": str, "points": (T, 2)}} -> {i: {"label", "scores", "raw_distances"}}"""
        out = {}
        for k, v in inf_episode_dict.items():
            if k in META_KEYS:
                out[k] = v
                continue
            label, points = v.get("label"), v.get("points")
            if label is None or points is None:
                raise ValueError("Inference entries must contain 'label' and 'points'")
            scores, raw = self.score_trajectory_against_label(points, label)
            out[k] = {"label": label, "scores": scores, "raw_distances": raw}
        return out
