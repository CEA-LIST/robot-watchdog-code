"""Soft-DTW: minimum soft-DTW cost to the expert trajectories (trajectory matching baseline)."""

from collections import defaultdict

import torch

from robot_watchdog.anomaly_detection.soft_dtw_cuda import SoftDTW
from robot_watchdog.dataset.io import META_KEYS


class DTWTrajectoryAnomalyScorer:
    """Trajectories longer than `max_len` steps are uniformly downsampled to `max_len` and
    zero-padded to a common length; the score is the minimum soft-DTW cost over the experts of the
    same label, standardised with the mean / std of the cost between expert pairs.
    """

    def __init__(self, expert_episodes_list, gamma=0.1, normalize=True, eps=1e-8, max_len=300):
        self.normalize = normalize
        self.eps = eps
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # the CUDA kernel of SoftDTW is limited to sequences of at most 1024 steps (and needs libNVVM):
        # the CPU (numba) implementation is used
        self.soft_dtw = SoftDTW(use_cuda=False, gamma=gamma)

        self.experts_by_label = defaultdict(list)
        for ep in expert_episodes_list:
            for k, v in ep.items():
                if k in META_KEYS:
                    continue
                points = self._to_tensor(v["points"])
                if points.ndim != 2 or points.shape[1] != 2:
                    raise ValueError("Trajectories must have shape (T, 2)")
                self.experts_by_label[v["label"].split("_")[0]].append(points)

        self._baseline_cache = {}

    def _to_tensor(self, pts):
        if isinstance(pts, torch.Tensor):
            return pts.detach().to(self.device).float()
        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    def _downsample(self, traj):
        T = traj.shape[0]
        if T <= self.max_len:
            return traj
        idx = torch.linspace(0, T - 1, self.max_len).long()
        return traj[idx]

    def _compute_baseline_stats(self, label):
        if label in self._baseline_cache:
            return self._baseline_cache[label]

        experts = [self._downsample(e) for e in self.experts_by_label.get(label, [])]
        n = len(experts)
        if n <= 1:
            self._baseline_cache[label] = (0.0, 1.0)
            return 0.0, 1.0

        max_len = max(e.shape[0] for e in experts)
        padded = torch.zeros((n, max_len, 2), device=self.device)
        for i, e in enumerate(experts):
            padded[i, : e.shape[0]] = e

        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(self.soft_dtw(padded[i : i + 1], padded[j : j + 1]).squeeze())
        dists = torch.stack(dists)
        mu = dists.mean().item()
        sigma = max(dists.std().item() if dists.numel() > 1 else 0.0, self.eps)
        self._baseline_cache[label] = (mu, sigma)
        return mu, sigma

    def score_trajectory_against_label(self, test_points, label):
        """Returns the (normalised) score and the soft-DTW cost to every expert."""
        test = self._downsample(self._to_tensor(test_points))
        if test.shape[0] < 1:
            empty = torch.empty(0, device=self.device)
            return empty.mean(), empty

        experts = [self._downsample(e) for e in self.experts_by_label.get(label, [])]
        if len(experts) == 0:
            raw = torch.full((1,), 1e6, device=self.device)
            return raw[0] / (1.0 + self.eps), raw

        mu, sigma = self._compute_baseline_stats(label)

        max_len = max(e.shape[0] for e in experts)
        test_padded = torch.zeros((1, max_len, 2), device=self.device)
        test_padded[0, : test.shape[0]] = test
        expert_padded = torch.zeros((len(experts), max_len, 2), device=self.device)
        for i, e in enumerate(experts):
            expert_padded[i, : e.shape[0]] = e

        raw_dists = self.soft_dtw(test_padded.expand(len(experts), -1, -1), expert_padded)
        min_dist = torch.min(raw_dists)
        score = torch.abs(min_dist - mu) / (sigma + self.eps) if self.normalize else min_dist
        return score, raw_dists

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
            score, raw = self.score_trajectory_against_label(points, label)
            out[k] = {"label": label, "scores": score.item(), "raw_distances": raw.tolist()}
        return out
