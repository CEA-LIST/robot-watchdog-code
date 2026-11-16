"""AP-Frechet: arc-length prefix Frechet-type distance to the expert trajectories (baseline).

Trajectories are resampled by arc length to a fixed number of points, which aligns them point to
point; the coupling-based discrete Frechet distance is then approximated on GPU by the maximum
pointwise distance between the aligned curves. The score is the minimum over the experts of the
same label, standardised with the mean / std of this distance between expert pairs.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy.interpolate import interp1d

from robot_watchdog.dataset.io import META_KEYS


class FrechetTrajectoryAnomalyScorer:
    def __init__(self, expert_episodes_list, resample_N=198, use_cuda=True, normalize=True, eps=1e-8):
        self.resample_N = int(resample_N)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.normalize = normalize
        self.eps = eps

        # label -> list of expert trajectories (T_i, 2)
        self.experts_by_label = defaultdict(list)
        for ep in expert_episodes_list:
            for k, v in ep.items():
                if k in META_KEYS:
                    continue
                label, pts = v.get("label"), v.get("points")
                if label is None or pts is None:
                    continue
                self.experts_by_label[label.split("_")[0]].append(np.array(pts, dtype=np.float64))

        self._resampled_experts_cpu = {}  # label -> list of arrays (resample_N, 2)
        self._resampled_experts_gpu = {}  # label -> tensor (n_experts, resample_N, 2)
        self._baseline_cache = {}  # label -> (mu, sigma)
        self._prepare_experts_resampled()

    def _resample_by_arclength_np(self, curve, target_len, eps=1e-8):
        if curve is None or len(curve) == 0:
            return np.zeros((target_len, 2), dtype=np.float64)
        curve = np.asarray(curve, dtype=np.float64)
        if curve.ndim == 1:
            if curve.shape[0] != 2:
                raise ValueError(f"Invalid curve shape {curve.shape}: expected (N, 2) or (2,)")
            curve = curve[None, :]
        if curve.shape[0] == 1:
            return np.tile(curve[0], (target_len, 1))

        seg_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        clean = curve[np.concatenate(([True], seg_lengths > eps))]
        if clean.shape[0] < 2:
            return np.tile(clean[0], (target_len, 1))

        seg_lengths = np.linalg.norm(np.diff(clean, axis=0), axis=1)
        s = np.concatenate(([0], np.cumsum(seg_lengths)))
        total = s[-1]
        if total < eps:
            return np.tile(clean[0], (target_len, 1))
        s = s / total

        f = interp1d(s, clean, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
        return f(np.linspace(0.0, 1.0, target_len))

    def _prepare_experts_resampled(self):
        device = "cuda" if self.use_cuda else "cpu"
        for label, trajs in self.experts_by_label.items():
            cpu_res = [self._resample_by_arclength_np(t, self.resample_N) for t in trajs]
            self._resampled_experts_cpu[label] = cpu_res
            self._resampled_experts_gpu[label] = torch.tensor(
                np.stack(cpu_res, axis=0), dtype=torch.float32, device=device
            )

    def _prefix_fast_max(self, test_res, experts):
        """test_res: (N, 2), experts: (M, N, 2) -> min over experts of the max pointwise distance."""
        test = test_res.to(device=experts.device, dtype=torch.float32)
        dists = torch.norm(experts - test.unsqueeze(0), dim=2)  # (M, N)
        cummax = torch.cummax(dists, dim=1).values  # prefix maximum
        raw_min = torch.min(cummax, dim=0).values  # (N,)
        return raw_min[-1].detach().cpu().numpy().astype(np.float64)

    def _compute_baseline_stats(self, label):
        if label in self._baseline_cache:
            return self._baseline_cache[label]
        res_experts = self._resampled_experts_cpu.get(label, [])
        if len(res_experts) <= 1:
            self._baseline_cache[label] = (0.0, 1.0)
            return 0.0, 1.0
        dists = np.array(
            [float(np.max(np.linalg.norm(a - b, axis=1))) for a, b in combinations(res_experts, 2)],
            dtype=np.float64,
        )
        mu = float(np.mean(dists))
        sigma = max(float(np.std(dists)) if dists.size > 1 else self.eps, self.eps)
        self._baseline_cache[label] = (mu, sigma)
        return mu, sigma

    def score_trajectory_against_label(self, test_points, label):
        """Returns the (normalised) score and the raw distance."""
        if isinstance(test_points, torch.Tensor):
            test_points = test_points.detach().cpu().numpy().astype(np.float64)
        test_res = self._resample_by_arclength_np(np.array(test_points, dtype=np.float64), self.resample_N)

        if label not in self._resampled_experts_gpu:
            raw = np.full(self.resample_N, 1e6, dtype=np.float64)
            return raw / (1.0 + self.eps), raw

        experts = self._resampled_experts_gpu[label]
        raw_min = self._prefix_fast_max(torch.tensor(test_res, dtype=torch.float32, device=experts.device), experts)

        mu, sigma = self._compute_baseline_stats(label)
        scores = (raw_min - mu) / (sigma + self.eps) if self.normalize else raw_min
        return scores, raw_min

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
