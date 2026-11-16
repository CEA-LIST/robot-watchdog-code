"""TPC: Trajectory Projection and Correlation (non-parametric failure score).

For each tracked object, the score is the temporally weighted mean (exponentially increasing
weights, so that recent deviations dominate) of the point-to-polyline distance between the observed
trajectory and the closest expert trajectory of the same label. Optionally, a correlation penalty
compares the pairwise Pearson correlations between object motions with the expert ones.
"""

import math
from collections import defaultdict

import numpy as np
import torch

from robot_watchdog.dataset.io import META_KEYS


class TPCAnomalyScorer:
    def __init__(
        self,
        expert_episodes_list,
        weight_mode="exp",
        decay=0.2,
        score_type="proj_only",
        normalize=False,
        incremental=True,
        device="cuda",
        eps=1e-6,
    ):
        """
        Args:
            expert_episodes_list: reference expert episodes (dicts {"1": {"label", "points"}, ...})
            weight_mode: temporal weights, "exp", "linear" or "uniform"
            decay: rate of the exponential weights
            score_type: "proj_only", "corr_only" or "both"
            normalize: standardise the distances with the expert pairwise statistics of each label
            incremental: streaming computation of the projection score (only the new points of a
                growing prefix are processed), same values as the full computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.weight_mode = weight_mode
        self.decay = decay
        self.score_type = score_type
        self.normalize = normalize
        self.incremental = incremental

        self.experts_by_label = defaultdict(list)
        for ep in expert_episodes_list:
            for k, v in ep.items():
                if k in META_KEYS:
                    continue
                label = v["label"].split("_")[0]
                self.experts_by_label[label].append(self._to_torch(torch.tensor(v["points"])))

        self.baseline_stats = {}
        if normalize:
            self._compute_baseline_stats()

        # caches of the streaming computation
        self._clean_experts_cache = {}
        self._stream_cache = {}

    # ------------------------- utils -------------------------

    def _to_torch(self, pts):
        if isinstance(pts, np.ndarray):
            return torch.tensor(pts, dtype=torch.float32, device=self.device)
        if isinstance(pts, torch.Tensor):
            return pts.to(self.device, dtype=torch.float32)
        raise ValueError("Unsupported point format")

    def _clean_static_segments(self, traj):
        """Remove consecutive duplicate points (static segments).

        Returns the cleaned trajectory (K, 2) and the boolean mask (T,) of the kept points.
        """
        T = traj.shape[0]
        if T < 2:
            return traj, torch.ones(T, dtype=torch.bool, device=traj.device)

        diffs = torch.norm(traj[1:] - traj[:-1], dim=1)
        keep = torch.cat([torch.tensor([True], device=traj.device), diffs > self.eps])
        clean = traj[keep]
        if clean.shape[0] < 2:
            # static trajectory: duplicate its first point to keep a valid polyline
            return traj[:1].repeat(2, 1), keep
        return clean, keep

    def _make_weights(self, T):
        if self.weight_mode == "linear":
            w = torch.linspace(0.2, 1.0, T, device=self.device)
        elif self.weight_mode == "exp":
            # largest weight on the most recent point
            w = torch.exp(torch.linspace(0, self.decay * (T - 1), T, device=self.device))
        elif self.weight_mode == "uniform":
            w = torch.ones(T, device=self.device)
        else:
            raise ValueError(f"Invalid weight_mode: {self.weight_mode}")
        return w / torch.sum(w)

    # ------------------------- distances -------------------------

    def _point_to_polyline_distance(self, P, V):
        """Vectorised point-to-polyline distance.

        P: (N, 2) points, V: (M, 2) polyline. Returns the distances (N,) and the projected points (N, 2).
        """
        P = P.to(self.device)
        V = V.to(self.device)

        if V.shape[0] < 2:
            d = torch.norm(P - V[0], dim=1)
            return d, V[0].unsqueeze(0).repeat(P.shape[0], 1)

        A = V[:-1]  # (M-1, 2) segment starts
        S = V[1:] - A  # (M-1, 2) segment vectors
        L2 = torch.clamp(torch.sum(S**2, dim=1), min=self.eps)

        P_exp = P[:, None, :]  # (N, 1, 2)
        A_exp = A[None, :, :]  # (1, M-1, 2)
        S_exp = S[None, :, :]

        u = torch.sum(torch.abs((P_exp - A_exp) * S_exp), dim=2) / (L2[None, :] + self.eps)
        u = torch.clamp(u, 0.0, 1.0)

        proj = A_exp + u[..., None] * S_exp  # (N, M-1, 2)
        dist2 = torch.sum((P_exp - proj) ** 2, dim=2)

        min_idx = torch.argmin(dist2, dim=1)
        min_d2 = dist2[torch.arange(P.shape[0]), min_idx]
        proj_pts = proj[torch.arange(P.shape[0]), min_idx]

        return torch.sqrt(min_d2 + self.eps), proj_pts

    def _compute_baseline_stats(self):
        """Mean / std of the projection distances between expert pairs, per label."""
        for label, trajs in self.experts_by_label.items():
            dists = []
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    d, _ = self._point_to_polyline_distance(trajs[i], trajs[j])
                    dists.append(torch.mean(d).item())
            if len(dists) > 0:
                self.baseline_stats[label] = (np.mean(dists), np.std(dists))
            else:
                self.baseline_stats[label] = (0.0, 1.0)

    # ------------------------- correlation -------------------------

    def _resample(self, traj, T):
        """Resample a trajectory to length T by linear interpolation."""
        if traj.shape[0] == T:
            return traj
        idx = torch.linspace(0, traj.shape[0] - 1, T, device=self.device)
        low = torch.floor(idx).long()
        high = torch.ceil(idx).long()
        alpha = idx - low
        return traj[low] * (1 - alpha).unsqueeze(1) + traj[high] * alpha.unsqueeze(1)

    def correlation_score(self, inf_episode_dict):
        """l1 distance between the pairwise correlations of the observed objects and of the experts."""
        labels = []
        inf_trajs = []
        for k, v in inf_episode_dict.items():
            if k in META_KEYS:
                continue
            labels.append(v["label"].split("_")[0])
            inf_trajs.append(self._to_torch(torch.tensor(v["points"])))

        N = len(inf_trajs)
        if N < 2:
            return torch.tensor(0.0, device=self.device)

        T = max(t.shape[0] for t in inf_trajs)
        inf_trajs = [self._resample(t, T) for t in inf_trajs]

        inf_corr = []
        for i in range(N):
            for j in range(i + 1, N):
                corr_x = torch.corrcoef(torch.stack([inf_trajs[i][:, 0], inf_trajs[j][:, 0]]))[0, 1]
                corr_y = torch.corrcoef(torch.stack([inf_trajs[i][:, 1], inf_trajs[j][:, 1]]))[0, 1]
                inf_corr.append((corr_x + corr_y) / 2)

        # reference correlations: first expert demonstration of each label
        exp_corr = []
        for i in range(N):
            for j in range(i + 1, N):
                if labels[i] in self.experts_by_label and labels[j] in self.experts_by_label:
                    e_i = self.experts_by_label[labels[i]][0]
                    e_j = self.experts_by_label[labels[j]][0]
                    T = max(e_i.shape[0], e_j.shape[0])
                    e_i = self._resample(e_i, T)
                    e_j = self._resample(e_j, T)
                    corr_x = torch.corrcoef(torch.stack([e_i[:, 0], e_j[:, 0]]))[0, 1]
                    corr_y = torch.corrcoef(torch.stack([e_i[:, 1], e_j[:, 1]]))[0, 1]
                    exp_corr.append((corr_x + corr_y) / 2)

        if len(exp_corr) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.sum(torch.abs(torch.stack(inf_corr) - torch.stack(exp_corr)))

    # ------------------------- projection score -------------------------

    def _clean_experts(self, label):
        if label not in self._clean_experts_cache:
            self._clean_experts_cache[label] = [
                self._clean_static_segments(e)[0] for e in self.experts_by_label[label]
            ]
        return self._clean_experts_cache[label]

    def _best_distances(self, inf_clean, label):
        """Distance of every point of inf_clean to the closest expert trajectory of the label."""
        experts = self._clean_experts(label)
        per_expert = torch.stack([self._point_to_polyline_distance(inf_clean, e)[0] for e in experts], dim=0)
        best = torch.min(per_expert, dim=0).values
        if self.normalize:
            mu, sigma = self.baseline_stats.get(label, (0.0, 1.0))
            best = abs(best - mu) / (sigma + self.eps)
        return best

    def score_trajectory_against_label(self, inf_points, label):
        """Weighted projection score of a full trajectory (reference computation)."""
        inf_traj = self._to_torch(inf_points)
        inf_clean, _ = self._clean_static_segments(inf_traj)

        if len(self.experts_by_label[label]) == 0:
            return torch.full((inf_traj.shape[0],), 1e6, device=self.device)

        per_time_best = self._best_distances(inf_clean, label)  # (T_clean,)
        weights = self._make_weights(inf_clean.shape[0])
        return torch.sum(per_time_best * weights)

    def score_trajectory_incremental(self, inf_points, label, key):
        """Same value as score_trajectory_against_label, computing only the points not seen yet.

        The streaming evaluation feeds growing prefixes of the same episode: the per-point distances
        of the prefix are already known, only the last point(s) are new, and the normalised
        exponential weights are folded into a running (numerator, denominator) pair.
        """
        inf_traj = self._to_torch(torch.as_tensor(inf_points))
        T = inf_traj.shape[0]
        # keyed by object, not by label: several objects of an episode can share a label
        state = self._stream_cache.get(key)
        if state is None or T < state["T_seen"]:  # new episode
            state = {"T_seen": 0, "n_kept": 0, "num": 0.0, "den": 0.0, "last_raw": None}
            self._stream_cache[key] = state
        if T == state["T_seen"]:
            return torch.tensor(state["num"] / max(state["den"], self.eps), device=self.device)

        if len(self.experts_by_label[label]) == 0:
            return torch.full((1,), 1e6, device=self.device)[0]

        new = inf_traj[state["T_seen"] : T]
        keep = []
        prev = state["last_raw"]
        for i in range(new.shape[0]):
            pt = new[i]
            if prev is None or torch.norm(pt - prev) > self.eps:
                keep.append(pt)
            prev = pt
        state["last_raw"] = prev
        state["T_seen"] = T

        if keep:
            best = self._best_distances(torch.stack(keep), label)
            for k, b in enumerate(best.tolist()):
                w = math.exp(self.decay * (state["n_kept"] + k))
                state["num"] += w * b
                state["den"] += w
            state["n_kept"] += len(keep)
        if state["den"] == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.tensor(state["num"] / state["den"], device=self.device)

    # ------------------------- scoring -------------------------

    def score_inference(self, inf_episode_dict):
        """inf_episode_dict: {i: {"label": str, "points": (T, 2)}} -> {i: {"label": str, "scores": float}}"""
        out = {}
        for k, v in inf_episode_dict.items():
            if k in META_KEYS:
                out[k] = v
                continue
            label = v["label"].split("_")[0]
            if self.incremental:
                score = self.score_trajectory_incremental(v["points"], label, k)
            else:
                score = self.score_trajectory_against_label(v["points"], label)
            out[k] = {"label": label, "scores": float(score.item())}

        object_keys = [k for k in out if k not in META_KEYS]
        for k in object_keys:
            if math.isnan(out[k]["scores"]):
                out[k]["scores"] = 0.0

        if self.score_type == "proj_only":
            return out

        corr_penalty = torch.nan_to_num(self.correlation_score(inf_episode_dict), nan=0.0).item()
        if self.score_type == "corr_only":
            for k in object_keys:
                out[k]["scores"] = corr_penalty
        elif self.score_type == "both":
            # the correlation penalty is added to the projection score of every object
            for k in object_keys:
                out[k]["scores"] += corr_penalty
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}")
        return out
