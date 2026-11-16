"""Time-varying conformal band (functional split conformal prediction, Diquigiovanni et al., 2021).

The calibration scores (nominal episodes) are split into D_A, which gives the mean profile mu_t and
the modulation s(t), and D_B, which gives the band width h: the (1 - alpha)-quantile of
max_t |mu_t - S_t| / s(t). The threshold is eta_t = mu_t + h * s(t). Two modulations are computed:
    variant1: s1(t) = 1 / T                       (uniform band)
    variant2: s2(t) = max_{i in D_A} |S_t^i - mu_t| (wider where the experts disagree)
One threshold is built for every alpha of cfg.alpha_all and both variants.
"""

import random

import numpy as np
import torch


class ConformalPredictionTimeAll:
    def __init__(self, cfg, scores: torch.Tensor):
        """
        cfg: threshold config (alpha_all, split_rate)
        scores: calibration scores [nb_episodes, T] (NaN = after the end of an episode)
        """
        self.alpha_all = list(cfg.alpha_all)
        self.split_rate = cfg.split_rate

        scores = torch.nan_to_num(scores, nan=0.0)
        self.nb_episodes, self.T = scores.shape

        self.split_datasets(scores)
        self.mu_t = self.DcalA.mean(dim=0)

        self.scalA_t_var1 = torch.ones(self.T) / self.T
        self.scalA_t_var2 = torch.abs(self.DcalA - self.mu_t).max(dim=0).values

        self.thresholds = []
        self.param_list = []
        for variant, scal in (("variant1", self.scalA_t_var1), ("variant2", self.scalA_t_var2)):
            for alpha, h in zip(self.alpha_all, self.compute_quantile_h(scal)):
                self.thresholds.append(self.mu_t + h * scal)
                self.param_list.append({"alpha": alpha, "split_rate": self.split_rate, "scal_variant": variant})

    def split_datasets(self, scores):
        all_indices = list(range(self.nb_episodes))
        random.shuffle(all_indices)
        N1 = int(self.split_rate * self.nb_episodes)
        self.DcalA = scores[all_indices[:N1], :]  # [N1, T]
        self.DcalB = scores[all_indices[N1:], :]  # [N2, T]

    def compute_quantile_h(self, scal):
        """h for every alpha: (1 - alpha)-quantile of the maximum normalised deviation on D_B."""
        scal[scal == 0] = 1e-6
        deviations = [
            (torch.abs(self.mu_t - self.DcalB[j]) / scal).max().item() for j in range(self.DcalB.shape[0])
        ]
        return [np.quantile(deviations, 1 - alpha) for alpha in self.alpha_all]

    def __call__(self, score, t):
        """Binary decision (score > eta_t) for every (alpha, variant), and the thresholds at t."""
        score = score.item() if isinstance(score, torch.Tensor) else score
        # test episodes can be longer than the calibration episodes: keep the last threshold value
        t = min(t, self.T - 1)
        thresholds_t = [threshold[t].item() for threshold in self.thresholds]
        return [score > thr for thr in thresholds_t], thresholds_t
