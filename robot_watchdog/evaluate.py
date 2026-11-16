"""Online evaluation of a failure detector on the test episodes of a task.

At every time step the detector outputs one score per tracked object; the failure score S_t is
their sum over objects. The conformal band calibrated on held-out expert demonstrations turns S_t
into a binary decision for every (alpha, modulation) configuration.
"""

import json
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from robot_watchdog.anomaly_detection.DTW import DTWTrajectoryAnomalyScorer
from robot_watchdog.anomaly_detection.Frechet import FrechetTrajectoryAnomalyScorer
from robot_watchdog.anomaly_detection.RMS import RMSTrajectoryAnomalyScorer
from robot_watchdog.anomaly_detection.TPC import TPCAnomalyScorer
from robot_watchdog.conformal_prediction import ConformalPredictionTimeAll
from robot_watchdog.dataset.graph import GraphCustomDataset, collate_seq
from robot_watchdog.dataset.io import META_KEYS, episodes_from_data_dict, load_labels, load_trajectories
from robot_watchdog.dataset.streaming import (
    PrefixDataset,
    episodes_sum_over_objects,
    extract_scalar,
    list_to_tensor,
    sum_over_objects,
)
from robot_watchdog.metrics import compute_metrics
from robot_watchdog.prediction_error import prediction_error
from robot_watchdog.train import build_model, calibration_split_path, forward, weights_path


# ======================================================================================
# Non-parametric scorers (TPC, RMS, Frechet, DTW)
# ======================================================================================


def build_scorer(cfg, reference_episodes):
    m_cfg = cfg.method
    if m_cfg.name == "TPC":
        return TPCAnomalyScorer(
            reference_episodes,
            weight_mode=m_cfg.weight_mode,
            decay=m_cfg.decay,
            score_type=m_cfg.score_type,
            normalize=m_cfg.normalize,
            incremental=m_cfg.incremental,
            device=cfg.device,
            eps=m_cfg.eps,
        )
    if m_cfg.name == "RMS":
        return RMSTrajectoryAnomalyScorer(reference_episodes)
    if m_cfg.name == "Frechet":
        return FrechetTrajectoryAnomalyScorer(
            reference_episodes, resample_N=m_cfg.resample_N, normalize=m_cfg.normalize, eps=m_cfg.eps
        )
    if m_cfg.name == "DTW":
        return DTWTrajectoryAnomalyScorer(
            reference_episodes, gamma=m_cfg.gamma, normalize=m_cfg.normalize, eps=m_cfg.eps
        )
    raise NotImplementedError(f"Unknown non-parametric method: {m_cfg.name}")


def calibration_scores_non_parametric(scorer, episodes):
    """Per-object score at every step of the calibration episodes -> tensor [N, T] of the sums."""
    all_scores = []
    for episode_dict in tqdm(episodes, desc="calibration episodes"):
        episode_scores = {}
        for prefix in PrefixDataset(episode_dict):
            for k, v in scorer.score_inference(prefix).items():
                if not isinstance(v, dict):
                    continue
                episode_scores.setdefault(k, {"label": v.get("label"), "scores": []})
                episode_scores[k]["scores"].append(extract_scalar(v.get("scores")))
        all_scores.append(episode_scores)
    return episodes_sum_over_objects(all_scores)


def evaluate_non_parametric(cfg):
    # 1) split the expert demonstrations into the reference set and the calibration set
    expert_dict = load_trajectories(cfg.data_dir, cfg.task, "expert")
    random.shuffle(expert_dict)
    split_idx = int(len(expert_dict) * cfg.method.reference_ratio)
    reference, calibration = expert_dict[:split_idx], expert_dict[split_idx:]
    print(f"{len(reference)} reference / {len(calibration)} calibration expert demonstrations")

    scorer = build_scorer(cfg, reference)

    # 2) conformal calibration
    cp = ConformalPredictionTimeAll(cfg.threshold, calibration_scores_non_parametric(scorer, calibration))

    # 3) streaming evaluation of the test episodes
    all_scores = []
    for episode_dict in tqdm(load_trajectories(cfg.data_dir, cfg.task, "anomaly"), desc="test episodes"):
        episode = {"raw_scores": [], "binary_scores": []}
        for t, prefix in enumerate(PrefixDataset(episode_dict)):
            score = sum_over_objects(scorer.score_inference(prefix))
            binary, _ = cp(score, t)
            episode["raw_scores"].append(float(score))
            episode["binary_scores"].append(binary)
        all_scores.append(episode)

    return save_and_evaluate(cfg, all_scores, cp.param_list)


# ======================================================================================
# Predictive models (GnnT and variants)
# ======================================================================================


class EMASmoother:
    """Exponential moving average of the online failure score."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.current_ema = None

    def update(self, new_score):
        if self.current_ema is None:
            self.current_ema = new_score
        else:
            self.current_ema = self.alpha * new_score + (1 - self.alpha) * self.current_ema
        return self.current_ema


@torch.no_grad()
def window_scores(model, cfg, episode, device, batch_size=128):
    """Prediction error of every sliding window (m observed frames, L predicted) of an episode."""
    dataset = GraphCustomDataset([episode], m=cfg.method.m, L=cfg.method.L)
    scores = []
    for start in range(0, len(dataset), batch_size):
        batch = collate_seq([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        preds = forward(model, cfg, batch, device)
        scores.append(prediction_error(preds, batch["target_pos"].to(device), per_sample=True))
    return torch.cat(scores) if scores else torch.empty(0, device=device)


def evaluate_predictive(cfg):
    device = torch.device(cfg.device)

    with open(calibration_split_path(cfg), "rb") as f:
        calibration_episodes = episodes_from_data_dict(pickle.load(f))
    test_dict = load_trajectories(cfg.data_dir, cfg.task, "anomaly")
    test_episodes = episodes_from_data_dict(test_dict)

    C = len([k for k in test_dict[0] if k not in META_KEYS])
    model = build_model(cfg, C).to(device)
    model.load_state_dict(torch.load(weights_path(cfg), map_location=device)["model_state"])
    model.eval()

    # 1) conformal calibration on the held-out expert demonstrations
    calib_scores = [
        window_scores(model, cfg, ep, device).cpu().tolist()
        for ep in tqdm(calibration_episodes, desc="calibration episodes")
    ]
    cp = ConformalPredictionTimeAll(cfg.threshold, list_to_tensor(calib_scores))

    # 2) online evaluation; the score is smoothed by an exponential moving average whose state is
    #    carried over from one test episode to the next
    smoother = EMASmoother(alpha=0.2)
    all_scores = []
    for ep in tqdm(test_episodes, desc="test episodes"):
        episode = {"raw_scores": [], "binary_scores": []}
        for t, score in enumerate(window_scores(model, cfg, ep, device)):
            score = smoother.update(score)
            binary, _ = cp(score, t)
            episode["raw_scores"].append(score.detach().cpu().item())
            episode["binary_scores"].append(binary)
        all_scores.append(episode)

    return save_and_evaluate(cfg, all_scores, cp.param_list)


# ======================================================================================
# Outputs
# ======================================================================================


def save_and_evaluate(cfg, all_scores, param_list):
    tag = f"{cfg.method.name}_{cfg.task}"
    os.makedirs(os.path.join(cfg.result_dir, "scores"), exist_ok=True)
    os.makedirs(os.path.join(cfg.result_dir, "metrics"), exist_ok=True)

    with open(os.path.join(cfg.result_dir, "scores", f"{tag}.pkl"), "wb") as f:
        pickle.dump({"scores": all_scores, "param_list": param_list}, f)

    metrics = compute_metrics(all_scores, load_labels(cfg.data_dir, cfg.task), param_list)
    metrics.update(method=cfg.method.name, task=cfg.task, seed=cfg.seed)
    with open(os.path.join(cfg.result_dir, "metrics", f"{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cont = metrics["continuous"]
    calibrated = [
        r["MCC"] for r in metrics["conformal"] if r["scal_variant"] == "variant1" and 0.8 - 1e-9 <= r["alpha"] <= 0.95 + 1e-9
    ]
    print(
        f"\n{cfg.method.name} | {cfg.task} | seed {cfg.seed}\n"
        f"  AUPR {cont['aupr']:.3f} | max MCC {cont['max_mcc']:.3f} | AUROC {cont['auroc']:.3f}\n"
        f"  calibrated MCC (s1, alpha in [0.80, 0.95]) {np.mean(calibrated):.3f}"
    )
    return metrics
