"""Evaluate one failure detector on one task for one seed.

    uv run python -m robot_watchdog.run method=TPC task=sort_trash_conveyor seed=1
    uv run python -m robot_watchdog.run method=gnn_transformer_3 task=aloha_setTheTable_2 seed=1

Non-parametric methods are evaluated directly; predictive methods are first trained on the expert
demonstrations. Scores and metrics are written to <result_dir>/{scores,metrics}/<method>_<task>.*
"""

import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(int(cfg.seed))
    print(f"method: {cfg.method.name} | task: {cfg.task} | seed: {cfg.seed}")

    from robot_watchdog.evaluate import evaluate_non_parametric, evaluate_predictive
    from robot_watchdog.train import train

    if cfg.method.kind == "non_parametric":
        evaluate_non_parametric(cfg)
    elif cfg.method.kind == "predictive":
        train(cfg)
        torch.cuda.empty_cache()
        evaluate_predictive(cfg)
    else:
        raise ValueError(f"Unknown method kind: {cfg.method.kind}")


if __name__ == "__main__":
    main()
