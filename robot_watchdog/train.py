"""Training of the predictive models on the nominal (expert) demonstrations only."""

import os
import pickle
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from robot_watchdog.anomaly_detection.gnn_transformer import GNNTransformerPredictor
from robot_watchdog.anomaly_detection.gnn_transformer_3 import STGraphTransformer_3
from robot_watchdog.anomaly_detection.gnn_transformer_PE_global import STGraphTransformer_PE_global
from robot_watchdog.dataset.graph import GraphCustomDataset, collate_seq
from robot_watchdog.dataset.io import episodes_from_data_dict, load_trajectories
from robot_watchdog.prediction_error import prediction_error


def build_model(cfg, C):
    m_cfg = cfg.method
    if m_cfg.name == "gnn_transformer":
        return GNNTransformerPredictor(
            gnn_hidden=m_cfg.hidden_dim,
            transformer_dim=m_cfg.hidden_dim,
            nhead=m_cfg.nhead,
            num_layers=m_cfg.num_layers,
            m=m_cfg.m,
            L=m_cfg.L,
            C=C,
        )
    if m_cfg.name == "gnn_transformer_3":
        return STGraphTransformer_3(
            hidden_dim=m_cfg.hidden_dim, nhead=m_cfg.nhead, num_layers=m_cfg.num_layers, L=m_cfg.L, C=C
        )
    if m_cfg.name == "gnn_transformer_PE_global":
        return STGraphTransformer_PE_global(
            hidden_dim=m_cfg.hidden_dim, nhead=m_cfg.nhead, num_layers=m_cfg.num_layers, L=m_cfg.L, C=C
        )
    raise NotImplementedError(f"Unknown predictive method: {m_cfg.name}")


def forward(model, cfg, batch, device):
    inputs = [g.to(device) for g in batch["input_graphs"]]
    if cfg.method.name == "gnn_transformer_PE_global":
        return model(inputs, batch["progress"].to(device))
    return model(inputs)


def weights_path(cfg):
    return os.path.join(cfg.weights_dir, f"{cfg.method.name}_{cfg.task}_seed_{cfg.seed}.pt")


def calibration_split_path(cfg):
    return os.path.join(cfg.result_dir, "calibration_split", f"{cfg.method.name}_{cfg.task}.pkl")


def train(cfg):
    mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    device = torch.device(cfg.device)
    m_cfg = cfg.method

    # 1) split the expert demonstrations: one half trains the model, the other half calibrates
    #    the conformal threshold
    data_dict = load_trajectories(cfg.data_dir, cfg.task, "expert")
    nb_episodes = len(data_dict)
    episode_ids = list(range(nb_episodes))
    random.shuffle(episode_ids)
    split_point = (nb_episodes + 1) // 2
    train_dict = [data_dict[i] for i in episode_ids[:split_point]]
    calib_dict = [data_dict[i] for i in episode_ids[split_point:]]

    os.makedirs(os.path.dirname(calibration_split_path(cfg)), exist_ok=True)
    with open(calibration_split_path(cfg), "wb") as f:
        pickle.dump(calib_dict, f)

    train_episodes = episodes_from_data_dict(train_dict)  # list of (T, C, 2)
    C = train_episodes[0].shape[1]
    print(f"training on {len(train_episodes)} expert demonstrations, {C} objects")

    # 2) model, data and optimiser
    model = build_model(cfg, C).to(device)

    train_dataset = GraphCustomDataset(train_episodes, m=m_cfg.m, L=m_cfg.L)
    dataloader = DataLoader(
        train_dataset,
        batch_size=m_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_seq,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=m_cfg.lr, weight_decay=m_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 3) training loop: future position prediction
    model.train()
    for epoch in range(m_cfg.epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{m_cfg.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            preds = forward(model, cfg, batch, device)  # (B, L, C, 2)
            loss = prediction_error(preds, batch["target_pos"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")
        scheduler.step()
        print(f"epoch {epoch + 1}/{m_cfg.epochs}  loss {running_loss / len(dataloader):.5f}")

    os.makedirs(os.path.dirname(weights_path(cfg)), exist_ok=True)
    torch.save({"epoch": m_cfg.epochs, "model_state": model.state_dict()}, weights_path(cfg))
    print(f"weights saved: {weights_path(cfg)}")
