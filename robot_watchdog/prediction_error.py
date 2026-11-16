"""Prediction error of the predictive models: training loss and failure score."""

import torch

ALPHA_POS = 10.0  # weight of the position term
BETA_EDGE = 0.5  # weight of the edge (pairwise relation) term


def compute_edge_features(preds, targets, eps=1e-8):
    """Pairwise relation features of the predicted and target positions (B, L, C, 2).

    Edge features (i, j): [dx, dy, distance, sin(theta), cos(theta)], self-pairs masked out.
    The predictions are detached: the edge term weighs in the failure score, not in the gradient.
    Returns mask_edge (1, 1, C, C, 1), edge_pred and edge_tgt (B, L, C, C, 5).
    """
    preds = preds.detach()
    C = preds.size(2)

    def edges(pos):
        diffs = pos.unsqueeze(3) - pos.unsqueeze(2)  # (B, L, C, C, 2)
        dx, dy = diffs[..., 0], diffs[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + eps)
        angle = torch.atan2(dy, dx)
        return torch.stack([dx, dy, dist, torch.sin(angle), torch.cos(angle)], dim=-1)

    idx = torch.arange(C, device=preds.device)
    mask_edge = (idx[:, None] != idx[None, :]).view(1, 1, C, C, 1).float()

    return mask_edge, edges(preds) * mask_edge, edges(targets) * mask_edge


def prediction_error(preds, targets, eps=1e-8, per_sample=False):
    """L = ALPHA_POS * L_pos + BETA_EDGE * L_edge, with L_pos the MSE of the predicted positions.

    Returns a scalar for the batch (training loss), or one value per window if per_sample (failure
    score, equal to the scalar computed on a batch of one window).
    """
    dims = (1, 2, 3) if per_sample else None
    loss_pos = ((preds - targets) ** 2).mean(dim=dims)
    mask_edge, edge_pred, edge_tgt = compute_edge_features(preds, targets, eps)
    edge_err = ((edge_pred - edge_tgt) ** 2) * mask_edge
    edge_err = edge_err.flatten(1).sum(dim=1) if per_sample else edge_err.sum()
    loss_edge = edge_err / (mask_edge.sum() * 5 + eps)
    return ALPHA_POS * loss_pos + BETA_EDGE * loss_edge
