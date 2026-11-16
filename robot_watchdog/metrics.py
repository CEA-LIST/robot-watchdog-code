"""Step-level evaluation of a failure detector against the per-frame ground truth.

All the test episodes of a task are concatenated before computing the metrics.
    - threshold-independent: AUROC, AUPR and the MCC at the threshold maximising the F1 score along
      the precision-recall curve ("max MCC" of the paper figures and tables);
    - end-to-end: TPR, TNR and MCC of the binary decisions of every conformal configuration.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def align(all_scores, all_labels):
    """Concatenate scores, binary decisions and labels of all episodes.

    The score sequence of an episode can be shorter than its labels (the predictive models need a
    full observation window and prediction horizon): labels are then truncated to the scores.
    """
    assert len(all_scores) == len(all_labels), "not the same number of episodes in the scores and the labels"
    y_true, y_score, y_bin = [], [], []
    for scene, gt in zip(all_scores, all_labels):
        raw = np.array(scene["raw_scores"])
        binary = scene["binary_scores"]
        assert len(raw) == len(binary), "not the same number of steps in the raw and binary scores"
        if len(raw) < len(gt):
            gt = gt[: len(raw)]
        assert len(raw) == len(gt), "not the same number of steps in the scores and the labels"
        y_true.append(gt)
        y_score.append(raw)
        y_bin.append(binary)
    return np.concatenate(y_true), np.concatenate(y_score), np.concatenate(y_bin, axis=0)


def continuous_metrics(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    f1s = (2 * precision * recall) / (precision + recall + 1e-12)
    best_threshold = thresholds[np.nanargmax(f1s)]
    preds = (y_score >= best_threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "max_mcc": float(matthews_corrcoef(y_true, preds)),
        "best_threshold": float(best_threshold),
    }


def binary_metrics(y_true, y_pred_bin, param_list):
    """y_pred_bin: [T, K], one column per conformal configuration (param_list[k])."""
    y_true = y_true[: y_pred_bin.shape[0]]
    results = []
    for k in range(y_pred_bin.shape[1]):
        preds = y_pred_bin[:, k]
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        results.append(
            {
                **param_list[k],
                "TPR": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
                "TNR": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
                "MCC": float(matthews_corrcoef(y_true, preds)),
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        )
    return results


def compute_metrics(all_scores, all_labels, param_list):
    y_true, y_score, y_bin = align(all_scores, all_labels)
    return {
        "continuous": continuous_metrics(y_true, y_score),
        "conformal": binary_metrics(y_true, y_bin, param_list),
    }
