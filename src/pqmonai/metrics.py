from __future__ import annotations

from typing import Dict

import torch


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def f1_binary_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute F1 for binary classification from logits (shape [B,2]) and labels (shape [B])."""
    pred = torch.argmax(logits, dim=1)
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp / denom) if denom > 0 else 0.0)


def auc_binary_from_probs(probs_pos: torch.Tensor, y: torch.Tensor) -> float:
    """Simple ROC AUC implementation (no sklearn dependency).

    probs_pos: shape [B], y: shape [B] in {0,1}
    """
    # Sort by predicted score descending
    scores, order = torch.sort(probs_pos, descending=True)
    y_sorted = y[order]
    # Compute TPR/FPR points
    P = (y_sorted == 1).sum().item()
    N = (y_sorted == 0).sum().item()
    if P == 0 or N == 0:
        return float('nan')
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_score = None
    prev_tpr = 0.0
    prev_fpr = 0.0
    for s, label in zip(scores.tolist(), y_sorted.tolist()):
        if prev_score is None or s != prev_score:
            # trapezoid area
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_fpr, prev_tpr = fpr, tpr
            prev_score = s
        if label == 1:
            tpr += 1.0 / P
        else:
            fpr += 1.0 / N
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
    return float(auc)
