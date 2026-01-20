from __future__ import annotations

import torch


def loss_binary_cross_entropy_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary classification with 2-class logits (B,2) using CrossEntropyLoss."""
    return torch.nn.functional.cross_entropy(logits, y.long())


def loss_multitask(
    out: dict,
    y_bin: torch.Tensor,
    y_score: torch.Tensor | None,
    y_art: torch.Tensor | None,
    w: dict,
) -> torch.Tensor:
    """Optional: multi-task loss.

    out should contain keys: 'logits_bin', and optionally 'logits_score', 'logits_art'.
    """
    loss = torch.nn.functional.cross_entropy(out["logits_bin"], y_bin.long()) * float(w.get("bin", 1.0))

    if y_score is not None and "logits_score" in out:
        loss = loss + torch.nn.functional.cross_entropy(out["logits_score"], y_score.long()) * float(w.get("score", 0.5))

    if y_art is not None and "logits_art" in out:
        # multi-label BCE (expect y_art shape [B, A] in {0,1})
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out["logits_art"], y_art.float()) * float(w.get("artefacts", 0.5))

    return loss
