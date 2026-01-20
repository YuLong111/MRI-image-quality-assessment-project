from __future__ import annotations

from typing import Any, Dict, List

import torch


@torch.no_grad()
def predict_binary(model, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model.eval()
    x = batch["image"].to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    return {
        "pred": pred.detach().cpu(),
        "prob_pos": probs[:, 1].detach().cpu(),
    }
