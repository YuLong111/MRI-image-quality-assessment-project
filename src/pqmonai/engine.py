from __future__ import annotations

from typing import Any, Dict

import torch
from monai.data import DataLoader
from tqdm import tqdm

from .losses import loss_binary_cross_entropy_logits, loss_multitask
from .metrics import accuracy_from_logits, f1_binary_from_logits, auc_binary_from_probs


def train_one_epoch_binary(model, loader: DataLoader, optimizer, device: torch.device, label_key: str = "label_bin") -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)
        y = batch[label_key].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_binary_cross_entropy_logits(logits, y)
        loss.backward()
        optimizer.step()

        b = x.shape[0]
        total_loss += float(loss.item()) * b
        total_acc += accuracy_from_logits(logits.detach(), y) * b
        total_f1 += f1_binary_from_logits(logits.detach(), y) * b
        n += b

    return {
        "loss": total_loss / max(1, n),
        "acc": total_acc / max(1, n),
        "f1": total_f1 / max(1, n),
    }


@torch.no_grad()
def eval_binary(model, loader: DataLoader, device: torch.device, label_key: str = "label_bin") -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_y = []
    total_acc = 0.0
    total_f1 = 0.0
    n = 0

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch[label_key].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        all_probs.append(probs)
        all_y.append(y.detach().cpu())

        b = x.shape[0]
        total_acc += accuracy_from_logits(logits.detach(), y) * b
        total_f1 += f1_binary_from_logits(logits.detach(), y) * b
        n += b

    probs_pos = torch.cat(all_probs) if all_probs else torch.tensor([])
    y_all = torch.cat(all_y) if all_y else torch.tensor([])
    auc = auc_binary_from_probs(probs_pos, y_all) if len(probs_pos) else float('nan')

    return {
        "acc": total_acc / max(1, n),
        "f1": total_f1 / max(1, n),
        "auc": auc,
    }
