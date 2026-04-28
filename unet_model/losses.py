from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def batch_dice(pred_masks: torch.Tensor, true_masks: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred_masks.float().flatten(start_dim=1)
    true = true_masks.float().flatten(start_dim=1)
    intersection = (pred * true).sum(dim=1)
    denominator = pred.sum(dim=1) + true.sum(dim=1)
    empty = denominator == 0
    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice = torch.where(empty, torch.ones_like(dice), dice)
    return dice


def batch_sensitivity(pred_masks: torch.Tensor, true_masks: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred_masks.float().flatten(start_dim=1)
    true = true_masks.float().flatten(start_dim=1)
    tp = (pred * true).sum(dim=1)
    true_pixels = true.sum(dim=1)
    pred_pixels = pred.sum(dim=1)
    sensitivity = (tp + eps) / (true_pixels + eps)
    empty_score = torch.where(pred_pixels == 0, torch.ones_like(sensitivity), torch.zeros_like(sensitivity))
    return torch.where(true_pixels == 0, empty_score, sensitivity)


def composite_score(dice: torch.Tensor, sensitivity: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * dice + (1.0 - alpha) * sensitivity

