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


class TverskyLoss(nn.Module):
    def __init__(self, fp_weight: float = 0.3, fn_weight: float = 0.7, eps: float = 1e-7) -> None:
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        score = (tp + self.eps) / (tp + self.fp_weight * fp + self.fn_weight * fn + self.eps)
        return 1.0 - score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        tversky_weight: float = 0.2,
        focal_weight: float = 0.0,
        tversky_fp_weight: float = 0.3,
        tversky_fn_weight: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(fp_weight=tversky_fp_weight, fn_weight=tversky_fn_weight)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.bce_weight:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.tversky_weight:
            loss = loss + self.tversky_weight * self.tversky(logits, targets)
        if self.focal_weight:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss


def build_loss(
    name: str,
    bce_weight: float = 0.4,
    dice_weight: float = 0.4,
    tversky_weight: float = 0.2,
    focal_weight: float = 0.0,
    tversky_fp_weight: float = 0.3,
    tversky_fn_weight: float = 0.7,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    if name == "dice_tversky":
        return CombinedSegmentationLoss(
            bce_weight=0.0,
            dice_weight=0.5,
            tversky_weight=0.5,
            tversky_fp_weight=tversky_fp_weight,
            tversky_fn_weight=tversky_fn_weight,
        )
    if name == "bce_dice_tversky":
        return CombinedSegmentationLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            tversky_weight=tversky_weight,
            tversky_fp_weight=tversky_fp_weight,
            tversky_fn_weight=tversky_fn_weight,
        )
    if name == "focal_tversky":
        return CombinedSegmentationLoss(
            bce_weight=0.0,
            dice_weight=0.2,
            tversky_weight=0.5,
            focal_weight=0.3 if focal_weight == 0.0 else focal_weight,
            tversky_fp_weight=tversky_fp_weight,
            tversky_fn_weight=tversky_fn_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
    raise ValueError(f"Unknown loss: {name}")


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
