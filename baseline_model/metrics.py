from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    image_id: str
    dice: float
    sensitivity: float
    score_alpha_025: float
    score_alpha_050: float
    score_alpha_075: float
    pred_pixels: int
    true_pixels: int


def as_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = as_binary(pred_mask)
    true = as_binary(true_mask)

    intersection = int(np.logical_and(pred, true).sum())
    pred_pixels = int(pred.sum())
    true_pixels = int(true.sum())
    denominator = pred_pixels + true_pixels

    if denominator == 0:
        return 1.0

    return float((2 * intersection + eps) / (denominator + eps))


def sensitivity_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = as_binary(pred_mask)
    true = as_binary(true_mask)

    true_pixels = int(true.sum())
    pred_pixels = int(pred.sum())

    if true_pixels == 0:
        return 1.0 if pred_pixels == 0 else 0.0

    true_positive = int(np.logical_and(pred, true).sum())
    return float((true_positive + eps) / (true_pixels + eps))


def composite_score(dice: float, sensitivity: float, alpha: float) -> float:
    return float(alpha * dice + (1.0 - alpha) * sensitivity)


def evaluate_mask(image_id: str, pred_mask: np.ndarray, true_mask: np.ndarray) -> MetricResult:
    pred = as_binary(pred_mask)
    true = as_binary(true_mask)
    dice = dice_score(pred, true)
    sensitivity = sensitivity_score(pred, true)

    return MetricResult(
        image_id=str(image_id),
        dice=dice,
        sensitivity=sensitivity,
        score_alpha_025=composite_score(dice, sensitivity, alpha=0.25),
        score_alpha_050=composite_score(dice, sensitivity, alpha=0.50),
        score_alpha_075=composite_score(dice, sensitivity, alpha=0.75),
        pred_pixels=int(pred.sum()),
        true_pixels=int(true.sum()),
    )

