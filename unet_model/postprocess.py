from __future__ import annotations

import cv2
import numpy as np


def postprocess_probability_map(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 5,
    close_kernel_size: int = 3,
    dilation_iterations: int = 0,
    keep_single_component: bool = True,
) -> np.ndarray:
    binary = (probability_map >= threshold).astype(np.uint8)

    if min_area > 1 or keep_single_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            mean_prob = float(probability_map[labels == label].mean())
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            elongation = max(width, height) / max(1, min(width, height))
            score = area * max(1.0, elongation) * max(mean_prob, 1e-6)
            candidates.append((score, label))

        cleaned = np.zeros_like(binary, dtype=np.uint8)
        if keep_single_component:
            if candidates:
                _, best_label = max(candidates, key=lambda item: item[0])
                cleaned[labels == best_label] = 1
        else:
            for _, label in candidates:
                cleaned[labels == label] = 1
        binary = cleaned

    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if dilation_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilation_iterations)

    return (binary > 0).astype(np.uint8)

