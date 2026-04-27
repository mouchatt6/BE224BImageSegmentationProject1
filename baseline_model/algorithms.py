from __future__ import annotations

import cv2
import numpy as np


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_area <= 1:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned


def keep_best_component(mask: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_label = 0
    best_score = -1.0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        elongation = max(width, height) / max(1, min(width, height))

        if image is not None:
            mean_intensity = float(image[labels == label].mean())
        else:
            mean_intensity = 1.0

        score = area * max(1.0, elongation) * max(1.0, mean_intensity)
        if score > best_score:
            best_score = score
            best_label = label

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    if best_label > 0:
        cleaned[labels == best_label] = 1
    return cleaned


def close_and_dilate(mask: np.ndarray, close_kernel_size: int = 3, dilation_iterations: int = 0) -> np.ndarray:
    cleaned = (mask > 0).astype(np.uint8)

    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if dilation_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.dilate(cleaned, kernel, iterations=dilation_iterations)

    return (cleaned > 0).astype(np.uint8)


def percentile_threshold_segmentation(
    image: np.ndarray,
    percentile: float = 99.5,
    blur_kernel_size: int = 3,
    min_area: int = 5,
    close_kernel_size: int = 3,
    keep_single_component: bool = True,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Segment bright needle candidates using high-intensity thresholding."""
    image_u8 = normalize_uint8(image)

    if blur_kernel_size and blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        working = cv2.GaussianBlur(image_u8, (blur_kernel_size, blur_kernel_size), 0)
    else:
        working = image_u8

    threshold = np.percentile(working, percentile)
    mask = (working >= threshold).astype(np.uint8)
    mask = remove_small_components(mask, min_area=min_area)

    if keep_single_component:
        mask = keep_best_component(mask, image=working)

    return close_and_dilate(mask, close_kernel_size=close_kernel_size, dilation_iterations=dilation_iterations)


def otsu_threshold_segmentation(
    image: np.ndarray,
    min_area: int = 5,
    close_kernel_size: int = 3,
    keep_single_component: bool = True,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Segment bright structures with Otsu thresholding."""
    image_u8 = normalize_uint8(image)
    blurred = cv2.GaussianBlur(image_u8, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = remove_small_components(mask, min_area=min_area)

    if keep_single_component:
        mask = keep_best_component(mask, image=blurred)

    return close_and_dilate(mask, close_kernel_size=close_kernel_size, dilation_iterations=dilation_iterations)


def hough_line_segmentation(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 20,
    min_line_length: int = 25,
    max_line_gap: int = 5,
    line_thickness: int = 3,
    high_intensity_percentile: float = 97.0,
    close_kernel_size: int = 3,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Detect linear bright structures using Canny edges and probabilistic Hough lines."""
    image_u8 = normalize_uint8(image)
    blurred = cv2.GaussianBlur(image_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    bright_threshold = np.percentile(blurred, high_intensity_percentile)
    bright_mask = (blurred >= bright_threshold).astype(np.uint8)
    edges = (edges > 0).astype(np.uint8) * bright_mask * 255

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    mask = np.zeros_like(image_u8, dtype=np.uint8)
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=line_thickness)

    mask = mask * bright_mask
    mask = keep_best_component(mask, image=blurred)
    return close_and_dilate(mask, close_kernel_size=close_kernel_size, dilation_iterations=dilation_iterations)

