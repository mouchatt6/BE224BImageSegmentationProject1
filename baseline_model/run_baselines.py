from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

try:
    from .algorithms import (
        hough_line_segmentation,
        otsu_threshold_segmentation,
        percentile_threshold_segmentation,
    )
    from .data_io import load_binary_mask, load_dataset, load_grayscale, save_binary_png
    from .metrics import evaluate_mask
except ImportError:
    from algorithms import hough_line_segmentation, otsu_threshold_segmentation, percentile_threshold_segmentation
    from data_io import load_binary_mask, load_dataset, load_grayscale, save_binary_png
    from metrics import evaluate_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classical needle segmentation baselines.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Path to the Git repo root.")
    parser.add_argument(
        "--mode",
        choices=("validate", "predict-test"),
        default="validate",
        help="Validate on a stratified split or export test masks.",
    )
    parser.add_argument(
        "--method",
        choices=("percentile", "otsu", "hough"),
        default="percentile",
        help="Classical segmentation algorithm.",
    )
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation split.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_model"), help="Output folder.")

    parser.add_argument("--percentile", type=float, default=99.5, help="High-intensity percentile threshold.")
    parser.add_argument("--min-area", type=int, default=5, help="Minimum connected-component area.")
    parser.add_argument("--close-kernel-size", type=int, default=3, help="Morphological closing kernel size.")
    parser.add_argument("--dilation-iterations", type=int, default=0, help="Optional dilation iterations.")
    parser.add_argument("--keep-single-component", action="store_true", default=True)
    parser.add_argument("--keep-all-components", action="store_false", dest="keep_single_component")

    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=20)
    parser.add_argument("--min-line-length", type=int, default=25)
    parser.add_argument("--max-line-gap", type=int, default=5)
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--high-intensity-percentile", type=float, default=97.0)

    return parser


def segment_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.method == "percentile":
        return percentile_threshold_segmentation(
            image,
            percentile=args.percentile,
            min_area=args.min_area,
            close_kernel_size=args.close_kernel_size,
            keep_single_component=args.keep_single_component,
            dilation_iterations=args.dilation_iterations,
        )

    if args.method == "otsu":
        return otsu_threshold_segmentation(
            image,
            min_area=args.min_area,
            close_kernel_size=args.close_kernel_size,
            keep_single_component=args.keep_single_component,
            dilation_iterations=args.dilation_iterations,
        )

    if args.method == "hough":
        return hough_line_segmentation(
            image,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            hough_threshold=args.hough_threshold,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap,
            line_thickness=args.line_thickness,
            high_intensity_percentile=args.high_intensity_percentile,
            close_kernel_size=args.close_kernel_size,
            dilation_iterations=args.dilation_iterations,
        )

    raise ValueError(f"Unknown method: {args.method}")


def summarize_metric_rows(rows: list[dict]) -> dict[str, float]:
    numeric_columns = [
        "dice",
        "sensitivity",
        "score_alpha_025",
        "score_alpha_050",
        "score_alpha_075",
        "pred_pixels",
        "true_pixels",
    ]
    return {column: float(np.mean([row[column] for row in rows])) for column in numeric_columns}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(args: argparse.Namespace) -> None:
    data_root, train_df, train_images, train_masks, _ = load_dataset(args.repo_root)

    _, valid_df = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train_df["status"],
    )

    rows = []
    missing_ids = []

    for image_id in valid_df["imageID"].astype(str):
        if image_id not in train_images or image_id not in train_masks:
            missing_ids.append(image_id)
            continue

        image = load_grayscale(train_images[image_id])
        true_mask = load_binary_mask(train_masks[image_id])
        pred_mask = segment_image(image, args)
        rows.append(asdict(evaluate_mask(image_id, pred_mask, true_mask)))

    if missing_ids:
        print(f"Skipped {len(missing_ids)} IDs missing image or mask files: {missing_ids[:10]}")
    if not rows:
        raise RuntimeError("No validation rows were evaluated.")

    output_dir = args.repo_root / args.output_dir / args.method
    write_csv(output_dir / "validation_metrics.csv", rows)
    summary = summarize_metric_rows(rows)
    write_csv(output_dir / "validation_summary.csv", [summary])

    print(f"Data root: {data_root}")
    print(f"Method: {args.method}")
    print(f"Validation images evaluated: {len(rows)}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")
    print(f"Wrote metrics to: {output_dir}")


def run_test_prediction(args: argparse.Namespace) -> None:
    data_root, _, _, _, test_images = load_dataset(args.repo_root)
    output_dir = args.repo_root / args.output_dir / args.method / "test_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_path in sorted(test_images.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        image = load_grayscale(image_path)
        pred_mask = segment_image(image, args)
        save_binary_png(output_dir / f"{image_id}_mask.png", pred_mask)

    print(f"Data root: {data_root}")
    print(f"Method: {args.method}")
    print(f"Exported {len(test_images)} masks to: {output_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.mode == "validate":
        run_validation(args)
    else:
        run_test_prediction(args)


if __name__ == "__main__":
    main()

