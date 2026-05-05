from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_model.dataset import NeedleDataset, infer_repo_root, load_training_index
from unet_model.sweep_thresholds import (
    choose_device,
    collect_validation_probabilities,
    evaluate_config,
    resolve_path,
    write_csv,
)
from .model import build_unetpp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep U-Net++ thresholds and post-processing settings on validation data.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/unetpp_model/best_unetpp.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/unetpp_model/threshold_sweep.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.250,
            0.275,
            0.300,
            0.325,
            0.350,
            0.375,
            0.400,
            0.425,
            0.450,
            0.475,
            0.500,
            0.525,
            0.550,
            0.575,
            0.600,
            0.625,
            0.650,
            0.675,
            0.700,
            0.725,
            0.750,
        ],
    )
    parser.add_argument("--min-areas", type=int, nargs="+", default=[3, 5, 10, 20, 40, 80, 120, 160])
    parser.add_argument("--close-kernel-sizes", type=int, nargs="+", default=[0, 3, 5, 7, 9])
    parser.add_argument("--dilation-iterations", type=int, nargs="+", default=[0])
    parser.add_argument("--keep-single-component", action="store_true", default=True)
    parser.add_argument("--keep-all-components", action="store_false", dest="keep_single_component")
    parser.add_argument("--limit-valid-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = (args.repo_root if args.repo_root is not None else infer_repo_root()).resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    output_csv = resolve_path(repo_root, args.output_csv)

    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    seed = int(args.seed if args.seed is not None else checkpoint_args.get("seed", 42))
    valid_size = float(args.valid_size if args.valid_size is not None else checkpoint_args.get("valid_size", 0.2))
    base_channels = int(args.base_channels if args.base_channels is not None else checkpoint_args.get("base_channels", 32))
    deep_supervision = bool(checkpoint_args.get("deep_supervision", False))

    data_root, train_df, train_images, train_masks, _ = load_training_index(repo_root, data_root=data_root)
    _, valid_df = train_test_split(
        train_df,
        test_size=valid_size,
        random_state=seed,
        stratify=train_df["status"],
    )

    valid_dataset = NeedleDataset(valid_df, train_images, train_masks, augment=False, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_unetpp(base_channels=base_channels, deep_supervision=deep_supervision).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Repo root: {repo_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Device: {device}")
    print(f"Validation images: {len(valid_dataset)}")

    probabilities, targets, _ = collect_validation_probabilities(
        model,
        valid_loader,
        device,
        limit_valid_batches=args.limit_valid_batches,
    )

    rows = []
    total = len(args.thresholds) * len(args.min_areas) * len(args.close_kernel_sizes) * len(args.dilation_iterations)
    with tqdm(total=total, desc="sweep") as progress:
        for threshold in args.thresholds:
            for min_area in args.min_areas:
                for close_kernel_size in args.close_kernel_sizes:
                    for dilation_iterations in args.dilation_iterations:
                        rows.append(
                            evaluate_config(
                                probabilities,
                                targets,
                                threshold=threshold,
                                min_area=min_area,
                                close_kernel_size=close_kernel_size,
                                dilation_iterations=dilation_iterations,
                                keep_single_component=args.keep_single_component,
                            )
                        )
                        progress.update(1)

    rows = sorted(rows, key=lambda item: (item["score_alpha_050"], item["score_alpha_025"]), reverse=True)
    write_csv(output_csv, rows)

    print("\nTop 10 configs by score_alpha_050:")
    for row in rows[:10]:
        print(
            f"threshold={row['threshold']:.3f} min_area={row['min_area']} "
            f"close={row['close_kernel_size']} dilate={row['dilation_iterations']} "
            f"dice={row['dice']:.4f} sens={row['sensitivity']:.4f} "
            f"score25={row['score_alpha_025']:.4f} score50={row['score_alpha_050']:.4f} "
            f"score75={row['score_alpha_075']:.4f}"
        )


if __name__ == "__main__":
    main()

