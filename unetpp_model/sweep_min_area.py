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
    parser = argparse.ArgumentParser(description="Sweep U-Net++ min_area while holding other post-processing settings fixed.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/unetpp_model/best_unetpp.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/unetpp_model/min_area_sweep.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--min-areas", type=int, nargs="+", default=[40, 80, 120, 160, 200, 240, 320])
    parser.add_argument("--close-kernel-size", type=int, default=7)
    parser.add_argument("--dilation-iterations", type=int, default=0)
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
    _, valid_df = train_test_split(train_df, test_size=valid_size, random_state=seed, stratify=train_df["status"])
    valid_dataset = NeedleDataset(valid_df, train_images, train_masks, augment=False, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_unetpp(base_channels=base_channels, deep_supervision=deep_supervision).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Validation images: {len(valid_dataset)}")
    print(
        "Fixed settings: "
        f"threshold={args.threshold:.3f} close_kernel_size={args.close_kernel_size} "
        f"dilation_iterations={args.dilation_iterations}"
    )

    probabilities, targets, _ = collect_validation_probabilities(
        model,
        valid_loader,
        device,
        limit_valid_batches=args.limit_valid_batches,
    )

    rows = []
    for min_area in tqdm(args.min_areas, desc="min_area sweep"):
        rows.append(
            evaluate_config(
                probabilities,
                targets,
                threshold=args.threshold,
                min_area=min_area,
                close_kernel_size=args.close_kernel_size,
                dilation_iterations=args.dilation_iterations,
                keep_single_component=args.keep_single_component,
            )
        )

    rows = sorted(rows, key=lambda item: (item["score_alpha_050"], item["score_alpha_025"]), reverse=True)
    write_csv(output_csv, rows)

    print("\nTop min_area configs by score_alpha_050:")
    for row in rows[:10]:
        print(
            f"min_area={row['min_area']} threshold={row['threshold']:.3f} "
            f"close={row['close_kernel_size']} dilate={row['dilation_iterations']} "
            f"dice={row['dice']:.4f} sens={row['sensitivity']:.4f} "
            f"score50={row['score_alpha_050']:.4f}"
        )


if __name__ == "__main__":
    main()

