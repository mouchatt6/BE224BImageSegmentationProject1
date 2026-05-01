from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NeedleDataset, infer_repo_root, load_training_index
from .losses import batch_dice, batch_sensitivity, composite_score
from .model import build_unet
from .postprocess import postprocess_probability_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep U-Net thresholds and post-processing settings on validation data.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/unet_model/best_unet.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/unet_model/threshold_sweep.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    )
    parser.add_argument("--min-areas", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--close-kernel-sizes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--dilation-iterations", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--keep-single-component", action="store_true", default=True)
    parser.add_argument("--keep-all-components", action="store_false", dest="keep_single_component")
    parser.add_argument("--limit-valid-batches", type=int, default=None)
    return parser


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_validation_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_valid_batches: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    image_ids: list[str] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="predict valid", leave=False), start=1):
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        masks = batch["mask"].detach().cpu().numpy()

        for prob, mask, image_id in zip(probs, masks, batch["image_id"]):
            probabilities.append(prob[0].astype(np.float32))
            targets.append((mask[0] > 0).astype(np.uint8))
            image_ids.append(str(image_id))

        if limit_valid_batches is not None and batch_idx >= limit_valid_batches:
            break

    return probabilities, targets, image_ids


def evaluate_config(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float,
    min_area: int,
    close_kernel_size: int,
    dilation_iterations: int,
    keep_single_component: bool,
) -> dict[str, float | int]:
    pred_tensors = []
    true_tensors = []
    pred_pixels = []
    true_pixels = []

    for prob, true_mask in zip(probabilities, targets):
        pred_mask = postprocess_probability_map(
            prob,
            threshold=threshold,
            min_area=min_area,
            close_kernel_size=close_kernel_size,
            dilation_iterations=dilation_iterations,
            keep_single_component=keep_single_component,
        )
        pred_tensors.append(torch.from_numpy(pred_mask).unsqueeze(0))
        true_tensors.append(torch.from_numpy(true_mask).unsqueeze(0))
        pred_pixels.append(int(pred_mask.sum()))
        true_pixels.append(int(true_mask.sum()))

    pred_batch = torch.stack(pred_tensors, dim=0)
    true_batch = torch.stack(true_tensors, dim=0)
    dice = batch_dice(pred_batch, true_batch)
    sensitivity = batch_sensitivity(pred_batch, true_batch)

    return {
        "threshold": threshold,
        "min_area": min_area,
        "close_kernel_size": close_kernel_size,
        "dilation_iterations": dilation_iterations,
        "keep_single_component": int(keep_single_component),
        "dice": float(dice.mean()),
        "sensitivity": float(sensitivity.mean()),
        "score_alpha_025": float(composite_score(dice, sensitivity, 0.25).mean()),
        "score_alpha_050": float(composite_score(dice, sensitivity, 0.50).mean()),
        "score_alpha_075": float(composite_score(dice, sensitivity, 0.75).mean()),
        "pred_pixels": float(np.mean(pred_pixels)),
        "true_pixels": float(np.mean(true_pixels)),
        "nonempty_predictions": int(sum(value > 0 for value in pred_pixels)),
    }


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

    data_root, train_df, train_images, train_masks, _ = load_training_index(repo_root, data_root=data_root)
    _, valid_df = train_test_split(
        train_df,
        test_size=valid_size,
        random_state=seed,
        stratify=train_df["status"],
    )

    valid_dataset = NeedleDataset(valid_df, train_images, train_masks, augment=False, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_unet(base_channels=base_channels).to(device)
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
                        row = evaluate_config(
                            probabilities,
                            targets,
                            threshold=threshold,
                            min_area=min_area,
                            close_kernel_size=close_kernel_size,
                            dilation_iterations=dilation_iterations,
                            keep_single_component=args.keep_single_component,
                        )
                        rows.append(row)
                        progress.update(1)

    rows = sorted(rows, key=lambda item: (item["score_alpha_050"], item["score_alpha_025"]), reverse=True)
    write_csv(output_csv, rows)

    print("\nTop 10 configs by score_alpha_050:")
    for row in rows[:10]:
        print(
            f"threshold={row['threshold']:.2f} min_area={row['min_area']} "
            f"close={row['close_kernel_size']} dilate={row['dilation_iterations']} "
            f"dice={row['dice']:.4f} sens={row['sensitivity']:.4f} "
            f"score25={row['score_alpha_025']:.4f} score50={row['score_alpha_050']:.4f} "
            f"score75={row['score_alpha_075']:.4f}"
        )


if __name__ == "__main__":
    main()

