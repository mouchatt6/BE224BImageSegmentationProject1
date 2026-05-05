from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NeedleDataset, infer_repo_root, load_training_index
from .losses import batch_dice, batch_sensitivity, build_loss, composite_score
from .model import build_unet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first U-Net needle segmentation model.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unet_model"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--loss",
        choices=["bce_dice", "dice_tversky", "bce_dice_tversky", "focal_tversky"],
        default="bce_dice_tversky",
        help="Training loss to optimize. Use bce_dice to reproduce the original run.",
    )
    parser.add_argument("--bce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.4)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--focal-weight", type=float, default=0.0)
    parser.add_argument("--tversky-fp-weight", type=float, default=0.3)
    parser.add_argument("--tversky-fn-weight", type=float, default=0.7)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-batch-checks", action="store_true", help="Disable CPU-side image/mask validation checks.")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-valid-batches", type=int, default=None)
    return parser


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_output_dir(repo_root: Path, output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else repo_root / output_dir


def validate_batch_tensors(batch: dict, batch_idx: int, phase: str) -> None:
    images = batch["image"]
    masks = batch.get("mask")

    if not torch.isfinite(images).all():
        raise ValueError(f"{phase} batch {batch_idx} contains NaN or infinite image values.")

    if masks is None:
        return

    if not torch.isfinite(masks).all():
        raise ValueError(f"{phase} batch {batch_idx} contains NaN or infinite mask values.")

    mask_min = float(masks.min())
    mask_max = float(masks.max())
    if mask_min < 0.0 or mask_max > 1.0:
        raise ValueError(
            f"{phase} batch {batch_idx} has mask values outside [0, 1]: "
            f"min={mask_min:.6f}, max={mask_max:.6f}. Check mask loading/binarization."
        )


def assert_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or infinite values.")


def train_one_epoch(model, loader, optimizer, criterion, device, limit_batches=None, validate_batches=True) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if validate_batches:
            validate_batch_tensors(batch, batch_idx, "train")

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        assert_finite_tensor(logits, f"train batch {batch_idx} logits")
        loss = criterion(logits, masks)
        assert_finite_tensor(loss, f"train batch {batch_idx} loss")
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_seen += images.size(0)

        if limit_batches is not None and batch_idx >= limit_batches:
            break

    return total_loss / max(1, total_seen)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float, limit_batches=None, validate_batches=True) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    dice_values = []
    sensitivity_values = []
    pred_pixels = []
    true_pixels = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="valid", leave=False), start=1):
        if validate_batches:
            validate_batch_tensors(batch, batch_idx, "valid")

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        assert_finite_tensor(logits, f"valid batch {batch_idx} logits")
        loss = criterion(logits, masks)
        assert_finite_tensor(loss, f"valid batch {batch_idx} loss")
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        dice = batch_dice(preds, masks)
        sensitivity = batch_sensitivity(preds, masks)

        total_loss += float(loss.item()) * images.size(0)
        total_seen += images.size(0)
        dice_values.extend(dice.detach().cpu().tolist())
        sensitivity_values.extend(sensitivity.detach().cpu().tolist())
        pred_pixels.extend(preds.flatten(start_dim=1).sum(dim=1).detach().cpu().tolist())
        true_pixels.extend(masks.flatten(start_dim=1).sum(dim=1).detach().cpu().tolist())

        if limit_batches is not None and batch_idx >= limit_batches:
            break

    dice_arr = torch.tensor(dice_values)
    sens_arr = torch.tensor(sensitivity_values)
    return {
        "valid_loss": total_loss / max(1, total_seen),
        "dice": float(dice_arr.mean()),
        "sensitivity": float(sens_arr.mean()),
        "score_alpha_025": float(composite_score(dice_arr, sens_arr, 0.25).mean()),
        "score_alpha_050": float(composite_score(dice_arr, sens_arr, 0.50).mean()),
        "score_alpha_075": float(composite_score(dice_arr, sens_arr, 0.75).mean()),
        "pred_pixels": float(np.mean(pred_pixels)),
        "true_pixels": float(np.mean(true_pixels)),
    }


def write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    repo_root = (args.repo_root if args.repo_root is not None else infer_repo_root()).resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device()

    data_root, train_df, train_images, train_masks, _ = load_training_index(repo_root, data_root=data_root)
    train_df_split, valid_df = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train_df["status"],
    )

    train_dataset = NeedleDataset(train_df_split, train_images, train_masks, augment=True, seed=args.seed)
    valid_dataset = NeedleDataset(valid_df, train_images, train_masks, augment=False, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_unet(base_channels=args.base_channels).to(device)
    criterion = build_loss(
        args.loss,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
        focal_weight=args.focal_weight,
        tversky_fp_weight=args.tversky_fp_weight,
        tversky_fn_weight=args.tversky_fn_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    print(f"Repo root: {repo_root}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Valid images: {len(valid_dataset)}")
    print(f"Loss: {args.loss}")
    print(f"Batch checks: {not args.skip_batch_checks}")

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            limit_batches=args.limit_train_batches,
            validate_batches=not args.skip_batch_checks,
        )
        metrics = validate(
            model,
            valid_loader,
            criterion,
            device,
            threshold=args.threshold,
            limit_batches=args.limit_valid_batches,
            validate_batches=not args.skip_batch_checks,
        )
        scheduler.step(metrics["score_alpha_050"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time() - start,
        }
        history.append(row)
        write_history(output_dir / "training_history.csv", history)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} valid_loss={metrics['valid_loss']:.4f} "
            f"dice={metrics['dice']:.4f} sens={metrics['sensitivity']:.4f} "
            f"score50={metrics['score_alpha_050']:.4f}"
        )

        if metrics["score_alpha_050"] > best_score:
            best_score = metrics["score_alpha_050"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_score_alpha_050": best_score,
                "threshold": args.threshold,
            }
            torch.save(checkpoint, output_dir / "best_unet.pt")

    print(f"Best validation alpha=0.50 score: {best_score:.6f}")
    print(f"Best checkpoint: {output_dir / 'best_unet.pt'}")


if __name__ == "__main__":
    main()
