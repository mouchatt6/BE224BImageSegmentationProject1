from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_model.data_io import save_binary_png
from unet_model.dataset import NeedleDataset, infer_repo_root, load_training_index, make_test_dataframe
from unet_model.postprocess import postprocess_probability_map
from unet_model.predict_unet import choose_device, resolve_path
from .model import build_unetpp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export U-Net++ test masks as binary PNGs.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/unetpp_model/best_unetpp.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unetpp_model/test_masks"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--close-kernel-size", type=int, default=3)
    parser.add_argument("--dilation-iterations", type=int, default=0)
    parser.add_argument("--keep-single-component", action="store_true", default=True)
    parser.add_argument("--keep-all-components", action="store_false", dest="keep_single_component")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    repo_root = (args.repo_root if args.repo_root is not None else infer_repo_root()).resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    _, _, _, _, test_images = load_training_index(repo_root, data_root=data_root)
    test_df = make_test_dataframe(test_images)
    dataset = NeedleDataset(test_df, test_images, mask_paths=None, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = int(checkpoint_args.get("base_channels", args.base_channels))
    deep_supervision = bool(checkpoint_args.get("deep_supervision", False))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))

    model = build_unetpp(base_channels=base_channels, deep_supervision=deep_supervision).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for batch in tqdm(loader, desc="predict"):
        images = batch["image"].to(device)
        image_ids = batch["image_id"]
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()

        for prob, image_id in zip(probs, image_ids):
            mask = postprocess_probability_map(
                prob[0],
                threshold=threshold,
                min_area=args.min_area,
                close_kernel_size=args.close_kernel_size,
                dilation_iterations=args.dilation_iterations,
                keep_single_component=args.keep_single_component,
            )
            save_binary_png(output_dir / f"{image_id}_mask.png", mask)

    print(f"Exported {len(test_df)} masks to: {output_dir}")


if __name__ == "__main__":
    main()

