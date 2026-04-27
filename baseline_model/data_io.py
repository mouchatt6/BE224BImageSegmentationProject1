from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_data_root(start_path: Path) -> Path:
    """Find the folder that contains the local image folders."""
    start_path = Path(start_path).resolve()
    candidates = [start_path, *start_path.parents]

    for candidate in candidates:
        if all((candidate / folder).exists() for folder in ("trainImages", "trainMasks", "testImages")):
            return candidate

    raise FileNotFoundError(
        "Could not find trainImages/trainMasks/testImages in the current folder or any parent folder."
    )


def find_image_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def parse_image_id(path: Path) -> str:
    return Path(path).stem


def parse_mask_id(path: Path) -> str:
    stem = Path(path).stem
    return stem.removesuffix("_mask")


def build_file_index(paths: list[Path], id_parser) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in paths:
        image_id = id_parser(path)
        if image_id in index:
            duplicates.append(image_id)
        index[image_id] = path

    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate image IDs found: {duplicate_preview}")

    return index


def load_dataset(repo_root: Path) -> tuple[Path, pd.DataFrame, dict[str, Path], dict[str, Path], dict[str, Path]]:
    repo_root = Path(repo_root).resolve()
    data_root = find_data_root(repo_root)

    train_csv = repo_root / "trainSet.csv"
    if not train_csv.exists():
        train_csv = data_root / "trainSet.csv"
    if not train_csv.exists():
        raise FileNotFoundError("Could not find trainSet.csv in the repo root or data root.")

    train_df = pd.read_csv(train_csv)
    train_df["imageID"] = train_df["imageID"].astype(str)

    train_image_paths = find_image_files(data_root / "trainImages")
    train_mask_paths = find_image_files(data_root / "trainMasks")
    test_image_paths = find_image_files(data_root / "testImages")

    train_images = build_file_index(train_image_paths, parse_image_id)
    train_masks = build_file_index(train_mask_paths, parse_mask_id)
    test_images = build_file_index(test_image_paths, parse_image_id)

    return data_root, train_df, train_images, train_masks, test_images


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def load_binary_mask(path: Path) -> np.ndarray:
    mask = load_grayscale(path)
    return (mask > 0).astype(np.uint8)


def save_binary_png(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_mask = ((mask > 0).astype(np.uint8) * 255)
    ok = cv2.imwrite(str(path), png_mask)
    if not ok:
        raise OSError(f"Failed to write mask: {path}")

