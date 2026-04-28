from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baseline_model.data_io import load_binary_mask, load_dataset, load_grayscale


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_training_index(repo_root: Path, data_root: Path | None = None):
    return load_dataset(repo_root, data_root=data_root)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    mean = float(image.mean())
    std = float(image.std())
    if std > 1e-6:
        image = (image - mean) / std
    else:
        image = image - mean
    return image.astype(np.float32)


def random_augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    if rng.random() < 0.35:
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-0.08, 0.08))
        image = np.clip(image * alpha + beta, 0.0, 1.0)

    if rng.random() < 0.20:
        noise = rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class NeedleDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_paths: dict[str, Path],
        mask_paths: dict[str, Path] | None = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.df["imageID"] = self.df["imageID"].astype(str)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        image_id = str(row["imageID"])
        image = load_grayscale(self.image_paths[image_id])

        if self.mask_paths is None:
            mask = None
        else:
            mask = load_binary_mask(self.mask_paths[image_id]).astype(np.float32)

        image_float = image.astype(np.float32) / 255.0
        if mask is not None and self.augment:
            rng = np.random.default_rng(self.seed + idx)
            image_float, mask = random_augment(image_float, mask, rng)

        image_norm = normalize_image((image_float * 255.0).astype(np.uint8))
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).float()

        item: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "image_id": image_id,
        }

        if mask is not None:
            item["mask"] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).float()

        return item


def make_test_dataframe(test_images: dict[str, Path]) -> pd.DataFrame:
    image_ids = sorted(test_images, key=lambda value: int(value) if value.isdigit() else value)
    return pd.DataFrame({"imageID": image_ids})

