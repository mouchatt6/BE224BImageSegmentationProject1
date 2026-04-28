from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from .data_io import load_dataset
    from .run_baselines import infer_repo_root
except ImportError:
    from data_io import load_dataset
    from run_baselines import infer_repo_root


def main() -> None:
    repo_root = infer_repo_root()
    data_root, train_df, train_images, train_masks, test_images = load_dataset(repo_root)

    print(f"Repo root:    {repo_root}")
    print(f"Data root:    {data_root}")
    print(f"NumPy:        {np.__version__}")
    print(f"pandas:       {pd.__version__}")
    print(f"OpenCV:       {cv2.__version__}")
    print(f"trainSet rows: {len(train_df)}")
    print(f"Train images: {len(train_images)}")
    print(f"Train masks:  {len(train_masks)}")
    print(f"Test images:  {len(test_images)}")


if __name__ == "__main__":
    main()

