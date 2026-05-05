from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(__file__).resolve().parent / "assets"


METRICS = [
    {
        "model": "Classical\nbaseline",
        "log_loss": 0.0590,
        "brier": 0.00366,
        "dice": 0.0341,
        "sensitivity": 0.0333,
        "alpha25": 0.0335,
        "alpha50": 0.0337,
        "alpha75": 0.0339,
        "kaggle": 0.298,
    },
    {
        "model": "U-Net\nraw",
        "log_loss": 0.1003,
        "brier": 0.00992,
        "dice": 0.3069,
        "sensitivity": 0.3069,
        "alpha25": 0.3069,
        "alpha50": 0.3069,
        "alpha75": 0.3069,
        "kaggle": 0.413,
    },
    {
        "model": "U-Net\ntuned",
        "log_loss": 0.1003,
        "brier": 0.00992,
        "dice": 0.4556,
        "sensitivity": 0.4551,
        "alpha25": 0.4553,
        "alpha50": 0.4554,
        "alpha75": 0.4555,
        "kaggle": np.nan,
    },
]


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 60, 40), alpha=0.55) -> np.ndarray:
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)
    active = mask > 0
    if active.any():
        color_arr = np.array(color, dtype=np.float32)
        rgb_float = rgb.astype(np.float32)
        rgb_float[active] = (1.0 - alpha) * rgb_float[active] + alpha * color_arr
        rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)
    return rgb


def make_mask_comparison() -> None:
    test_dir = REPO_ROOT.parent / "testImages" / "testImages"
    percentile_dir = REPO_ROOT / "outputs" / "baseline_model" / "percentile" / "test_masks"
    hough_dir = REPO_ROOT / "outputs" / "baseline_model" / "hough" / "test_masks"
    unet_dir = REPO_ROOT / "outputs" / "unet_model" / "test_masks_t020"

    image_ids = ["1443", "11238", "12077", "20288"]
    columns = ["CT image", "Baseline percentile", "Baseline Hough", "U-Net mask"]

    fig, axes = plt.subplots(len(image_ids), len(columns), figsize=(7.2, 6.8), dpi=240)
    for row_idx, image_id in enumerate(image_ids):
        image = read_gray(test_dir / f"{image_id}.jpg")
        masks = [
            None,
            read_gray(percentile_dir / f"{image_id}_mask.png"),
            read_gray(hough_dir / f"{image_id}_mask.png"),
            read_gray(unet_dir / f"{image_id}_mask.png"),
        ]

        for col_idx, (title, mask) in enumerate(zip(columns, masks)):
            ax = axes[row_idx, col_idx]
            if mask is None:
                shown = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                ax.imshow(shown, cmap="gray")
            else:
                ax.imshow(overlay_mask(image, mask))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"ID {image_id}", fontsize=8)

    fig.suptitle("Qualitative comparison of generated test masks", fontsize=11, y=0.995)
    fig.tight_layout(pad=0.6)
    fig.savefig(ASSET_DIR / "mask_comparison.png", bbox_inches="tight")
    fig.savefig(ASSET_DIR / "mask_grid_4x4.png", bbox_inches="tight")
    plt.close(fig)


def make_mask_comparison_compact() -> None:
    test_dir = REPO_ROOT.parent / "testImages" / "testImages"
    percentile_dir = REPO_ROOT / "outputs" / "baseline_model" / "percentile" / "test_masks"
    hough_dir = REPO_ROOT / "outputs" / "baseline_model" / "hough" / "test_masks"
    unet_dir = REPO_ROOT / "outputs" / "unet_model" / "test_masks_t020"

    image_id = "1443"
    image = read_gray(test_dir / f"{image_id}.jpg")
    panels = [
        ("CT image", None),
        ("Percentile", read_gray(percentile_dir / f"{image_id}_mask.png")),
        ("Hough", read_gray(hough_dir / f"{image_id}_mask.png")),
        ("U-Net", read_gray(unet_dir / f"{image_id}_mask.png")),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.0), dpi=240)
    for ax, (title, mask) in zip(axes, panels):
        if mask is None:
            shown = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            ax.imshow(shown, cmap="gray")
        else:
            ax.imshow(overlay_mask(image, mask))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.3, w_pad=0.25)
    fig.savefig(ASSET_DIR / "mask_comparison_compact.png", bbox_inches="tight")
    plt.close(fig)


def make_metric_plot() -> None:
    labels = [item["model"] for item in METRICS]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=220)
    for offset, key, label in [
        (-1.5 * width, "dice", "Dice"),
        (-0.5 * width, "sensitivity", "Sensitivity"),
        (0.5 * width, "alpha25", r"$\alpha=0.25$"),
        (1.5 * width, "alpha50", r"$\alpha=0.50$"),
        (2.5 * width, "alpha75", r"$\alpha=0.75$"),
    ]:
        ax.bar(x + offset, [item[key] for item in METRICS], width=width, label=label)

    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 0.52)
    ax.legend(ncol=3, fontsize=7, frameon=True, loc="upper left")
    ax.set_title("Overlap and hidden-alpha validation metrics")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "metric_bars.png", bbox_inches="tight")
    plt.close(fig)


def make_calibration_plot() -> None:
    labels = [item["model"] for item in METRICS]
    x = np.arange(len(labels))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(7.2, 2.8), dpi=220)
    ax1.bar(x - width / 2, [item["log_loss"] for item in METRICS], width=width, label="Log-loss")
    ax1.set_ylabel("Log-loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 0.12)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        [item["brier"] for item in METRICS],
        width=width,
        color="#55a868",
        label="Brier calibration",
        alpha=0.9,
    )
    ax2.set_ylabel("Brier score")
    ax2.set_ylim(0, 0.012)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper left")
    ax1.set_title("Pixel-level log-loss and calibration")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "calibration_bars.png", bbox_inches="tight")
    plt.close(fig)


def make_results_summary_plot() -> None:
    labels = [item["model"].replace("\n", " ") for item in METRICS]
    x = np.arange(len(labels))
    width = 0.15

    fig, (ax_score, ax_cal) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.1),
        dpi=240,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    for offset, key, label in [
        (-2 * width, "dice", "Dice"),
        (-1 * width, "sensitivity", "Sensitivity"),
        (0 * width, "alpha25", r"$\alpha=0.25$"),
        (1 * width, "alpha50", r"$\alpha=0.50$"),
        (2 * width, "alpha75", r"$\alpha=0.75$"),
    ]:
        ax_score.bar(x + offset, [item[key] for item in METRICS], width=width, label=label)

    ax_score.set_xticks(x)
    ax_score.set_xticklabels(labels, fontsize=9)
    ax_score.set_ylabel("Validation score", fontsize=10)
    ax_score.set_ylim(0, 0.52)
    ax_score.set_title("Overlap and hidden-alpha metrics", fontsize=11)
    ax_score.legend(ncol=3, fontsize=8, frameon=True, loc="upper left")

    width_cal = 0.28
    ax_cal.bar(x - width_cal / 2, [item["log_loss"] for item in METRICS], width=width_cal, label="Log-loss")
    ax_cal.set_ylabel("Log-loss", fontsize=10)
    ax_cal.set_xticks(x)
    ax_cal.set_xticklabels(labels, fontsize=9)
    ax_cal.set_ylim(0, 0.12)
    ax_cal.set_title("Pixel-level log-loss and calibration", fontsize=11)

    ax_brier = ax_cal.twinx()
    ax_brier.bar(
        x + width_cal / 2,
        [item["brier"] for item in METRICS],
        width=width_cal,
        color="#55a868",
        label="Brier calibration",
        alpha=0.9,
    )
    ax_brier.set_ylabel("Brier score", fontsize=10)
    ax_brier.set_ylim(0, 0.012)

    handles1, labels1 = ax_cal.get_legend_handles_labels()
    handles2, labels2 = ax_brier.get_legend_handles_labels()
    ax_cal.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.tight_layout(h_pad=1.15)
    fig.savefig(ASSET_DIR / "results_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    make_mask_comparison()
    make_mask_comparison_compact()
    make_metric_plot()
    make_calibration_plot()
    make_results_summary_plot()


if __name__ == "__main__":
    main()
