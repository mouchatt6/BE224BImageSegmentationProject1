# BE224B Image Segmentation Project 1

This project builds binary segmentation models for CT images where the foreground class is a thin needle. The output masks are converted into Kaggle submissions with `Process_Images.py`.

## Project Structure

```text
BE224BImageSegmentationProject1/
├── baseline_model/
│   ├── algorithms.py              # Classical thresholding and morphology baselines
│   ├── data_io.py                 # Dataset discovery, image loading, mask export helpers
│   ├── metrics.py                 # Dice, sensitivity, and hidden-alpha composite scores
│   └── run_baselines.py           # Baseline CLI
├── unet_model/
│   ├── dataset.py                 # PyTorch dataset and augmentation
│   ├── losses.py                  # BCE, Dice, Tversky, Focal, and combined losses
│   ├── model.py                   # Baseline U-Net architecture
│   ├── postprocess.py             # Thresholding, connected components, closing, dilation
│   ├── predict_unet.py            # U-Net test-mask export
│   ├── sweep_thresholds.py        # Joint threshold/post-processing validation sweep
│   ├── sweep_min_area.py          # Focused min-area validation sweep
│   ├── sweep_close_kernel_size.py # Focused closing-kernel validation sweep
│   └── train_unet.py              # U-Net training CLI
├── unetpp_model/
│   ├── model.py                   # U-Net++ architecture
│   ├── predict_unetpp.py          # U-Net++ test-mask export
│   ├── sweep_thresholds.py        # U-Net++ threshold/post-processing sweep
│   ├── sweep_min_area.py          # U-Net++ focused min-area sweep
│   ├── sweep_close_kernel_size.py # U-Net++ focused closing-kernel sweep
│   └── train_unetpp.py            # U-Net++ training CLI
├── eda_and_visual_checks.ipynb    # Dataset audit, visualization, and sanity checks
├── Process_Images.py              # Converts exported masks into submission.csv
├── NEEDLE_SEGMENTATION_ACTION_PLAN.md
├── requirements.txt
├── requirements-eda.txt
├── requirements-unet.txt
└── trainSet.csv
```

Expected data layout:

```text
data_root/
├── trainImages/
│   └── trainImages/
│       └── {imageID}.jpg
├── trainMasks/
│   └── trainMasks/
│       └── {imageID}_mask.png
├── testImages/
│   └── testImages/
│       └── {imageID}.jpg
└── trainSet.csv
```

## Motivation And Project Exploration

The task is difficult because the needle occupies a tiny fraction of each `512 x 512` CT image. Most pixels are background, and many training examples have empty masks. A naive model can look stable during training while still missing the needle or producing small false-positive components that hurt Dice and sensitivity.

The project explored a staged modeling path:

1. Classical baseline models: percentile thresholding, Otsu thresholding, Hough-line style ideas, connected-component filtering, and morphology. These gave an interpretable starting point and exposed how noisy high-intensity structures can look like needles.
2. Baseline U-Net: a full-resolution neural segmentation model using PyTorch, geometric/intensity augmentation, Dice-style validation metrics, and validation-driven threshold tuning.
3. Improved U-Net workflow: stronger loss options, including BCE + Dice + Tversky, plus focused post-processing sweeps for threshold, minimum component area, closing kernel size, and dilation.
4. U-Net++: a denser skip-connection model added after the baseline U-Net plateaued. U-Net++ is intended to improve thin-structure localization and reconnect fragmented predictions through nested decoder pathways.

The scoring workflow reports Dice, sensitivity, and composite scores for multiple possible hidden-alpha settings:

```text
score = alpha * Dice + (1 - alpha) * Sensitivity
```

The code ranks primarily by `alpha = 0.50`, but also reports `alpha = 0.25` and `alpha = 0.75` because the exact Kaggle weighting may not be fully transparent.

## Environment Setup And Initialization

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the consolidated dependencies:

```bash
pip install -r requirements.txt
```

For Kaggle notebooks, install from the repo root:

```bash
pip install -r requirements.txt
```

If using Kaggle's preinstalled PyTorch environment, this may already satisfy `torch` and `torchvision`; installing the file is still the simplest reproducible setup.

Check the baseline environment:

```bash
python -m baseline_model.check_environment
```

## Dependencies

Core dependencies are listed in `requirements.txt`:

```text
numpy==1.26.4
pandas==2.2.2
pillow==10.4.0
matplotlib==3.9.2
opencv-python==4.10.0.84
scikit-learn==1.5.1
ipykernel==6.29.5
torch>=2.4.0
torchvision>=0.19.0
tqdm>=4.66.0
```

The older split files are still present:

- `requirements-eda.txt`: notebook and visualization dependencies.
- `requirements-unet.txt`: neural model dependencies.

## Code Usage Examples

### Run Classical Baselines

```bash
python -m baseline_model.run_baselines \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --method percentile \
  --percentile 99.5 \
  --min-area 5 \
  --close-kernel-size 3 \
  --dilation-iterations 0
```

### Train Baseline U-Net

```bash
python -m unet_model.train_unet \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --epochs 75 \
  --batch-size 4 \
  --base-channels 64 \
  --lr 0.001 \
  --loss bce_dice_tversky \
  --output-dir outputs/unet_model
```

### Sweep U-Net Post-Processing

```bash
python -m unet_model.sweep_thresholds \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --output-csv outputs/unet_model/threshold_sweep.csv
```

Focused sweeps around a promising configuration:

```bash
python -m unet_model.sweep_min_area \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --threshold 0.68 \
  --close-kernel-size 7 \
  --dilation-iterations 0 \
  --output-csv outputs/unet_model/min_area_sweep_t068.csv
```

```bash
python -m unet_model.sweep_close_kernel_size \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --threshold 0.68 \
  --min-area 120 \
  --dilation-iterations 0 \
  --output-csv outputs/unet_model/close_kernel_sweep_t068.csv
```

### Export U-Net Test Masks

Replace the post-processing values with the best validation sweep settings.

```bash
python -m unet_model.predict_unet \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --output-dir outputs/unet_model/test_masks \
  --threshold 0.68 \
  --min-area 120 \
  --close-kernel-size 7 \
  --dilation-iterations 0
```

### Train U-Net++

U-Net++ is heavier than U-Net. Start with `base_channels=32` for a safer Kaggle run. If memory allows, try `base_channels=48`.

```bash
python -m unetpp_model.train_unetpp \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --epochs 75 \
  --batch-size 4 \
  --base-channels 32 \
  --lr 0.0005 \
  --loss bce_dice_tversky \
  --output-dir outputs/unetpp_model
```

Lower-memory fallback:

```bash
python -m unetpp_model.train_unetpp \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --epochs 75 \
  --batch-size 2 \
  --base-channels 32 \
  --lr 0.0005 \
  --loss bce_dice_tversky \
  --output-dir outputs/unetpp_model
```

### Sweep And Export U-Net++

```bash
python -m unetpp_model.sweep_thresholds \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unetpp_model/best_unetpp.pt \
  --output-csv outputs/unetpp_model/threshold_sweep.csv
```

```bash
python -m unetpp_model.predict_unetpp \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unetpp_model/best_unetpp.pt \
  --output-dir outputs/unetpp_model/test_masks \
  --threshold 0.68 \
  --min-area 120 \
  --close-kernel-size 7 \
  --dilation-iterations 0
```

### Create Kaggle Submission CSV

After exporting binary PNG masks, use `Process_Images.py` or call `processImages()` with the exported mask directory. The exported masks should be `512 x 512`, binary, and named as `{imageID}_mask.png`.

## Future Work Exploration Ideas

- Run 3-fold or 5-fold cross-validation and ensemble probability maps before thresholding.
- Add an empty-image gate using maximum probability, total probability mass, or largest component score.
- Try patch-based or crop-based training centered around likely needle regions while preserving full-resolution inference.
- Add attention gates or a pretrained encoder if compute and package constraints allow.
- Tune loss variants more systematically: `bce_dice_tversky`, `dice_tversky`, and `focal_tversky`.
- Use test-time augmentation with horizontal/vertical flips and average probability maps.
- Save validation overlays for worst Dice cases to distinguish false positives from missed needles.
- Compare post-processing policies: largest component only, top-k elongated components, and line-fit scoring.
- Calibrate threshold selection against all reported alpha settings, not only `score_alpha_050`.
