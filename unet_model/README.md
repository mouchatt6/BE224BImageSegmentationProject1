# First U-Net Baseline

This folder contains the first neural segmentation pipeline for the needle CT project.

## Install Training Dependencies

```bash
.venv/bin/python -m pip install -r requirements-unet.txt
```

## Smoke Test Training

Run a quick two-batch pass before a longer training run:

```bash
.venv/bin/python -m unet_model.train_unet \
  --epochs 1 \
  --batch-size 2 \
  --base-channels 16 \
  --limit-train-batches 2 \
  --limit-valid-batches 2
```

## First Real Training Run

```bash
.venv/bin/python -m unet_model.train_unet \
  --epochs 30 \
  --batch-size 4 \
  --base-channels 32 \
  --lr 0.001
```

Outputs:

```text
outputs/unet_model/best_unet.pt
outputs/unet_model/training_history.csv
```

The model selection score is the balanced `alpha = 0.50` validation score, while the history also records `alpha = 0.25` and `alpha = 0.75`.

## Export Test Masks

```bash
.venv/bin/python -m unet_model.predict_unet \
  --checkpoint outputs/unet_model/best_unet.pt \
  --output-dir outputs/unet_model/test_masks \
  --threshold 0.5
```

Then create a Kaggle CSV with `Process_Images.py` or by passing `outputs/unet_model/test_masks` to `processImages()`.

## Threshold Sweep

Run this after training and before exporting test masks:

```bash
python -m unet_model.sweep_thresholds \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --output-csv outputs/unet_model/threshold_sweep.csv
```

The CSV is ranked by `score_alpha_050` and also reports `score_alpha_025` and `score_alpha_075`.
Use the top validation configurations to choose one or two daily Kaggle submissions.

Example export using a selected configuration:

```bash
python -m unet_model.predict_unet \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unet_model/best_unet.pt \
  --output-dir outputs/unet_model/test_masks_t030 \
  --threshold 0.30 \
  --min-area 3 \
  --close-kernel-size 3 \
  --dilation-iterations 1
```
