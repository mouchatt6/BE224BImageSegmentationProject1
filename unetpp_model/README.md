# U-Net++ Model

This folder mirrors the baseline `unet_model` workflow with a U-Net++ architecture.
It reuses the same dataset loader, loss functions, validation metrics, and post-processing.

## Train

For one Kaggle run, start with `base_channels=32`; U-Net++ is heavier than the baseline U-Net.
If Kaggle memory allows it, try `base_channels=48`.

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

## Sweep Threshold And Post-Processing

```bash
python -m unetpp_model.sweep_thresholds \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unetpp_model/best_unetpp.pt \
  --output-csv outputs/unetpp_model/threshold_sweep.csv
```

Optional focused sweeps after the first threshold sweep:

```bash
python -m unetpp_model.sweep_min_area \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unetpp_model/best_unetpp.pt \
  --threshold 0.68 \
  --close-kernel-size 7 \
  --dilation-iterations 0 \
  --output-csv outputs/unetpp_model/min_area_sweep.csv
```

```bash
python -m unetpp_model.sweep_close_kernel_size \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER \
  --checkpoint outputs/unetpp_model/best_unetpp.pt \
  --threshold 0.68 \
  --min-area 120 \
  --dilation-iterations 0 \
  --output-csv outputs/unetpp_model/close_kernel_size_sweep.csv
```

## Predict

Replace the threshold, min-area, and closing kernel values with the best validation settings.

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

