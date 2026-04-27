# Classical Baseline Model

This folder contains classical image-processing baselines for the CT needle segmentation project.

## Methods

- `percentile`: high-intensity percentile thresholding plus connected-component cleanup.
- `otsu`: Otsu thresholding plus connected-component cleanup.
- `hough`: Canny edges plus probabilistic Hough-line detection for bright linear structures.

The current primary baselines are `percentile` and `hough`. `otsu` is included as a quick reference point.

## Setup

Use the project EDA environment:

```bash
source .venv/bin/activate
```

or run commands directly with:

```bash
.venv/bin/python
```

## Validate A Baseline

From the repo root:

```bash
.venv/bin/python -m baseline_model.run_baselines --mode validate --method percentile
```

```bash
.venv/bin/python -m baseline_model.run_baselines --mode validate --method hough
```

Outputs are written under:

```text
outputs/baseline_model/{method}/
```

The validation summary reports:

- DICE
- sensitivity
- composite score for `alpha = 0.25`
- composite score for `alpha = 0.50`
- composite score for `alpha = 0.75`

This matches the project evaluation structure:

```text
Score = alpha * DICE + (1 - alpha) * Sensitivity
```

Because `alpha` is hidden, compare methods across all three alpha values.

## Export Test Masks

```bash
.venv/bin/python -m baseline_model.run_baselines --mode predict-test --method percentile
```

The exported PNG masks are binary `0/255` files under:

```text
outputs/baseline_model/percentile/test_masks/
```

These files are for local experimentation and are ignored by Git.

