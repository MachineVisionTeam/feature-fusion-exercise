# feature-fusion-exercise

Detect nuclei from an H&E-style histology image, export **centroids** \((x, y)\), and compute **last fully-connected (penultimate) features** for each nucleus using a ResNet-18 backbone.

---

## ✨ What this does

1. **Segments nuclei** (dark blue/purple) from RGB using HED color deconvolution (hematoxylin channel).  
2. **Improves recall** on faint/clustered nuclei via a softened threshold and light watershed split.  
3. **Computes centroids** for each nucleus: `x` = column, `y` = row (pixel coordinates, origin at top-left).  
4. **Extracts 512-D features** from the **penultimate layer** (after global average pooling, right before the final FC) of **ResNet-18** for each cropped nucleus.

Outputs are saved to the `outputs/` folder:
- `overlay.png` – QA visualization (centroids + boundaries)  
- `nuclei_centroids.csv` – one row per nucleus with `(x, y)` and shape stats  
- `nuclei_features.csv` – same rows plus `feat_0 … feat_511` (penultimate features)

---

## Requirements
```bash
pip install numpy scipy pandas pillow scikit-image matplotlib torch torchvision
```

## Usage

```bash
python nuclei_detect_and_features.py --image Easy1.png --outdir outputs
```


# Handcrafted Features Extraction for Nuclei

This script extracts **44 handcrafted features** from nuclei in H\&E histology images based on pre-computed centroids.

## What This Does

The script takes an H\&E image and a CSV file containing nuclei centroids, then extracts handcrafted features for each nucleus:

* **Shape Features (9)**: Area, perimeter, circularity, aspect ratio, solidity, eccentricity, extent, Feret diameter, roundness
* **Texture Features (6)**: Haralick (contrast, homogeneity, energy, correlation) and LBP (uniformity, entropy)
* **Color Features (22)**: RGB channel statistics (mean, std, median, skewness, kurtosis) and color ratios/differences
* **Intensity Features (8)**: Grayscale intensity statistics (mean, std, median, min, max, range, skewness, kurtosis)

##  Requirements

```bash
pip install -r requirements_handcrafted.txt
```

Or install manually:

```bash
pip install numpy pandas Pillow opencv-python scikit-image scipy matplotlib
```

##  Usage

### Basic Usage

```bash
python extract_handcrafted_features.py --image Easy1.png --centroids outputs/nuclei_centroids.csv --outdir outputs
```

### With Parameters

```bash
python extract_handcrafted_features.py \
    --image Easy1.png \
    --centroids outputs/nuclei_centroids.csv \
    --outdir outputs \
    --radius 30
```

### Parameters

* `--image`: Path to input H\&E image (PNG, JPG, etc.)
* `--centroids`: Path to CSV file with nuclei centroids (columns: nucleus\_id, x, y)
* `--outdir`: Output directory (default: "outputs")
* `--radius`: Radius around centroid to analyze in pixels (default: 30)

##  Output

The script generates `outputs/handcrafted_features.csv` with one row per nucleus containing centroid coordinates and 44 handcrafted features.

##  Testing

Run the test script to verify feature extraction:

```bash
python test_features.py
```

---

##  Related Files

* `extract_handcrafted_features.py`: Main feature extraction script
* `requirements_handcrafted.txt`: Python dependencies
* `outputs/handcrafted_features.csv`: Generated features

---
