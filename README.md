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

