#!/usr/bin/env python3
"""
Nuclei detection (with faint/cluster fix) + centroids + 512-D penultimate features.

Usage:
  python nuclei_detect_and_features.py --image Easy1.png --outdir outputs
  # optional flags:
  #   --device cuda
  #   --delta_h 0.03         # increase if faint nuclei are missed (0.02–0.05)
  #   --min_area 50 --max_area 4000
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from skimage import io, color, filters, morphology, measure, segmentation, exposure
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt


# ----------------- SEGMENTATION -----------------
def segment_nuclei_fixed(
    img_rgb: np.ndarray,
    close_radius: int = 1,
    min_area: int = 50,
    max_area: int = 4000,
    delta_h: float = 0.03,
    watershed_min_distance: int = 7,
):
    """
    HED->H channel -> (Otsu - delta_h) threshold -> cleanup ->
    distance-based watershed to split big/touching blobs -> shape/area filter.
    """
    # drop alpha if present
    if img_rgb.ndim == 3 and img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]

    # HED deconvolution and use H channel (nuclei)
    H = color.rgb2hed(img_rgb)[:, :, 0]
    H = exposure.rescale_intensity(H, in_range="image", out_range=(0, 1))

    # --- (1) lower threshold slightly to catch faint nuclei ---
    t = filters.threshold_otsu(H)
    thr = max(0.0, min(1.0, t - float(delta_h)))
    nuclei = H > thr

    # --- cleanup ---
    if close_radius > 0:
        nuclei = morphology.binary_closing(nuclei, morphology.disk(close_radius))
    nuclei = morphology.remove_small_holes(nuclei, area_threshold=32)
    nuclei = morphology.remove_small_objects(nuclei, min_size=max(25, min_area // 2))

    # --- (2) watershed to split close/large blobs ---
    distance = ndi.distance_transform_edt(nuclei)
    coords = peak_local_max(
        distance,
        footprint=np.ones((2 * watershed_min_distance + 1, 2 * watershed_min_distance + 1)),
        labels=nuclei,
        exclude_border=False,
    )
    markers = np.zeros_like(distance, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers = ndi.label(markers)[0]
    labels_ws = segmentation.watershed(-distance, markers, mask=nuclei)

    # --- area + light shape gate (avoid fibers/noise) ---
    keep = np.zeros_like(labels_ws, dtype=bool)
    for r in measure.regionprops(labels_ws):
        if not (min_area <= r.area <= max_area):
            continue
        if r.solidity < 0.80 or r.eccentricity > 0.98:
            continue
        keep[labels_ws == r.label] = True

    labels = measure.label(keep)
    regions = measure.regionprops(labels)
    return labels, regions


def overlay_with_centroids(img_rgb, labels, regions, dot_radius=5, save_path=None):
    """Overlay red boundaries and green dots at centroids."""
    overlay = img_rgb.copy()
    if overlay.dtype != np.uint8:
        overlay = (255 * np.clip(overlay, 0, 1)).astype(np.uint8)
    edges = segmentation.find_boundaries(labels, mode="outer")
    overlay[edges] = [255, 0, 0]
    for r in regions:
        cy, cx = r.centroid
        cy, cx = int(round(cy)), int(round(cx))
        rr, cc = np.ogrid[:overlay.shape[0], :overlay.shape[1]]
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= dot_radius ** 2
        overlay[mask] = [0, 255, 0]
    if save_path:
        plt.figure(figsize=(10, 6))
        plt.imshow(overlay)
        plt.title("Detected nuclei: green = centroids, red = boundaries")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    return overlay


# ----------------- FEATURES (penultimate) -----------------
class PenultimateExtractor(nn.Module):
    """Return 512-D penultimate vector of ResNet-18 (after GAP, before FC)."""
    def __init__(self, backbone):
        super().__init__()
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,  # -> N x 512 x 1 x 1
        )
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)  # N x 512


def load_resnet18_extractor(device="cpu"):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(weights=None)
        print("[WARN] Using randomly-initialized ResNet-18 (no weights cache).")
    model = PenultimateExtractor(net).to(device)
    model.eval()
    return model


def extract_penultimate_features(img_rgb: np.ndarray, regions, pad=4, device="cpu", batch=64):
    """Crop around each nucleus, resize to 224, and get 512-D features (pre-FC)."""
    H, W, _ = img_rgb.shape
    tfm = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    crops = []
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        y0 = max(y0 - pad, 0); x0 = max(x0 - pad, 0)
        y1 = min(y1 + pad, H); x1 = min(x1 + pad, W)
        crop = Image.fromarray(img_rgb[y0:y1, x0:x1])
        crops.append(tfm(crop))
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    tens = torch.stack(crops, 0).to(device)
    model = load_resnet18_extractor(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            out = model(tens[i:i+batch])  # (B, 512)
            outs.append(out.cpu().numpy())
    return np.concatenate(outs, 0)


# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input RGB image")
    ap.add_argument("--outdir", default="outputs", help="Where to write CSVs and overlay")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--min_area", type=int, default=50)
    ap.add_argument("--max_area", type=int, default=4000)
    ap.add_argument("--delta_h", type=float, default=0.03)
    ap.add_argument("--watershed_min_distance", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    img = io.imread(args.image)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = (255 * (img.astype(np.float32) / max(1, img.max()))).astype(np.uint8)

    labels, regions = segment_nuclei_fixed(
        img,
        close_radius=1,
        min_area=args.min_area,
        max_area=args.max_area,
        delta_h=args.delta_h,
        watershed_min_distance=args.watershed_min_distance,
    )

    # Centroids table
    rows = []
    for i, r in enumerate(regions, start=1):
        cy, cx = r.centroid  # (row, col)
        rows.append({
            "nucleus_id": i,
            "x": float(cx),   # column = x
            "y": float(cy),   # row    = y
            "area_px": int(r.area),
            "equivalent_diameter_px": float(r.equivalent_diameter),
            "solidity": float(r.solidity),
            "eccentricity": float(r.eccentricity),
        })
    df_cent = pd.DataFrame(rows)
    df_cent.to_csv(outdir / "nuclei_centroids.csv", index=False)

    # Overlay
    overlay_with_centroids(img, labels, regions, save_path=outdir / "overlay.png")

    # Features (512-D penultimate)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    feats = extract_penultimate_features(img, regions, device=device)
    n = min(len(rows), feats.shape[0])
    base = pd.DataFrame(rows[:n])
    feat_df = pd.DataFrame({f"feat_{j}": feats[:n, j] for j in range(feats.shape[1])})
    pd.concat([base, feat_df], axis=1).to_csv(outdir / "nuclei_features.csv", index=False)

    print(f"[OK] Nuclei detected: {len(df_cent)}")
    print(f"[OK] Wrote: {outdir/'overlay.png'}")
    print(f"[OK] Wrote: {outdir/'nuclei_centroids.csv'}")
    print(f"[OK] Wrote: {outdir/'nuclei_features.csv'}")


if __name__ == "__main__":
    main()
