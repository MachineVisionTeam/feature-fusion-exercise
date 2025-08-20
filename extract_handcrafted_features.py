#!/usr/bin/env python3
"""
Extract handcrafted features from nuclei in H&E images.

This script takes an H&E image and nuclei centroids CSV, then extracts
~32 handcrafted features (shape, texture, color, intensity) for each nucleus.

Usage:
  python extract_handcrafted_features.py --image Easy1.png --centroids outputs/nuclei_centroids.csv --outdir outputs
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage import io, color, filters, morphology, measure, feature, exposure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


def load_image_and_centroids(image_path, centroids_path):
    """Load the H&E image and nuclei centroids."""
    # Load image
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]  # Remove alpha channel
    if img.dtype != np.uint8:
        img = (255 * (img.astype(np.float32) / max(1, img.max()))).astype(np.uint8)
    
    # Load centroids
    centroids_df = pd.read_csv(centroids_path)
    
    return img, centroids_df


def create_nucleus_mask(img, centroid_x, centroid_y, radius=30):
    """Create a circular mask around a nucleus centroid."""
    H, W = img.shape[:2]
    y, x = np.ogrid[:H, :W]
    
    # Create circular mask
    mask = (x - centroid_x) ** 2 + (y - centroid_y) ** 2 <= radius ** 2
    
    # Ensure mask is within image bounds
    mask = mask & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    
    return mask


def extract_shape_features(mask):
    """Extract shape-based features from a binary mask."""
    # Get region properties
    regions = measure.regionprops(mask.astype(int))
    if not regions:
        return {}
    
    r = regions[0]
    
    features = {
        'area': r.area,
        'perimeter': r.perimeter,
        'circularity': (4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0,
        'aspect_ratio': r.major_axis_length / r.minor_axis_length if r.minor_axis_length > 0 else 0,
        'solidity': r.solidity,
        'eccentricity': r.eccentricity,
        'extent': r.extent,
        'feret_diameter_max': r.feret_diameter_max,
        'roundness': (4 * r.area) / (np.pi * r.major_axis_length ** 2) if r.major_axis_length > 0 else 0,
    }
    
    return features


def extract_texture_features(img_gray, mask):
    """Extract texture features using Haralick and LBP."""
    # Apply mask to get nucleus region
    masked_img = img_gray.copy()
    masked_img[~mask] = 0
    
    # Only process if we have enough pixels
    if np.sum(mask) < 100:
        return {
            'haralick_contrast': 0, 'haralick_homogeneity': 0, 'haralick_energy': 0,
            'haralick_correlation': 0, 'lbp_uniformity': 0, 'lbp_entropy': 0
        }
    
    features = {}
    
    # Haralick texture features
    try:
        # Calculate GLCM
        glcm = graycomatrix(masked_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract properties
        features['haralick_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['haralick_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        features['haralick_energy'] = graycoprops(glcm, 'energy')[0, 0]
        features['haralick_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    except:
        features.update({
            'haralick_contrast': 0, 'haralick_homogeneity': 0, 
            'haralick_energy': 0, 'haralick_correlation': 0
        })
    
    # Local Binary Pattern features
    try:
        lbp = local_binary_pattern(masked_img, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        
        features['lbp_uniformity'] = np.sum(lbp_hist ** 2)  # Uniformity
        features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))  # Entropy
    except:
        features.update({'lbp_uniformity': 0, 'lbp_entropy': 0})
    
    return features


def extract_color_features(img, mask):
    """Extract color-based features from RGB image."""
    # Apply mask to get nucleus region
    masked_img = img.copy()
    masked_img[~mask] = 0
    
    # Get RGB channels
    r_channel = masked_img[:, :, 0]
    g_channel = masked_img[:, :, 1]
    b_channel = masked_img[:, :, 2]
    
    # Calculate statistics for each channel
    features = {}
    
    for channel_name, channel_data in [('r', r_channel), ('g', g_channel), ('b', b_channel)]:
        valid_pixels = channel_data[mask]
        if len(valid_pixels) > 0:
            features[f'{channel_name}_mean'] = np.mean(valid_pixels)
            features[f'{channel_name}_std'] = np.std(valid_pixels)
            features[f'{channel_name}_median'] = np.median(valid_pixels)
            features[f'{channel_name}_skewness'] = skew(valid_pixels)
            features[f'{channel_name}_kurtosis'] = kurtosis(valid_pixels)
        else:
            features.update({
                f'{channel_name}_mean': 0, f'{channel_name}_std': 0, f'{channel_name}_median': 0,
                f'{channel_name}_skewness': 0, f'{channel_name}_kurtosis': 0
            })
    
    # Color ratios and differences
    if np.sum(mask) > 0:
        r_mean = features['r_mean']
        g_mean = features['g_mean']
        b_mean = features['b_mean']
        
        features['rg_ratio'] = r_mean / (g_mean + 1e-10)
        features['rb_ratio'] = r_mean / (b_mean + 1e-10)
        features['gb_ratio'] = g_mean / (b_mean + 1e-10)
        features['rg_diff'] = r_mean - g_mean
        features['rb_diff'] = r_mean - b_mean
        features['gb_diff'] = g_mean - b_mean
    else:
        features.update({
            'rg_ratio': 0, 'rb_ratio': 0, 'gb_ratio': 0,
            'rg_diff': 0, 'rb_diff': 0, 'gb_diff': 0
        })
    
    return features


def extract_intensity_features(img_gray, mask):
    """Extract intensity-based features from grayscale image."""
    # Apply mask to get nucleus region
    masked_img = img_gray.copy()
    masked_img[~mask] = 0
    
    valid_pixels = masked_img[mask]
    
    if len(valid_pixels) > 0:
        features = {
            'intensity_mean': np.mean(valid_pixels),
            'intensity_std': np.std(valid_pixels),
            'intensity_median': np.median(valid_pixels),
            'intensity_min': np.min(valid_pixels),
            'intensity_max': np.max(valid_pixels),
            'intensity_range': np.max(valid_pixels) - np.min(valid_pixels),
            'intensity_skewness': skew(valid_pixels),
            'intensity_kurtosis': kurtosis(valid_pixels),
        }
    else:
        features = {
            'intensity_mean': 0, 'intensity_std': 0, 'intensity_median': 0,
            'intensity_min': 0, 'intensity_max': 0, 'intensity_range': 0,
            'intensity_skewness': 0, 'intensity_kurtosis': 0
        }
    
    return features


def extract_handcrafted_features(img, centroids_df, radius=30):
    """Extract handcrafted features for each nucleus."""
    # Convert to grayscale for some features
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    features_list = []
    
    for idx, row in centroids_df.iterrows():
        nucleus_id = row['nucleus_id']
        centroid_x = row['x']
        centroid_y = row['y']
        
        # Create mask for this nucleus
        mask = create_nucleus_mask(img, centroid_x, centroid_y, radius)
        
        # Extract features
        shape_feats = extract_shape_features(mask)
        texture_feats = extract_texture_features(img_gray, mask)
        color_feats = extract_color_features(img, mask)
        intensity_feats = extract_intensity_features(img_gray, mask)
        
        # Combine all features
        all_features = {
            'nucleus_id': nucleus_id,
            'x': centroid_x,
            'y': centroid_y,
            **shape_feats,
            **texture_feats,
            **color_feats,
            **intensity_feats
        }
        
        features_list.append(all_features)
    
    return pd.DataFrame(features_list)


def main():
    parser = argparse.ArgumentParser(description='Extract handcrafted features from nuclei in H&E images')
    parser.add_argument('--image', required=True, help='Path to input H&E image')
    parser.add_argument('--centroids', required=True, help='Path to nuclei centroids CSV')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    parser.add_argument('--radius', type=int, default=30, help='Radius around centroid to analyze (pixels)')
    
    args = parser.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image: {args.image}")
    print(f"Loading centroids: {args.centroids}")
    
    # Load data
    img, centroids_df = load_image_and_centroids(args.image, args.centroids)
    
    print(f"Image shape: {img.shape}")
    print(f"Number of nuclei: {len(centroids_df)}")
    
    # Extract features
    print("Extracting handcrafted features...")
    features_df = extract_handcrafted_features(img, centroids_df, args.radius)
    
    # Save features
    output_path = outdir / "handcrafted_features.csv"
    features_df.to_csv(output_path, index=False)
    
    print(f"Features extracted: {len(features_df.columns) - 3} features per nucleus")  # -3 for id, x, y
    print(f"Output saved to: {output_path}")
    
    # Display feature summary
    print("\nFeature categories:")
    print(f"  Shape features: {len([col for col in features_df.columns if any(x in col for x in ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity', 'eccentricity', 'extent', 'feret', 'roundness'])])}")
    print(f"  Texture features: {len([col for col in features_df.columns if any(x in col for x in ['haralick', 'lbp'])])}")
    print(f"  Color features: {len([col for col in features_df.columns if any(x in col for x in ['r_', 'g_', 'b_', 'rg_', 'rb_', 'gb_'])])}")
    print(f"  Intensity features: {len([col for col in features_df.columns if 'intensity' in col])}")
    
    # Show first few rows
    print(f"\nFirst few features:")
    print(features_df.head())


if __name__ == "__main__":
    main()
