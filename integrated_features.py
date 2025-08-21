#!/usr/bin/env python3
"""
Integrate deep and hand-crafted features for nuclei analysis.

This script merges deep features (from ResNet) and hand-crafted features,
producing fully integrated, PCA-reduced, variance-selected, and gated features.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def integrate_features(deep_features_path, hand_crafted_path, outdir="outputs"):
    # Load datasets
    df_deep = pd.read_csv(deep_features_path)
    df_hand = pd.read_csv(hand_crafted_path)

    # Merge on nucleus_id
    df_merged = pd.merge(df_deep, df_hand, on='nucleus_id', how='inner')

    # Print columns for debugging
    print("Columns in df_deep:", df_deep.columns.tolist())
    print("Columns in df_hand:", df_hand.columns.tolist())
    print("Columns in df_merged:", df_merged.columns.tolist())

    # Extract feature columns (excluding nucleus_id and basic metrics)
    deep_cols = [col for col in df_deep.columns if col.startswith('feat_')]
    hand_cols = [col for col in df_hand.columns if col not in ['nucleus_id', 'x', 'y', 'area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity', 'eccentricity', 'extent', 'feret_diameter_max', 'roundness']]
    all_cols = deep_cols + hand_cols

    # Verify that we have feature columns
    if not all_cols:
        raise ValueError("No feature columns found for integration.")

    # Fully integrated features: concatenate all features
    full_features = df_merged[all_cols].values
    base_cols = ['nucleus_id', 'x_x', 'y_x', 'area_px', 'equivalent_diameter_px', 'solidity_x', 'eccentricity_x']
    available_cols = [col for col in base_cols if col in df_merged.columns]
    if not available_cols:
        raise ValueError("Required columns for df_full not found in merged DataFrame.")
    
    df_full = df_merged[available_cols].copy()
    df_full.columns = [col.replace('_x', '') if col.endswith('_x') else col for col in df_full.columns]
    integrated_features_df = pd.DataFrame(
        {f'integrated_feat_{i}': full_features[:, i] for i in range(full_features.shape[1])},
        index=df_full.index
    )
    df_full = pd.concat([df_full, integrated_features_df], axis=1)

    # Partially integrated features: apply PCA
    n_components = min(32, full_features.shape[1])  # Ensure n_components <= number of features
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(full_features)
    df_pca = df_merged[available_cols].copy()
    df_pca.columns = [col.replace('_x', '') if col.endswith('_x') else col for col in df_pca.columns]
    pca_features_df = pd.DataFrame(
        {f'pca_feat_{i}': pca_features[:, i] for i in range(pca_features.shape[1])},
        index=df_pca.index
    )
    df_pca = pd.concat([df_pca, pca_features_df], axis=1)

    # Variance-based feature selection: select top features by variance
    var_selector = VarianceThreshold()
    var_selector.fit(full_features)
    variances = var_selector.variances_
    n_var_features = min(32, len(all_cols))
    top_indices = np.argsort(variances)[::-1][:n_var_features]
    var_features = full_features[:, top_indices]
    selected_cols = [all_cols[i] for i in top_indices]
    df_var = df_merged[available_cols].copy()
    df_var.columns = [col.replace('_x', '') if col.endswith('_x') else col for col in df_var.columns]
    var_features_df = pd.DataFrame(
        {f'var_feat_{i}': var_features[:, i] for i in range(var_features.shape[1])},
        index=df_var.index
    )
    df_var = pd.concat([df_var, var_features_df], axis=1)
    print(f"Selected {n_var_features} features with highest variance: {selected_cols}")

    # Gating mechanism: compute weights based on variance and apply to features
    variance_weights = 1 / (1 + np.exp(-variances / (variances.std() + 1e-10)))  # Sigmoid normalization
    gated_features = full_features * variance_weights
    df_gated = df_merged[available_cols].copy()
    df_gated.columns = [col.replace('_x', '') if col.endswith('_x') else col for col in df_gated.columns]
    gated_features_df = pd.DataFrame(
        {f'gated_feat_{i}': gated_features[:, i] for i in range(gated_features.shape[1])},
        index=df_gated.index
    )
    df_gated = pd.concat([df_gated, gated_features_df], axis=1)
    print(f"Applied gating weights based on variance: {variance_weights[:10]}... (first 10 shown)")

    # Save outputs
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(outdir / "fully_integrated_features.csv", index=False)
    df_pca.to_csv(outdir / "pca_integrated_features.csv", index=False)
    df_var.to_csv(outdir / "variance_integrated_features.csv", index=False)
    df_gated.to_csv(outdir / "gated_integrated_features.csv", index=False)
    print(f"[OK] Created fully integrated features for {len(df_full)} nuclei.")
    print(f"[OK] Wrote: {outdir/'fully_integrated_features.csv'}")
    print(f"[OK] Created PCA integrated features for {len(df_pca)} nuclei.")
    print(f"[OK] Wrote: {outdir/'pca_integrated_features.csv'}")
    print(f"[OK] Created variance-selected integrated features for {len(df_var)} nuclei.")
    print(f"[OK] Wrote: {outdir/'variance_integrated_features.csv'}")
    print(f"[OK] Created gated integrated features for {len(df_gated)} nuclei.")
    print(f"[OK] Wrote: {outdir/'gated_integrated_features.csv'}")

def main():
    parser = argparse.ArgumentParser(description='Integrate deep and hand-crafted features for nuclei analysis')
    parser.add_argument('--deep_features', required=True, help='Path to nuclei_features.csv')
    parser.add_argument('--hand_crafted', required=True, help='Path to handcrafted_features.csv')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading deep features: {args.deep_features}")
    print(f"Loading hand-crafted features: {args.hand_crafted}")
    
    integrate_features(args.deep_features, args.hand_crafted, args.outdir)

if __name__ == "__main__":
    main()