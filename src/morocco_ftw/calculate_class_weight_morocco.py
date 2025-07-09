#This scripts is used for calculate the weighte for three classes
import os
import numpy as np
import rasterio
from collections import Counter
from pathlib import Path

def calculate_class_weights(data_root, country="morocco"):
    """Calculate class weights for 3-class dataset"""
    
    country_path = Path(data_root) / "ftw" / country
    masks_3class_path = country_path / "label_masks" / "semantic_3class"
    
    if not masks_3class_path.exists():
        print(f"Masks not found at {masks_3class_path}")
        return None
    
    print(f"Analyzing masks in: {masks_3class_path}")
    
    class_counts = Counter()
    total_pixels = 0
    
    # Count pixels for each class across all masks
    for mask_file in masks_3class_path.glob("*.tif"):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
            unique, counts = np.unique(mask, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                class_counts[class_id] += count
                total_pixels += count
    
    print(f"\nClass distribution:")
    print(f"Total pixels: {total_pixels:,}")
    
    for class_id in sorted(class_counts.keys()):
        percentage = (class_counts[class_id] / total_pixels) * 100
        print(f"Class {class_id}: {class_counts[class_id]:,} pixels ({percentage:.2f}%)")
    
    # Calculate inverse frequency weights
    num_classes = len(class_counts)
    weights = []
    
    print(f"\nCalculated weights (inverse frequency):")
    for class_id in sorted(class_counts.keys()):
        if class_id == 0:  # Background/ignore class
            weight = 0.0
        else:
            class_freq = class_counts[class_id] / total_pixels
            weight = 1.0 / (num_classes * class_freq)
        
        weights.append(weight)
        print(f"Class {class_id}: {weight:.3f}")
    
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights * num_classes / weights.sum()
    
    print(f"\nNormalized weights: {weights.tolist()}")
    print(f"Config format: class_weights: {[round(w, 3) for w in weights]}")
    
    return weights

if __name__ == "__main__":
    data_root = "data"
    
    weights = calculate_class_weights(data_root, "morocco")
    
    if weights is not None:
        print(f"\nFor morocco_config.yaml:")
        print(f"class_weights: {[round(w, 3) for w in weights]}")