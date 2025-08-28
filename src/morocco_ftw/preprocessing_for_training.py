#This script is used for create 256*256 chips for GT and Sentinel2
import geopandas as gpd
import numpy as np
import subprocess
import os
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import re

def preprocess_for_ftw_training(sentinel_path, mask_path, output_dir, patch_size=256, stride=None, min_labeled_pixels=100):
    """
    Preprocess Morocco data to match FTW training format.
    Only keeps patches that have sufficient labeled pixels.
    """
    
    if stride is None:
        stride = patch_size
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    window_a_dir = output_dir / "window_a"
    window_b_dir = output_dir / "window_b" 
    masks_dir = output_dir / "masks"
    
    window_a_dir.mkdir(exist_ok=True)
    window_b_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    print(f"Processing {sentinel_path}")
    print(f"Output: {output_dir}")
    print(f"Patch size: {patch_size}x{patch_size}, stride: {stride}")
    print(f"Min labeled pixels threshold: {min_labeled_pixels}")
    
    # Verify file alignment
    s2_width, s2_height = get_raster_dimensions(sentinel_path)
    mask_width, mask_height = get_raster_dimensions(mask_path)
    
    if s2_width != mask_width or s2_height != mask_height:
        print(f"Warning: Dimension mismatch - S2: {s2_width}x{s2_height}, Mask: {mask_width}x{mask_height}")
    
    # Split 8-band Sentinel-2 into Window A and Window B
    window_a_full = str(output_dir / "window_a_full.tif")
    window_b_full = str(output_dir / "window_b_full.tif")
    
    split_sentinel_bands(sentinel_path, window_a_full, window_b_full)
    
    # Calculate patch grid
    width, height = s2_width, s2_height
    n_patches_x = (width - patch_size) // stride + 1
    n_patches_y = (height - patch_size) // stride + 1
    total_patches = n_patches_x * n_patches_y
    
    print(f"Extracting {total_patches} patches from {width}x{height} image")
    
    # Extract patches with label filtering
    patch_count = 0
    valid_patches = 0
    skipped_empty = 0
    skipped_mostly_empty = 0
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            x_start = j * stride
            y_start = i * stride
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Skip if patch extends beyond image
            if x_end > width or y_end > height:
                continue
            
            patch_name = f"morocco_patch_{patch_count:05d}"
            
            # Extract mask patch first to check label density
            mask_patch = masks_dir / f"{patch_name}.tif"
            extract_patch(mask_path, str(mask_patch), x_start, y_start, patch_size, patch_size)
            
            # Check label density using IMPROVED method
            labeled_pixels, total_pixels = improved_count_labeled_pixels(str(mask_patch))
            
            if labeled_pixels == 0:
                os.remove(str(mask_patch))
                skipped_empty += 1
            elif labeled_pixels < min_labeled_pixels:
                os.remove(str(mask_patch))
                skipped_mostly_empty += 1
            else:
                # Extract corresponding image patches
                window_a_patch = window_a_dir / f"{patch_name}.tif"
                extract_patch(window_a_full, str(window_a_patch), x_start, y_start, patch_size, patch_size)
                
                window_b_patch = window_b_dir / f"{patch_name}.tif"
                extract_patch(window_b_full, str(window_b_patch), x_start, y_start, patch_size, patch_size)
                
                valid_patches += 1
            
            patch_count += 1
            
            if patch_count % 100 == 0:
                print(f"Processed {patch_count}/{total_patches} patches, kept {valid_patches}")
    
    print(f"Completed: {valid_patches}/{patch_count} patches kept")
    print(f"Skipped: {skipped_empty} empty, {skipped_mostly_empty} insufficient labels")
    
    # Create dataset metadata
    create_dataset_metadata(output_dir, valid_patches, patch_size, min_labeled_pixels)
    
    # Clean up temporary full images
    for temp_file in [window_a_full, window_b_full]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return output_dir

def improved_count_labeled_pixels(mask_patch_path):
    """
    IMPROVED method to count labeled pixels using gdalinfo statistics
    This replaces the old crude estimation method
    """
    cmd = f'gdalinfo -stats "{mask_patch_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        return 0, 0
    
    # Parse the output to get actual statistics
    min_val = 0
    max_val = 0
    mean_val = 0
    
    for line in result.stdout.split('\n'):
        if 'STATISTICS_MINIMUM' in line:
            match = re.search(r'STATISTICS_MINIMUM=([0-9.-]+)', line)
            if match:
                min_val = float(match.group(1))
        elif 'STATISTICS_MAXIMUM' in line:
            match = re.search(r'STATISTICS_MAXIMUM=([0-9.-]+)', line)
            if match:
                max_val = float(match.group(1))
        elif 'STATISTICS_MEAN' in line:
            match = re.search(r'STATISTICS_MEAN=([0-9.-]+)', line)
            if match:
                mean_val = float(match.group(1))
    
    total_pixels = 256 * 256
    
    if max_val == 0:
        return 0, total_pixels
    
    # Better estimation based on actual statistics
    if mean_val > 0:
        # More accurate estimate based on mean value
        # For 3-class data (0=background, 1=field, 2=boundary):
        # mean ≈ (num_field_pixels * 1 + num_boundary_pixels * 2) / total_pixels
        
        # Conservative estimate: assume most labeled pixels are class 1 (fields)
        estimated_labeled = int((mean_val - min_val) / (max_val - min_val) * total_pixels)
        estimated_labeled = max(0, min(estimated_labeled, total_pixels))
        
        return estimated_labeled, total_pixels
    else:
        return 0, total_pixels

def split_sentinel_bands(input_file, window_a_output, window_b_output):
    """Split 8-band Sentinel-2 into 4-band Window A and Window B"""
    
    # Extract bands 1-4 for Window A
    cmd_a = f'gdal_translate -b 1 -b 2 -b 3 -b 4 -co "COMPRESS=LZW" "{input_file}" "{window_a_output}"'
    result_a = subprocess.run(cmd_a, shell=True, capture_output=True, text=True)
    
    if result_a.returncode != 0:
        raise Exception(f"Failed to create Window A: {result_a.stderr}")
    
    # Extract bands 5-8 for Window B
    cmd_b = f'gdal_translate -b 5 -b 6 -b 7 -b 8 -co "COMPRESS=LZW" "{input_file}" "{window_b_output}"'
    result_b = subprocess.run(cmd_b, shell=True, capture_output=True, text=True)
    
    if result_b.returncode != 0:
        raise Exception(f"Failed to create Window B: {result_b.stderr}")

def get_raster_dimensions(raster_path):
    """Get width and height of raster"""
    cmd = f'gdalinfo "{raster_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if 'Size is' in line:
            import re
            match = re.search(r'Size is (\d+), (\d+)', line)
            if match:
                return int(match.group(1)), int(match.group(2))
    
    raise Exception("Could not determine raster dimensions")

def extract_patch(input_raster, output_patch, x_start, y_start, width, height):
    """Extract a patch from raster using gdal_translate"""
    
    cmd = f'gdal_translate -srcwin {x_start} {y_start} {width} {height} -co "COMPRESS=LZW" "{input_raster}" "{output_patch}"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to extract patch: {result.stderr}")

def create_dataset_metadata(output_dir, patch_count, patch_size, min_labeled_pixels):
    """Create metadata file for the dataset"""
    
    metadata = f"""# Morocco FTW Training Dataset (Filtered Active Fields)  
Created: {datetime.now()}
Total Valid Patches: {patch_count}
Patch Size: {patch_size}x{patch_size}
Minimum Labeled Pixels: {min_labeled_pixels}
Format: GeoTIFF
CRS: EPSG:32629 (UTM Zone 29N)
Pixel Size: 10m

Data Source: NDVI-filtered active fields
Filter Conditions:
- |NDVI_A - NDVI_B| > 0.02 (temporal change)

Structure:
- window_a/: Sentinel-2 Window A patches (R,G,B,NIR)
- window_b/: Sentinel-2 Window B patches (R,G,B,NIR)  
- masks/: 3-class mask patches (1=field, 2=boundary, 0=background)

Band Order (Windows A & B):
1. Red (B04)
2. Green (B03)  
3. Blue (B02)
4. NIR (B08)

Note: 
- Only patches with sufficient labeled pixels are included
- Only active fields (passed NDVI filter) are used for ground truth
- Background pixels (value 0) represent unlabeled areas
- Use ignore_index=0 during training to ignore unlabeled pixels
"""
    
    with open(output_dir / "dataset_info.txt", 'w') as f:
        f.write(metadata)

if __name__ == "__main__":
    
    # FIXED CONFIGURATION - Using dataset-specific thresholds
    morocco_pairs = [
        {
            'name': 'Morocco1_BL',
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_bl_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\morocco_filtered_3class_Morocco1_BL_aligned.tif',
            'min_labeled_pixels': 100  # Standard threshold
        },
        {
            'name': 'Morocco2_TR', 
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_tr_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\morocco_filtered_3class_Morocco2_TR_aligned.tif',
            'min_labeled_pixels': 100  # Standard threshold
        },
        {
            'name': 'Stef_BR',
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_br_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\morocco_filtered_3class_Stef_BR_aligned.tif',
            'min_labeled_pixels': 25   # REDUCED threshold for Stef_BR (was 100)
        }
    ]
    
    base_output_dir = r'C:\Users\qin.xu\github\ftw-baselines\morocco_ftw_training_fixed'
    
    # Check GDAL installation
    try:
        result = subprocess.run(['gdalinfo', '--version'], capture_output=True, text=True)
        print(f"GDAL version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("GDAL not found. Please install GDAL and add to PATH.")
        exit(1)
    
    print("Creating FTW training chips with FIXED filtering...")
    print("="*70)
    
    try:
        valid_pairs = []
        
        # Check which files exist
        for pair in morocco_pairs:
            sentinel_exists = os.path.exists(pair['sentinel'])
            mask_exists = os.path.exists(pair['mask'])
            
            print(f"\n{pair['name']}:")
            print(f"  Sentinel: {'✓' if sentinel_exists else '✗'} {pair['sentinel']}")
            print(f"  Mask: {'✓' if mask_exists else '✗'} {pair['mask']}")
            print(f"  Min labeled pixels: {pair['min_labeled_pixels']}")
            
            if sentinel_exists and mask_exists:
                valid_pairs.append(pair)
            else:
                print(f"  → Skipping {pair['name']} - missing files")
        
        if not valid_pairs:
            print("\nNo valid file pairs found. Please check file paths.")
            exit(1)
        
        print(f"\nProcessing {len(valid_pairs)} valid pairs...")
        
        total_patches = 0
        
        for i, pair in enumerate(valid_pairs):
            print(f"\n{'-'*50}")
            print(f"Processing: {pair['name']} ({i+1}/{len(valid_pairs)})")
            print(f"{'-'*50}")
            
            output_subdir = Path(base_output_dir) / pair['name']
            
            result_dir = preprocess_for_ftw_training(
                sentinel_path=pair['sentinel'],
                mask_path=pair['mask'], 
                output_dir=str(output_subdir),
                patch_size=256,
                stride=256,
                min_labeled_pixels=pair['min_labeled_pixels']  # Use dataset-specific threshold
            )
            
            # Count patches created
            window_a_dir = result_dir / "window_a"
            patch_count = len(list(window_a_dir.glob("*.tif")))
            total_patches += patch_count
            
            print(f"✓ Created {patch_count} training patches for {pair['name']}")
        
        print(f"\n{'='*70}")
        print(f"FIXED PREPROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total patches created: {total_patches}")
        print(f"Training data ready at: {base_output_dir}")
        print(f"\nKey fixes applied:")
        print(f"  1. Improved count_labeled_pixels() function")
        print(f"  2. Dataset-specific thresholds:")
        print(f"     - Morocco1_BL & Morocco2_TR: 100 pixels")
        print(f"     - Stef_BR: 25 pixels (4x lower)")
        print(f"  3. Better statistics-based pixel counting")
        
        print(f"\nDataset structure:")
        print(f"  {base_output_dir}/")
        for pair in valid_pairs:
            print(f"    ├── {pair['name']}/")
            print(f"    │   ├── window_a/    # Sentinel-2 Window A patches")
            print(f"    │   ├── window_b/    # Sentinel-2 Window B patches")
            print(f"    │   ├── masks/       # 3-class ground truth masks")
            print(f"    │   └── dataset_info.txt")
        
        print(f"\nNext steps:")
        print(f"  1. Review dataset_info.txt files in each subfolder")
        print(f"  2. Compare patch counts - Stef_BR should now have more patches")
        print(f"  3. Use this data for FTW model training")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()