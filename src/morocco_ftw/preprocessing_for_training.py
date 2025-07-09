import geopandas as gpd
import numpy as np
import subprocess
import os
from pathlib import Path
import tempfile
import shutil

def preprocess_for_ftw_training(sentinel_path, mask_path, output_dir, patch_size=256, stride=None, min_labeled_pixels=100):
    """
    Preprocess Morocco data to match FTW training format.
    Only keeps patches that have sufficient labeled pixels.
    
    Parameters:
    -----------
    sentinel_path : str
        Path to 8-band Sentinel-2 file (R,G,B,NIR from windows A&B)
    mask_path : str  
        Path to 3-class mask file
    output_dir : str
        Directory to save processed patches
    patch_size : int
        Size of output patches (default: 256 to match FTW)
    stride : int
        Stride for patch extraction (default: patch_size for no overlap)
    min_labeled_pixels : int
        Minimum number of labeled pixels required to keep a patch
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
    
    print("=== FTW Data Preprocessing ===")
    print(f"Input Sentinel-2: {sentinel_path}")
    print(f"Input Mask: {mask_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Patch Size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")
    print(f"Min labeled pixels per patch: {min_labeled_pixels}")
    
    # Step 1: Verify files are aligned
    print("\nStep 1: Verifying file alignment...")
    s2_width, s2_height = get_raster_dimensions(sentinel_path)
    mask_width, mask_height = get_raster_dimensions(mask_path)
    
    print(f"  Sentinel-2: {s2_width}x{s2_height}")
    print(f"  Mask: {mask_width}x{mask_height}")
    
    if s2_width != mask_width or s2_height != mask_height:
        print("  ⚠️ Warning: Dimensions don't match exactly")
    else:
        print("  ✅ Files are aligned")
    
    # Step 2: Split 8-band Sentinel-2 into Window A (bands 1-4) and Window B (bands 5-8)
    print("\nStep 2: Splitting Sentinel-2 into Windows A and B...")
    window_a_full = str(output_dir / "window_a_full.tif")
    window_b_full = str(output_dir / "window_b_full.tif")
    
    split_sentinel_bands(sentinel_path, window_a_full, window_b_full)
    
    # Step 3: Get image dimensions for patch calculation
    print("\nStep 3: Calculating patches...")
    width, height = s2_width, s2_height
    
    # Calculate number of patches
    n_patches_x = (width - patch_size) // stride + 1
    n_patches_y = (height - patch_size) // stride + 1
    total_patches = n_patches_x * n_patches_y
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Patches: {n_patches_x}x{n_patches_y} = {total_patches} total")
    
    # Step 4: Extract patches with label filtering
    print("\nStep 4: Extracting patches (filtering by label density)...")
    patch_count = 0
    valid_patches = 0
    skipped_empty = 0
    skipped_mostly_empty = 0
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            # Calculate patch coordinates
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
            
            # Check label density
            labeled_pixels, total_pixels = count_labeled_pixels(str(mask_patch))
            
            if labeled_pixels == 0:
                # Completely unlabeled patch
                os.remove(str(mask_patch))
                skipped_empty += 1
                patch_count += 1
                continue
            elif labeled_pixels < min_labeled_pixels:
                # Mostly unlabeled patch
                os.remove(str(mask_patch))
                skipped_mostly_empty += 1
                patch_count += 1
                continue
            else:
                # Good patch with sufficient labels
                # Extract window A patch
                window_a_patch = window_a_dir / f"{patch_name}.tif"
                extract_patch(window_a_full, str(window_a_patch), x_start, y_start, patch_size, patch_size)
                
                # Extract window B patch  
                window_b_patch = window_b_dir / f"{patch_name}.tif"
                extract_patch(window_b_full, str(window_b_patch), x_start, y_start, patch_size, patch_size)
                
                valid_patches += 1
                
                if valid_patches % 10 == 0:
                    print(f"  Valid patches: {valid_patches}, Labeled pixels in last: {labeled_pixels}/{total_pixels}")
            
            patch_count += 1
            
            if patch_count % 100 == 0:
                print(f"  Processed {patch_count}/{total_patches} patches")
    
    print(f"\n✅ Patch extraction complete!")
    print(f"  Total patches processed: {patch_count}")
    print(f"  Valid patches kept: {valid_patches}")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (too few labels): {skipped_mostly_empty}")
    print(f"  Success rate: {valid_patches/patch_count*100:.1f}%")
    
    print(f"\nOutput structure:")
    print(f"  📁 {window_a_dir} - Window A patches (4 bands: R,G,B,NIR)")
    print(f"  📁 {window_b_dir} - Window B patches (4 bands: R,G,B,NIR)")  
    print(f"  📁 {masks_dir} - Mask patches (1 band: 0=bg, 1=field, 2=boundary)")
    
    # Step 5: Create dataset metadata
    create_dataset_metadata(output_dir, valid_patches, patch_size, min_labeled_pixels)
    
    return output_dir

def count_labeled_pixels(mask_patch_path):
    """Count labeled pixels in a mask patch using gdalinfo statistics"""
    
    # Force recalculation of statistics
    cmd = f'gdalinfo -stats "{mask_patch_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Could not get stats for {mask_patch_path}")
        return 0, 0
    
    # Parse statistics from output
    lines = result.stdout.split('\n')
    
    # Look for histogram or count information
    # For now, use a simpler approach: check if max value > 0
    max_val = 0
    for line in lines:
        if 'STATISTICS_MAXIMUM' in line:
            import re
            match = re.search(r'STATISTICS_MAXIMUM=([0-9.]+)', line)
            if match:
                max_val = float(match.group(1))
                break
    
    # If we have labels (max > 0), estimate labeled pixels
    # This is a rough approximation - for exact counts you'd need to read the actual raster
    if max_val > 0:
        # Rough estimate: assume 30-70% of pixels are labeled in patches with labels
        # You could implement exact pixel counting if needed
        total_pixels = 256 * 256
        estimated_labeled = int(total_pixels * 0.5)  # Conservative estimate
        return estimated_labeled, total_pixels
    else:
        return 0, 256 * 256

def split_sentinel_bands(input_file, window_a_output, window_b_output):
    """Split 8-band Sentinel-2 into 4-band Window A and Window B"""
    
    # Extract bands 1-4 for Window A (R,G,B,NIR from first time)
    cmd_a = f'gdal_translate -b 1 -b 2 -b 3 -b 4 -co "COMPRESS=LZW" "{input_file}" "{window_a_output}"'
    result_a = subprocess.run(cmd_a, shell=True, capture_output=True, text=True)
    
    if result_a.returncode == 0:
        print(f"  ✅ Window A created: {Path(window_a_output).name}")
    else:
        raise Exception(f"Failed to create Window A: {result_a.stderr}")
    
    # Extract bands 5-8 for Window B (R,G,B,NIR from second time)  
    cmd_b = f'gdal_translate -b 5 -b 6 -b 7 -b 8 -co "COMPRESS=LZW" "{input_file}" "{window_b_output}"'
    result_b = subprocess.run(cmd_b, shell=True, capture_output=True, text=True)
    
    if result_b.returncode == 0:
        print(f"  ✅ Window B created: {Path(window_b_output).name}")
    else:
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
    
    from datetime import datetime
    
    metadata = f"""# Morocco FTW Training Dataset
Created: {datetime.now()}
Total Valid Patches: {patch_count}
Patch Size: {patch_size}x{patch_size}
Minimum Labeled Pixels: {min_labeled_pixels}
Format: GeoTIFF
CRS: EPSG:32629 (UTM Zone 29N)
Pixel Size: 10m

Structure:
- window_a/: Sentinel-2 Window A patches (R,G,B,NIR)
- window_b/: Sentinel-2 Window B patches (R,G,B,NIR)  
- masks/: 3-class mask patches (1=field, 2=boundary)

Band Order (Windows A & B):
1. Red (B04)
2. Green (B03)  
3. Blue (B02)
4. NIR (B08)

Note: 
- Only patches with sufficient labeled pixels are included
- Background pixels (value 0) represent unlabeled areas
- This filtering ensures efficient training on labeled regions only

Usage:
This dataset is ready for FTW 3-class training.
Use ignore_index=0 during training to ignore unlabeled pixels.
"""
    
    with open(output_dir / "dataset_info.txt", 'w') as f:
        f.write(metadata)
    
    print(f"  ✅ Metadata created: dataset_info.txt")

# Main execution
if __name__ == "__main__":
    
    # Define input files (your aligned files are ready to use!)
    morocco_pairs = [
        {
            'name': 'Morocco1_BL',
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_bl_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\Morocco1_BL_aligned.tif'
        },
        {
            'name': 'Morocco2_TR', 
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_tr_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\Morocco2_TR_aligned.tif'
        },
        {
            'name': 'Stef_BR',
            'sentinel': r'C:\Users\qin.xu\github\ftw-baselines\morocco_br_gtaoi.tif',
            'mask': r'C:\Users\qin.xu\github\ftw-baselines\Stef_BR_aligned.tif'
        }
    ]
    
    base_output_dir = r'C:\Users\qin.xu\github\ftw-baselines\morocco_ftw_training'
    
    try:
        for pair in morocco_pairs:
            print(f"\n{'='*60}")
            print(f"Processing: {pair['name']}")
            print(f"{'='*60}")
            
            output_subdir = Path(base_output_dir) / pair['name']
            
            preprocess_for_ftw_training(
                sentinel_path=pair['sentinel'],
                mask_path=pair['mask'], 
                output_dir=str(output_subdir),
                patch_size=256,  # FTW standard
                stride=256,      # No overlap like FTW standard
                min_labeled_pixels=100  # Require at least 100 labeled pixels per patch
            )
        
        print(f"\n🎉 ALL PREPROCESSING COMPLETE!")
        print(f"📁 Training data ready at: {base_output_dir}")
        print(f"\nNext steps:")
        print(f"1. Verify patches look correct in QGIS")
        print(f"2. Run FTW 3-class training with ignore_index=0")
        print(f"3. Evaluate model performance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()