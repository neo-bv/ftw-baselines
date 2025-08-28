#This script is used for structure data exactly same as ftw
import os
import shutil
import pandas as pd
import geopandas as gpd
from pathlib import Path

BASE_PATH = Path(os.environ.get('FTW_BASE_PATH', Path(__file__).parent.parent.parent))
print(f"Using base path: {BASE_PATH}")

def organize_morocco_data_for_ftw(preprocessed_dir, ftw_data_dir):
    """
    Organize preprocessed Morocco patches to match FTW dataset structure
    
    Expected FTW structure:
    ftw_data_dir/
    ├── morocco/
    │   ├── chips_morocco.parquet
    │   ├── s2_images/
    │   │   ├── window_a/
    │   │   └── window_b/
    │   └── label_masks/
    │       └── semantic_3class/
    """
    
    print("Organizing Morocco data for FTW training...")
    
    preprocessed_dir = Path(preprocessed_dir)
    ftw_data_dir = Path(ftw_data_dir)
    
    # Create FTW structure with filtered data directory
    morocco_dir = ftw_data_dir / "morocco_filtered"
    s2_dir = morocco_dir / "s2_images"
    window_a_dir = s2_dir / "window_a"
    window_b_dir = s2_dir / "window_b"
    masks_dir = morocco_dir / "label_masks" / "semantic_3class"
    
    # Create directories
    for dir_path in [morocco_dir, s2_dir, window_a_dir, window_b_dir, masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {morocco_dir}")
    
    # Collect all preprocessed data from all regions
    all_patches = []
    patch_counter = 0
    
    # Process each region (Morocco1_BL, Morocco2_TR, Stef_BR)
    expected_regions = ["Morocco1_BL", "Morocco2_TR", "Stef_BR"]
    
    for region_name in expected_regions:
        region_dir = preprocessed_dir / region_name
        if not region_dir.exists() or not region_dir.is_dir():
            print(f"Warning: Region directory not found: {region_dir}")
            continue
            
        print(f"Processing region: {region_dir.name}")
        
        region_window_a = region_dir / "window_a"
        region_window_b = region_dir / "window_b"
        region_masks = region_dir / "masks"
        
        if not all([region_window_a.exists(), region_window_b.exists(), region_masks.exists()]):
            print(f"Skipping {region_dir.name} - incomplete data")
            continue
        
        # Get list of patches in this region
        window_a_patches = list(region_window_a.glob("*.tif"))
        print(f"Found {len(window_a_patches)} patches")
        
        for patch_file in window_a_patches:
            patch_name = patch_file.stem
            
            # Check if corresponding files exist
            window_b_file = region_window_b / f"{patch_name}.tif"
            mask_file = region_masks / f"{patch_name}.tif"
            
            if not all([window_b_file.exists(), mask_file.exists()]):
                print(f"Skipping {patch_name} - missing files")
                continue
            
            # Copy files with new sequential naming
            new_name = f"{patch_counter}.tif"
            
            shutil.copy2(str(patch_file), window_a_dir / new_name)
            shutil.copy2(str(window_b_file), window_b_dir / new_name)
            shutil.copy2(str(mask_file), masks_dir / new_name)
            
            # Record patch metadata
            all_patches.append({
                'aoi_id': patch_counter,
                'original_name': patch_name,
                'region': region_dir.name,
                'split': assign_split(patch_counter, len(window_a_patches))
            })
            
            patch_counter += 1
    
    print(f"Organized {patch_counter} total patches")
    
    # Create chips metadata file
    print("Creating metadata file...")
    chips_df = pd.DataFrame(all_patches)
    
    # Add required FTW columns with dummy geometries
    from shapely.geometry import Point
    chips_df['geometry'] = [Point(0, 0) for _ in range(len(chips_df))]
    
    chips_gdf = gpd.GeoDataFrame(chips_df, crs='EPSG:4326')
    chips_parquet = morocco_dir / "chips_morocco_filtered.parquet"
    chips_gdf.to_parquet(str(chips_parquet))
    
    print(f"Created: {chips_parquet}")
    
    # Print split summary
    split_counts = chips_df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"{split} patches: {count}")
    
    print(f"Morocco data ready for FTW training at: {morocco_dir}")
    
    return str(ftw_data_dir)

def assign_split(patch_idx, total_patches):
    """Assign train/val/test split (80%/10%/10%)"""
    if patch_idx % 10 < 8:
        return 'train'
    elif patch_idx % 10 == 8:
        return 'val'
    else:
        return 'test'



if __name__ == "__main__":
    # Configuration
    # Old preprocessed directory 
    # preprocessed_dir = BASE_PATH / "morocco_ftw_training"
    
    # New filtered preprocessed directory
    preprocessed_dir = BASE_PATH / "morocco_ftw_training"
    ftw_data_dir = BASE_PATH / "data" / "ftw"
    
    try:
        # Organize data
        final_data_dir = organize_morocco_data_for_ftw(str(preprocessed_dir), str(ftw_data_dir))
        print("Data organization complete")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()