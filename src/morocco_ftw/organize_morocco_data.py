import os
import shutil
import pandas as pd
import geopandas as gpd
from pathlib import Path

def organize_morocco_data_for_ftw(preprocessed_dir, ftw_data_dir):
    """
    Organize preprocessed Morocco patches to match FTW dataset structure
    
    Expected FTW structure:
    ftw_data_dir/
    ├── morocco/
    │   ├── chips_morocco.parquet  # Metadata file
    │   ├── s2_images/
    │   │   ├── window_a/
    │   │   │   ├── 0.tif
    │   │   │   ├── 1.tif
    │   │   │   └── ...
    │   │   └── window_b/
    │   │       ├── 0.tif
    │   │       ├── 1.tif
    │   │       └── ...
    │   └── label_masks/
    │       └── semantic_3class/
    │           ├── 0.tif
    │           ├── 1.tif
    │           └── ...
    """
    
    print("🗂️ Organizing Morocco data for FTW training...")
    
    preprocessed_dir = Path(preprocessed_dir)
    ftw_data_dir = Path(ftw_data_dir)
    
    # Create FTW structure
    morocco_dir = ftw_data_dir / "morocco"
    s2_dir = morocco_dir / "s2_images"
    window_a_dir = s2_dir / "window_a"
    window_b_dir = s2_dir / "window_b"
    masks_dir = morocco_dir / "label_masks" / "semantic_3class"
    
    # Create directories
    for dir_path in [morocco_dir, s2_dir, window_a_dir, window_b_dir, masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created FTW directory structure at: {morocco_dir}")
    
    # Collect all preprocessed data from all regions
    all_patches = []
    patch_counter = 0
    
    # Process each region (Morocco1_BL, Morocco2_TR, Stef_BR)
    expected_regions = ["Morocco1_BL", "Morocco2_TR", "Stef_BR"]
    
    for region_name in expected_regions:
        region_dir = preprocessed_dir / region_name
        if not region_dir.exists() or not region_dir.is_dir():
            print(f"  ⚠️ Region directory not found: {region_dir}")
            continue
            
        print(f"\nProcessing region: {region_dir.name}")
        
        region_window_a = region_dir / "window_a"
        region_window_b = region_dir / "window_b"
        region_masks = region_dir / "masks"
        
        if not all([region_window_a.exists(), region_window_b.exists(), region_masks.exists()]):
            print(f"  ⚠️ Skipping {region_dir.name} - incomplete data")
            continue
        
        # Get list of patches in this region
        window_a_patches = list(region_window_a.glob("*.tif"))
        print(f"  Found {len(window_a_patches)} patches")
        
        for patch_file in window_a_patches:
            patch_name = patch_file.stem  # e.g., "morocco_patch_00001"
            
            # Check if corresponding files exist
            window_b_file = region_window_b / f"{patch_name}.tif"
            mask_file = region_masks / f"{patch_name}.tif"
            
            if not all([window_b_file.exists(), mask_file.exists()]):
                print(f"    ⚠️ Skipping {patch_name} - missing files")
                continue
            
            # Copy files with new sequential naming
            new_name = f"{patch_counter}.tif"
            
            shutil.copy2(patch_file, window_a_dir / new_name)
            shutil.copy2(window_b_file, window_b_dir / new_name)
            shutil.copy2(mask_file, masks_dir / new_name)
            
            # Record patch metadata
            all_patches.append({
                'aoi_id': patch_counter,
                'original_name': patch_name,
                'region': region_dir.name,
                'split': assign_split(patch_counter, len(window_a_patches))  # Assign train/val/test
            })
            
            patch_counter += 1
    
    print(f"\n✅ Organized {patch_counter} total patches")
    
    # Create chips metadata file (required by FTW)
    print("\n📋 Creating metadata file...")
    chips_df = pd.DataFrame(all_patches)
    
    # Add required FTW columns
    chips_df['geometry'] = None  # You could add actual geometries if needed
    
    # Save as parquet (GeoDataFrame format expected by FTW)
    # Since we don't have actual geometries, create dummy ones
    from shapely.geometry import Point
    chips_df['geometry'] = [Point(0, 0) for _ in range(len(chips_df))]
    
    chips_gdf = gpd.GeoDataFrame(chips_df, crs='EPSG:4326')
    chips_parquet = morocco_dir / "chips_morocco.parquet"
    chips_gdf.to_parquet(chips_parquet)
    
    print(f"  Created: {chips_parquet}")
    print(f"  Train patches: {len(chips_df[chips_df['split'] == 'train'])}")
    print(f"  Val patches: {len(chips_df[chips_df['split'] == 'val'])}")
    print(f"  Test patches: {len(chips_df[chips_df['split'] == 'test'])}")
    
    print(f"\n🎉 Morocco data ready for FTW training!")
    print(f"📁 Data location: {morocco_dir}")
    
    return str(ftw_data_dir)

def assign_split(patch_idx, total_patches):
    """Assign train/val/test split similar to FTW methodology"""
    # Use deterministic split based on patch index
    # 80% train, 10% val, 10% test
    if patch_idx % 10 < 8:
        return 'train'
    elif patch_idx % 10 == 8:
        return 'val'
    else:
        return 'test'

def create_training_script(ftw_data_dir, config_path):
    """Create a simple training script"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Morocco FTW Training Script
"""

import os
import sys
from pathlib import Path

# Add FTW to path if needed
ftw_path = Path(__file__).parent / "src"
if str(ftw_path) not in sys.path:
    sys.path.insert(0, str(ftw_path))

def main():
    """Run Morocco FTW training"""
    
    # Set environment variables for best performance
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    os.environ["GDAL_MAX_RAW_BLOCK_CACHE_SIZE"] = "200000000"
    os.environ["GDAL_SWATH_SIZE"] = "200000000"
    os.environ["VSI_CURL_CACHE_SIZE"] = "200000000"
    
    # Import after setting environment
    from ftw_cli.model import fit
    
    print("🇲🇦 Starting Morocco FTW Training...")
    print(f"📁 Data directory: {ftw_data_dir}")
    print(f"⚙️ Config file: {config_path}")
    
    try:
        # Run training using FTW's built-in fit function
        fit(
            config="{config_path}",
            ckpt_path=None,  # Start from scratch
            cli_args=[]  # No additional CLI args
        )
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    script_path = Path("train_morocco_ftw.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(script_path, 0o755)
    except:
        pass
    
    print(f"📜 Created training script: {script_path}")
    return script_path

if __name__ == "__main__":
    # Paths
    preprocessed_dir = r"C:\Users\qin.xu\github\ftw-baselines\morocco_ftw_training"
    ftw_data_dir = r"C:\Users\qin.xu\github\ftw-baselines\data\ftw"
    config_path = r"C:\Users\qin.xu\github\ftw-baselines\morocco_config.yaml"
    
    try:
        # Organize data
        final_data_dir = organize_morocco_data_for_ftw(preprocessed_dir, ftw_data_dir)
        
        # Create training script
        script_path = create_training_script(final_data_dir, config_path)
        
        print(f"\n📋 Next Steps:")
        print(f"1. Save the YAML config as: {config_path}")
        print(f"2. Run training: python {script_path}")
        print(f"3. Monitor training: tensorboard --logdir logs/Morocco-FTW")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()