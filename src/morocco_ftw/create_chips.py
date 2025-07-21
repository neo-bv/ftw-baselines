#This scripts is used to generate chips_morocco_filtered.parquet based on the GT and sentinel2 chips
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
from shapely.geometry import box
import os

def create_chips_metadata(ftw_data_dir):
    """Create chips_morocco_filtered.parquet with real geometries from actual patch files"""
    
    print("Creating chips metadata from filtered patch files...")
    
    ftw_data_dir = Path(ftw_data_dir)
    # Updated to use filtered directory instead of original
    morocco_dir = ftw_data_dir / "morocco_filtered"
    
    # Paths to patch directories
    window_a_dir = morocco_dir / "s2_images" / "window_a"
    window_b_dir = morocco_dir / "s2_images" / "window_b"
    masks_dir = morocco_dir / "label_masks" / "semantic_3class"
    
    if not all([window_a_dir.exists(), window_b_dir.exists(), masks_dir.exists()]):
        print("Error: Required directories not found")
        return None
    
    # Get list of patch files
    window_a_files = list(window_a_dir.glob("*.tif"))
    print(f"Found {len(window_a_files)} patches")
    
    if len(window_a_files) == 0:
        print("Error: No patch files found")
        return None
    
    # Create metadata for each patch
    metadata_rows = []
    
    for patch_file in window_a_files:
        aoi_id = patch_file.stem
        
        # Check if corresponding files exist
        window_b_file = window_b_dir / f"{aoi_id}.tif"
        mask_file = masks_dir / f"{aoi_id}.tif"
        
        if not all([window_b_file.exists(), mask_file.exists()]):
            print(f"Warning: Skipping {aoi_id} - missing files")
            continue
        
        try:
            # Get spatial information from the patch file
            with rasterio.open(patch_file) as src:
                bounds = src.bounds
                crs = src.crs
                
                # Create bounding box geometry
                bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                
                # Transform to WGS84 for consistency with FTW
                from rasterio.warp import transform_bounds
                wgs84_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
                wgs84_bbox = box(*wgs84_bounds)
                
                # Assign split based on patch index (80/10/10 split)
                patch_idx = int(aoi_id)
                if patch_idx % 10 < 8:
                    split = 'train'
                elif patch_idx % 10 == 8:
                    split = 'val'
                else:
                    split = 'test'
                
                metadata_rows.append({
                    'aoi_id': aoi_id,
                    'split': split,
                    'geometry': wgs84_bbox,
                    'original_name': f"morocco_patch_{patch_idx:05d}",
                    'utm_bounds': f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
                    'utm_crs': str(crs)
                })
                
        except Exception as e:
            print(f"Error processing {aoi_id}: {e}")
            continue
    
    if len(metadata_rows) == 0:
        print("Error: No valid patches processed")
        return None
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(metadata_rows, crs='EPSG:4326')
    
    # Print summary
    split_counts = gdf['split'].value_counts()
    print(f"Total patches: {len(gdf)}")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")
    
    bounds = gdf.total_bounds
    print(f"Spatial extent: {bounds}")
    
    # Save as parquet with filtered naming
    chips_file = morocco_dir / "chips_morocco_filtered.parquet"
    gdf.to_parquet(chips_file)
    
    print(f"Created chips metadata: {chips_file}")
    
    # Verify the file
    try:
        test_gdf = gpd.read_parquet(chips_file)
        print(f"Verification: {len(test_gdf)} records, CRS: {test_gdf.crs}")
    except Exception as e:
        print(f"Verification failed: {e}")
    
    return chips_file

def test_ftw_loading(ftw_data_dir):
    """Test if FTW can load the filtered data properly"""
    
    print("Testing FTW data loading...")
    
    try:
        import sys
        sys.path.append(r"C:\Users\qin.xu\github\ftw-baselines\src")
        
        from ftw.datasets import FTW
        
        # Test datasets with filtered data
        splits = ["train", "val", "test"]
        dataset_sizes = {}
        
        for split in splits:
            dataset = FTW(
                root=str(ftw_data_dir),
                countries=["morocco_filtered"],  # Updated to use filtered directory
                split=split,
                load_boundaries=True,
                temporal_options="stacked"
            )
            dataset_sizes[split] = len(dataset)
            print(f"{split} dataset: {len(dataset)} samples")
        
        # Test loading first sample if available
        if dataset_sizes["train"] > 0:
            train_dataset = FTW(
                root=str(ftw_data_dir),
                countries=["morocco_filtered"],  # Updated to use filtered directory
                split="train",
                load_boundaries=True,
                temporal_options="stacked"
            )
            
            sample = train_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            if 'image' in sample:
                print(f"Image shape: {sample['image'].shape}")
            if 'mask' in sample:
                print(f"Mask shape: {sample['mask'].shape}")
                print(f"Mask unique values: {sample['mask'].unique()}")
        
        return all(size > 0 for size in dataset_sizes.values())
        
    except Exception as e:
        print(f"FTW loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    # Old file path (commented out)
    # ftw_data_dir = r"C:\Users\qin.xu\github\ftw-baselines\data\ftw"
    
    # Use same data directory but target the filtered subdirectory
    ftw_data_dir = r"C:\Users\qin.xu\github\ftw-baselines\data\ftw"
    
    try:
        # Create chips metadata for filtered data
        chips_file = create_chips_metadata(ftw_data_dir)
        
        if chips_file:
            # Test FTW loading with filtered data
            success = test_ftw_loading(ftw_data_dir)
            
            if success:
                print("Success: FTW can load filtered Morocco data")
                print("Ready for training: python src\\morocco_ftw\\train_morocco.py")
            else:
                print("FTW loading still has issues")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()