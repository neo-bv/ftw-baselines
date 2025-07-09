import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
from shapely.geometry import box
import os

def create_proper_chips_metadata(ftw_data_dir):
    """
    Create proper chips_morocco.parquet with real geometries from actual patch files
    """
    
    print("🗂️ Creating proper chips metadata from actual patch files...")
    
    ftw_data_dir = Path(ftw_data_dir)
    morocco_dir = ftw_data_dir / "morocco"
    
    # Paths to patch directories
    window_a_dir = morocco_dir / "s2_images" / "window_a"
    window_b_dir = morocco_dir / "s2_images" / "window_b"
    masks_dir = morocco_dir / "label_masks" / "semantic_3class"
    
    if not all([window_a_dir.exists(), window_b_dir.exists(), masks_dir.exists()]):
        print("❌ Required directories not found!")
        return
    
    # Get list of patch files
    window_a_files = list(window_a_dir.glob("*.tif"))
    print(f"Found {len(window_a_files)} patches")
    
    if len(window_a_files) == 0:
        print("❌ No patch files found!")
        return
    
    # Create metadata for each patch
    metadata_rows = []
    
    for patch_file in window_a_files:
        aoi_id = patch_file.stem  # e.g., "0", "1", "2", etc.
        
        # Check if corresponding files exist
        window_b_file = window_b_dir / f"{aoi_id}.tif"
        mask_file = masks_dir / f"{aoi_id}.tif"
        
        if not all([window_b_file.exists(), mask_file.exists()]):
            print(f"⚠️ Skipping {aoi_id} - missing files")
            continue
        
        try:
            # Get spatial information from the patch file
            with rasterio.open(patch_file) as src:
                # Get bounds in the file's CRS (likely UTM)
                bounds = src.bounds
                crs = src.crs
                
                # Create bounding box geometry
                bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                
                # Transform to WGS84 (EPSG:4326) for consistency with FTW
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
                    'original_name': f"morocco_patch_{patch_idx:05d}",  # Keep track of original name
                    'utm_bounds': f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
                    'utm_crs': str(crs)
                })
                
                print(f"  ✅ Processed patch {aoi_id} -> {split}")
                
        except Exception as e:
            print(f"  ❌ Error processing {aoi_id}: {e}")
            continue
    
    if len(metadata_rows) == 0:
        print("❌ No valid patches processed!")
        return
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(metadata_rows, crs='EPSG:4326')
    
    # Print summary
    split_counts = gdf['split'].value_counts()
    print(f"\n📊 Summary:")
    print(f"  Total patches: {len(gdf)}")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")
    
    # Print extent
    bounds = gdf.total_bounds
    print(f"  Spatial extent: {bounds}")
    print(f"  CRS: {gdf.crs}")
    
    # Save as parquet
    chips_file = morocco_dir / "chips_morocco.parquet"
    gdf.to_parquet(chips_file)
    
    print(f"\n✅ Created proper chips metadata: {chips_file}")
    
    # Verify the file
    print(f"\n🔍 Verifying created file...")
    try:
        test_gdf = gpd.read_parquet(chips_file)
        print(f"  ✅ File loads correctly")
        print(f"  ✅ {len(test_gdf)} records")
        print(f"  ✅ Columns: {list(test_gdf.columns)}")
        print(f"  ✅ CRS: {test_gdf.crs}")
        print(f"  ✅ Geometry type: {test_gdf.geometry.geom_type.iloc[0]}")
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
    
    return chips_file

def test_ftw_loading_after_fix(ftw_data_dir):
    """Test if FTW can now load the data properly"""
    
    print(f"\n🧪 Testing FTW data loading after fix...")
    
    try:
        import sys
        sys.path.append(r"C:\Users\qin.xu\github\ftw-baselines\src")
        
        from ftw.datasets import FTW
        
        # Test train dataset
        train_dataset = FTW(
            root=str(ftw_data_dir),
            countries=["morocco"],
            split="train",
            load_boundaries=True,
            temporal_options="stacked"
        )
        print(f"  📊 Train dataset: {len(train_dataset)} samples")
        
        # Test val dataset
        val_dataset = FTW(
            root=str(ftw_data_dir),
            countries=["morocco"],
            split="val",
            load_boundaries=True,
            temporal_options="stacked"
        )
        print(f"  📊 Val dataset: {len(val_dataset)} samples")
        
        # Test test dataset
        test_dataset = FTW(
            root=str(ftw_data_dir),
            countries=["morocco"],
            split="test",
            load_boundaries=True,
            temporal_options="stacked"
        )
        print(f"  📊 Test dataset: {len(test_dataset)} samples")
        
        if len(train_dataset) > 0:
            print(f"\n  🎉 Success! Trying to load first sample...")
            sample = train_dataset[0]
            print(f"    Sample keys: {sample.keys()}")
            if 'image' in sample:
                print(f"    Image shape: {sample['image'].shape}")
            if 'mask' in sample:
                print(f"    Mask shape: {sample['mask'].shape}")
                print(f"    Mask unique values: {sample['mask'].unique()}")
        
        return len(train_dataset) > 0 and len(val_dataset) > 0
        
    except Exception as e:
        print(f"  ❌ FTW loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    ftw_data_dir = r"C:\Users\qin.xu\github\ftw-baselines\data\ftw"
    
    try:
        # Create proper chips metadata
        chips_file = create_proper_chips_metadata(ftw_data_dir)
        
        if chips_file:
            # Test FTW loading
            success = test_ftw_loading_after_fix(ftw_data_dir)
            
            if success:
                print(f"\n🎉 SUCCESS! FTW can now load your Morocco data properly!")
                print(f"💡 You can now run training again:")
                print(f"   python src\\morocco_ftw\\train_morocco.py")
            else:
                print(f"\n❌ FTW loading still has issues. Check the error messages above.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()