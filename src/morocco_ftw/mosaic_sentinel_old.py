import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os

print("=== SCRIPT STARTING ===")
def mosaic_sentinel_tiles(tile1_path, tile2_path, output_path, aoi_shapefile=None):
    """
    Mosaic two Sentinel-2 tiles and optionally clip to AOI
    
    Parameters:
    tile1_path: str - Path to first Sentinel-2 tile
    tile2_path: str - Path to second Sentinel-2 tile
    output_path: str - Path for output mosaicked tile
    aoi_shapefile: str - Optional path to AOI shapefile for clipping
    """
    
    print("=== MOSAICKING SENTINEL-2 TILES ===")
    
    # Open and inspect the two Sentinel-2 tiles
    print("Opening Sentinel-2 tiles...")
    
    try:
        src1 = rasterio.open(tile1_path)
        print(f"✓ Tile 1 opened: {tile1_path}")
        print(f"  - Shape: {src1.shape}")
        print(f"  - Bands: {src1.count}")
        print(f"  - CRS: {src1.crs}")
        print(f"  - Bounds: {src1.bounds}")
        print(f"  - Data type: {src1.dtypes[0]}")
    except Exception as e:
        print(f"✗ Error opening tile 1: {e}")
        return False
    
    try:
        src2 = rasterio.open(tile2_path)
        print(f"✓ Tile 2 opened: {tile2_path}")
        print(f"  - Shape: {src2.shape}")
        print(f"  - Bands: {src2.count}")
        print(f"  - CRS: {src2.crs}")
        print(f"  - Bounds: {src2.bounds}")
        print(f"  - Data type: {src2.dtypes[0]}")
    except Exception as e:
        print(f"✗ Error opening tile 2: {e}")
        src1.close()
        return False
    
    # Check compatibility
    print("\nChecking tile compatibility...")
    
    if src1.crs != src2.crs:
        print(f"⚠️  Warning: Different CRS - Tile 1: {src1.crs}, Tile 2: {src2.crs}")
        print("   Rasterio will handle reprojection during merge")
    else:
        print(f"✓ CRS match: {src1.crs}")
    
    if src1.count != src2.count:
        print(f"⚠️  Warning: Different band counts - Tile 1: {src1.count}, Tile 2: {src2.count}")
    else:
        print(f"✓ Band count match: {src1.count} bands")
    
    if src1.dtypes != src2.dtypes:
        print(f"⚠️  Warning: Different data types - Tile 1: {src1.dtypes}, Tile 2: {src2.dtypes}")
    else:
        print(f"✓ Data types match: {src1.dtypes[0]}")
    
    # Perform the mosaic
    print("\nPerforming mosaic...")
    
    try:
        # Merge the tiles
        mosaic, out_trans = merge([src1, src2], method='first')
        print(f"✓ Mosaic successful")
        print(f"  - Output shape: {mosaic.shape}")
        print(f"  - Output bands: {mosaic.shape[0]}")
        
        # Get metadata from first source
        out_meta = src1.meta.copy()
        
        # Update metadata for merged raster
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })
        
    except Exception as e:
        print(f"✗ Error during mosaic: {e}")
        src1.close()
        src2.close()
        return False
    
    # Handle AOI clipping if provided
    if aoi_shapefile and os.path.exists(aoi_shapefile):
        print(f"\nClipping to AOI: {aoi_shapefile}")
        
        try:
            # Load AOI
            aoi_gdf = gpd.read_file(aoi_shapefile)
            print(f"✓ AOI loaded: {len(aoi_gdf)} features")
            
            # Reproject AOI if necessary
            if aoi_gdf.crs != src1.crs:
                print(f"Reprojecting AOI from {aoi_gdf.crs} to {src1.crs}")
                aoi_gdf = aoi_gdf.to_crs(src1.crs)
            
            # Create temporary mosaic file
            temp_mosaic_path = output_path.replace('.tif', '_temp_mosaic.tif')
            
            with rasterio.open(temp_mosaic_path, "w", **out_meta) as temp_dst:
                temp_dst.write(mosaic)
            
            # Clip with AOI
            with rasterio.open(temp_mosaic_path) as temp_src:
                geometries = [geom for geom in aoi_gdf.geometry]
                clipped_mosaic, clipped_transform = mask(
                    temp_src, 
                    geometries, 
                    crop=True, 
                    filled=True, 
                    nodata=0
                )
                
                # Update metadata for clipped output
                out_meta.update({
                    "height": clipped_mosaic.shape[1],
                    "width": clipped_mosaic.shape[2],
                    "transform": clipped_transform,
                    "nodata": 0
                })
                
                # Use clipped data
                mosaic = clipped_mosaic
            
            # Clean up temporary file
            os.remove(temp_mosaic_path)
            print("✓ AOI clipping completed")
            
        except Exception as e:
            print(f"⚠️  Warning: AOI clipping failed: {e}")
            print("   Proceeding without clipping...")
    
    # Save the final mosaic
    print(f"\nSaving mosaic to: {output_path}")
    
    try:
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)
        
        print("✓ Mosaic saved successfully!")
        
        # Show output file info
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"📁 Output file: {output_path}")
        print(f"📊 File size: {size_mb:.1f} MB")
        
        # Verify the output
        with rasterio.open(output_path) as check:
            print(f"📋 Final mosaic info:")
            print(f"   - Shape: {check.shape}")
            print(f"   - Bands: {check.count}")
            print(f"   - CRS: {check.crs}")
            print(f"   - Bounds: {check.bounds}")
        
    except Exception as e:
        print(f"✗ Error saving mosaic: {e}")
        src1.close()
        src2.close()
        return False
    
    # Close source files
    src1.close()
    src2.close()
    
    print(f"\n🎉 SUCCESS! Mosaic completed: {output_path}")
    return True

def check_sentinel_files(file_paths):
    """Check Sentinel-2 files and show their properties"""
    print("=== CHECKING SENTINEL-2 FILES ===")
    
    for i, file_path in enumerate(file_paths, 1):
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"✓ Tile {i}: {file_path} ({size_mb:.1f} MB)")
            
            # Try to get basic info
            try:
                with rasterio.open(file_path) as src:
                    print(f"   - Dimensions: {src.width} x {src.height}")
                    print(f"   - Bands: {src.count}")
                    print(f"   - Data type: {src.dtypes[0]}")
            except Exception as e:
                print(f"   - Warning: Could not read metadata: {e}")
        else:
            print(f"✗ Tile {i}: Missing - {file_path}")
            return False
    
    return True

def main():
    """Main function to mosaic Sentinel-2 tiles"""
    print("=== SENTINEL-2 TILE MOSAICKING TOOL ===")
    
    # Get the script directory and parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # File paths
    tile1_path = os.path.join(parent_dir, "morocco_aoi1c1.tif")
    tile2_path = os.path.join(parent_dir, "morocco_aoi2c1.tif")
    aoi_shapefile = os.path.join(parent_dir, "SECTEURS.shp")
    output_path = os.path.join(parent_dir, "morocco_mosaic.tif")
    
    print(f"Working directory: {parent_dir}")
    print("-" * 60)
    
    try:
        # Check if input files exist
        if not check_sentinel_files([tile1_path, tile2_path]):
            print("\n❌ Missing Sentinel-2 tiles. Please check file paths.")
            
            # Show available TIF files
            print("\n=== AVAILABLE TIF FILES ===")
            try:
                files = os.listdir(parent_dir)
                tif_files = [f for f in files if f.endswith('.tif')]
                for f in sorted(tif_files):
                    size_mb = os.path.getsize(os.path.join(parent_dir, f)) / (1024*1024)
                    print(f"  - {f} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"Error listing files: {e}")
            return
        
        # Check for AOI file
        if os.path.exists(aoi_shapefile):
            print(f"✓ AOI shapefile found: {aoi_shapefile}")
            use_aoi = True
        else:
            print(f"⚠️  AOI shapefile not found: {aoi_shapefile}")
            print("   Proceeding without AOI clipping...")
            use_aoi = False
        
        # Perform the mosaic
        print(f"\n=== STARTING MOSAIC OPERATION ===")
        success = mosaic_sentinel_tiles(
            tile1_path, 
            tile2_path, 
            output_path, 
            aoi_shapefile if use_aoi else None
        )
        
        if success:
            print(f"\n🎉 MOSAIC COMPLETED SUCCESSFULLY!")
            print(f"📁 Ready for FTW inference: {output_path}")
            print(f"\nNext step: Run FTW inference on the mosaicked tile:")
            print(f"   python your_ftw_script.py --input {output_path}")
        else:
            print(f"\n❌ MOSAIC FAILED")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()