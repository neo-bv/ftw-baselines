import os
import sys
from osgeo import gdal, ogr, osr
import subprocess

def mosaic_with_gdal_python(tile1_path, tile2_path, output_path, aoi_shapefile=None):
    """
    Mosaic two Sentinel-2 tiles using GDAL Python bindings
    
    Parameters:
    tile1_path: str - Path to first Sentinel-2 tile
    tile2_path: str - Path to second Sentinel-2 tile
    output_path: str - Path for output mosaicked tile
    aoi_shapefile: str - Optional path to AOI shapefile for clipping
    """
    
    print("Mosaicking with GDAL Python")
    
    # Check if input files exist
    for i, file_path in enumerate([tile1_path, tile2_path], 1):
        if not os.path.exists(file_path):
            print(f"Tile {i} not found: {file_path}")
            return False
        else:
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"Tile {i} found: {file_path} ({size_mb:.1f} MB)")
    
    # Open and inspect the datasets
    print("Inspecting input tiles...")
    
    try:
        ds1 = gdal.Open(tile1_path)
        ds2 = gdal.Open(tile2_path)
        
        if ds1 is None or ds2 is None:
            print("Error opening one or both datasets")
            return False
        
        print(f"Tile 1: {ds1.RasterXSize}x{ds1.RasterYSize}, {ds1.RasterCount} bands")
        print(f"Tile 2: {ds2.RasterXSize}x{ds2.RasterYSize}, {ds2.RasterCount} bands")
        
        # Get geotransforms and projections
        gt1 = ds1.GetGeoTransform()
        gt2 = ds2.GetGeoTransform()
        proj1 = ds1.GetProjection()
        proj2 = ds2.GetProjection()
        
        print(f"Tile 1 bounds: {gt1[0]:.0f}, {gt1[3] + gt1[5] * ds1.RasterYSize:.0f}, {gt1[0] + gt1[1] * ds1.RasterXSize:.0f}, {gt1[3]:.0f}")
        print(f"Tile 2 bounds: {gt2[0]:.0f}, {gt2[3] + gt2[5] * ds2.RasterYSize:.0f}, {gt2[0] + gt2[1] * ds2.RasterXSize:.0f}, {gt2[3]:.0f}")
        
        # Check if projections match
        if proj1 != proj2:
            print("Warning: Different projections detected")
        
    except Exception as e:
        print(f"Error inspecting datasets: {e}")
        return False
    
    # Create VRT mosaic
    print("Creating VRT mosaic")
    
    vrt_path = output_path.replace('.tif', '.vrt')
    
    try:
        # Create VRT
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            resampleAlg='nearest',
            srcNodata=0,
            VRTNodata=0
        )
        
        vrt_ds = gdal.BuildVRT(vrt_path, [tile1_path, tile2_path], options=vrt_options)
        
        if vrt_ds is None:
            print("Failed to create VRT")
            return False
        
        print(f"VRT created: {vrt_path}")
        print(f"VRT size: {vrt_ds.RasterXSize}x{vrt_ds.RasterYSize}")
        print(f"VRT bands: {vrt_ds.RasterCount}")
        
        # Close VRT to flush
        vrt_ds = None
        
    except Exception as e:
        print(f"Error creating VRT: {e}")
        return False
    
    # Convert VRT to GeoTIFF
    print("Converting VRT to GeoTIFF")
    
    try:
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        )
        
        final_ds = gdal.Translate(output_path, vrt_path, options=translate_options)
        
        if final_ds is None:
            print("Failed to create final GeoTIFF")
            return False
        
        print(f"GeoTIFF created: {output_path}")
        print(f"Final size: {final_ds.RasterXSize}x{final_ds.RasterYSize}")
        print(f"Final bands: {final_ds.RasterCount}")
        
        # Close dataset
        final_ds = None
        
    except Exception as e:
        print(f"Error creating GeoTIFF: {e}")
        return False
    
    # Optional: Clip with AOI
    if aoi_shapefile and os.path.exists(aoi_shapefile):
        print(f"Clipping with AOI: {aoi_shapefile}")
        
        clipped_path = output_path.replace('.tif', '_clipped.tif')
        
        try:
            warp_options = gdal.WarpOptions(
                format='GTiff',
                cutlineDSName=aoi_shapefile,
                cropToCutline=True,
                creationOptions=['COMPRESS=LZW', 'TILED=YES']
            )
            
            clipped_ds = gdal.Warp(clipped_path, output_path, options=warp_options)
            
            if clipped_ds is None:
                print("AOI clipping failed, keeping unclipped version")
            else:
                print(f"Clipped version created: {clipped_path}")
                print(f"Clipped size: {clipped_ds.RasterXSize}x{clipped_ds.RasterYSize}")
                clipped_ds = None
                
                # Replace original with clipped version
                os.replace(clipped_path, output_path)
                print("Replaced original with clipped version")
                
        except Exception as e:
            print(f"AOI clipping error: {e}")
            print("Keeping unclipped mosaic")
    
    # Clean up VRT file
    if os.path.exists(vrt_path):
        os.remove(vrt_path)
        print("Cleaned up temporary VRT file")
    
    # Final verification
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"Mosaic completed successfully")
        print(f"Output: {output_path}")
        print(f"Size: {size_mb:.1f} MB")
        
        # Show final dataset info
        final_check = gdal.Open(output_path)
        if final_check:
            gt = final_check.GetGeoTransform()
            print(f"Final mosaic info:")
            print(f"  Dimensions: {final_check.RasterXSize} x {final_check.RasterYSize}")
            print(f"  Bands: {final_check.RasterCount}")
            print(f"  Pixel size: {gt[1]:.1f} x {abs(gt[5]):.1f} meters")
            final_check = None
        
        return True
    else:
        print("Failed - output file not created")
        return False

def mosaic_with_gdal_command(tile1_path, tile2_path, output_path):
    """Alternative method using GDAL command line tools via subprocess"""
    print("Using GDAL command line")
    
    try:
        # Method 1: Try gdal_merge.py
        cmd = [
            'gdal_merge.py',
            '-o', output_path,
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            tile1_path,
            tile2_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print("gdal_merge.py succeeded")
            return True
        else:
            print(f"gdal_merge.py failed: {result.stderr}")
            
        # Method 2: Try gdalbuildvrt + gdal_translate
        vrt_path = output_path.replace('.tif', '.vrt')
        
        # Build VRT
        cmd_vrt = ['gdalbuildvrt', vrt_path, tile1_path, tile2_path]
        print(f"Running: {' '.join(cmd_vrt)}")
        result_vrt = subprocess.run(cmd_vrt, capture_output=True, text=True)
        
        if result_vrt.returncode == 0:
            print("VRT created successfully")
            
            # Translate to GeoTIFF
            cmd_translate = [
                'gdal_translate',
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES',
                vrt_path,
                output_path
            ]
            
            print(f"Running: {' '.join(cmd_translate)}")
            result_translate = subprocess.run(cmd_translate, capture_output=True, text=True)
            
            if result_translate.returncode == 0 and os.path.exists(output_path):
                print("Translation to GeoTIFF succeeded")
                os.remove(vrt_path)  # Clean up VRT
                return True
            else:
                print(f"Translation failed: {result_translate.stderr}")
        else:
            print(f"VRT creation failed: {result_vrt.stderr}")
            
    except Exception as e:
        print(f"Command line method error: {e}")
    
    return False

def main():
    """Main function to mosaic Sentinel-2 tiles"""
    print("GDAL-based Sentinel-2 Mosaic Tool")
    
    # Configure GDAL
    gdal.UseExceptions()
    
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Option 1: Original files
    # tile1_path = os.path.join(parent_dir, "morocco_aoi1c1.tif")
    # tile2_path = os.path.join(parent_dir, "morocco_aoi2c1.tif")
    # output_path = os.path.join(parent_dir, "morocco_mosaic.tif")
    
    # Option 2: Mid files
    tile1_path = os.path.join(parent_dir, "morocco_mid1_aoi.tif")
    tile2_path = os.path.join(parent_dir, "morocco_mid2_aoi.tif")
    output_path = os.path.join(parent_dir, "morocco_mid_mosaic.tif")
    
    aoi_shapefile = os.path.join(parent_dir, "SECTEURS.shp")

    print(f"Working directory: {parent_dir}")
    print(f"Tile 1: {tile1_path}")
    print(f"Tile 2: {tile2_path}")
    print(f"AOI: {aoi_shapefile}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    # Check GDAL version
    print(f"GDAL version: {gdal.VersionInfo()}")
    
    # Check available files
    print("Available TIF files:")
    try:
        files = os.listdir(parent_dir)
        tif_files = [f for f in files if f.endswith('.tif')]
        for f in sorted(tif_files):
            full_path = os.path.join(parent_dir, f)
            size_mb = os.path.getsize(full_path) / (1024*1024)
            print(f"  {f} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    try:
        # Try GDAL Python method first
        success = mosaic_with_gdal_python(tile1_path, tile2_path, output_path, aoi_shapefile)
        
        # If that fails, try command line method
        if not success:
            print("Primary method failed, trying command line approach...")
            success = mosaic_with_gdal_command(tile1_path, tile2_path, output_path)
        
        if success:
            print("Mosaic completed successfully")
            print(f"Ready for FTW inference: {output_path}")
        else:
            print("All methods failed")
            print("Please check your input files and try manual GDAL commands")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()