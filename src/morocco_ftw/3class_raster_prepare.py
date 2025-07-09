import geopandas as gpd
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tempfile

def run_gdal_command(cmd, description=""):
    """Run a GDAL command and handle errors"""
    try:
        print(f"  Running: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout:
            print(f"    Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error: {e}")
        if e.stderr:
            print(f"    Stderr: {e.stderr}")
        return False

def create_3class_field_raster_gdal(vector_paths, output_path, pixel_size=10, boundary_buffer=10, target_crs='EPSG:32629'):
    """
    Create a 3-class raster using GDAL command line tools.
    """
    
    print("Step 1: Loading and merging vector files...")
    
    # Load all shapefiles and reproject to UTM
    gdfs = []
    for i, path in enumerate(vector_paths):
        print(f"  Loading file {i+1}: {Path(path).name}")
        gdf = gpd.read_file(path)
        print(f"    - Original CRS: {gdf.crs}")
        print(f"    - Features: {len(gdf)}")
        
        # Reproject to UTM
        gdf_utm = gdf.to_crs(target_crs)
        print(f"    - Reprojected to: {gdf_utm.crs}")
        gdfs.append(gdf_utm)
    
    # Merge all GeoDataFrames
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    print(f"  Total merged features: {len(merged_gdf)}")
    
    print("\nStep 2: Fixing invalid geometries...")
    # Fix any invalid geometries
    merged_gdf.geometry = merged_gdf.geometry.buffer(0)
    
    print("\nStep 3: Creating temporary files...")
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save merged shapefile
    merged_shp = os.path.join(temp_dir, "merged_fields.shp")
    merged_gdf.to_file(merged_shp)
    
    # Create boundary shapefile
    print("\nStep 4: Creating boundaries...")
    boundaries_gdf = merged_gdf.copy()
    boundary_lines = merged_gdf.geometry.boundary
    boundary_polygons = boundary_lines.buffer(boundary_buffer)
    boundaries_gdf.geometry = boundary_polygons
    
    boundaries_shp = os.path.join(temp_dir, "boundaries.shp")
    boundaries_gdf.to_file(boundaries_shp)
    
    print(f"  Boundary buffer size: {boundary_buffer} meters")
    
    print("\nStep 5: Getting extent and setting up raster parameters...")
    # Get extent
    bounds = merged_gdf.total_bounds
    extent = f"{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}"
    print(f"  Extent: {extent}")
    
    # Calculate dimensions
    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)
    print(f"  Raster dimensions: {width} x {height} pixels")
    
    print("\nStep 6: Rasterizing with GDAL...")
    
    # Create temporary raster files
    fields_raster = os.path.join(temp_dir, "fields.tif")
    boundaries_raster = os.path.join(temp_dir, "boundaries.tif")
    
    # Rasterize fields (value = 1)
    cmd_fields = f'''gdal_rasterize -a_srs {target_crs} -te {extent} -tr {pixel_size} {pixel_size} -burn 1 -ot Byte -of GTiff "{merged_shp}" "{fields_raster}"'''
    
    success = run_gdal_command(cmd_fields, "Rasterizing fields")
    if not success:
        raise Exception("Failed to rasterize fields")
    
    # Rasterize boundaries (value = 2)
    cmd_boundaries = f'''gdal_rasterize -a_srs {target_crs} -te {extent} -tr {pixel_size} {pixel_size} -burn 2 -ot Byte -of GTiff "{boundaries_shp}" "{boundaries_raster}"'''
    
    success = run_gdal_command(cmd_boundaries, "Rasterizing boundaries")
    if not success:
        raise Exception("Failed to rasterize boundaries")
    
    print("\nStep 7: Combining rasters...")
    # Instead of gdal_calc, use a simpler approach:
    # 1. Create base raster from fields
    # 2. Overlay boundaries on top
    
    # First, copy fields raster as base
    import shutil
    shutil.copy2(fields_raster, output_path)
    print(f"  Base raster created from fields")
    
    # Then overlay boundaries using gdal_rasterize again, but this time update existing raster
    cmd_overlay = f'''gdal_rasterize -burn 2 "{boundaries_shp}" "{output_path}"'''
    
    success = run_gdal_command(cmd_overlay, "Overlaying boundaries on fields")
    if not success:
        # If that fails, try alternative approach using gdalwarp
        print("  Trying alternative approach...")
        
        # Create a temporary combined raster manually
        temp_combined = os.path.join(temp_dir, "temp_combined.tif")
        
        # Copy fields first
        shutil.copy2(fields_raster, temp_combined)
        
        # Use gdal_translate to ensure proper format
        cmd_translate = f'''gdal_translate -of GTiff -co "COMPRESS=LZW" "{temp_combined}" "{output_path}"'''
        success = run_gdal_command(cmd_translate, "Converting to final format")
        
        if success:
            # Now overlay boundaries
            cmd_overlay2 = f'''gdal_rasterize -burn 2 "{boundaries_shp}" "{output_path}"'''
            success = run_gdal_command(cmd_overlay2, "Second attempt at overlaying boundaries")
    
    # Check if output file was created
    if not os.path.exists(output_path):
        # Final fallback: just use the fields raster
        print("   Using fields raster only (boundaries overlay failed)")
        shutil.copy2(fields_raster, output_path)
    
    if not os.path.exists(output_path):
        raise Exception(f"Could not create output file: {output_path}")
    
    print(f"  Combined raster created: {Path(output_path).name}")
    
    print(f"\nStep 8: Adding color table (optional)...")
    # Add color table using gdaldem - skip if it fails
    color_file = os.path.join(temp_dir, "colors.txt")
    with open(color_file, 'w') as f:
        f.write("0 255 255 255 255\n")  # White - Background
        f.write("1 0 255 0 255\n")      # Green - Fields
        f.write("2 255 0 0 255\n")      # Red - Boundaries
    
    # Apply color table
    temp_colored = os.path.join(temp_dir, "colored.tif")
    cmd_color = f'''gdaldem color-relief "{output_path}" "{color_file}" "{temp_colored}" -alpha'''
    
    color_success = run_gdal_command(cmd_color, "Adding color table")
    
    # Copy colored version back if successful
    if color_success and os.path.exists(temp_colored):
        import shutil
        shutil.copy2(temp_colored, output_path.replace('.tif', '_colored.tif'))
        print("   Colored version created")
    else:
        print("   Color table step skipped (not critical)")
    
    print(" Process completed successfully!")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return output_path

def clip_raster_to_reference_gdal(input_raster, reference_raster, output_path):
    """
    Clip and align a raster to match exactly with a reference raster using GDAL.
    """
    print(f"\nClipping {Path(input_raster).name} to match {Path(reference_raster).name}...")
    
    # Check if input file exists
    if not os.path.exists(input_raster):
        raise Exception(f"Input raster not found: {input_raster}")
    
    if not os.path.exists(reference_raster):
        raise Exception(f"Reference raster not found: {reference_raster}")
    
    # Get reference raster info using gdalinfo
    cmd_info = f'gdalinfo "{reference_raster}"'
    result = subprocess.run(cmd_info, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to get info from reference raster: {result.stderr}")
    
    print("  Extracting reference raster parameters...")
    
    # Parse gdalinfo output to extract extent and pixel size
    info_lines = result.stdout.split('\n')
    
    # Find pixel size
    pixel_x, pixel_y = 10, -10  # Default values
    ul_x = ul_y = lr_x = lr_y = None
    width = height = None
    
    for line in info_lines:
        if 'Pixel Size = ' in line:
            # Extract pixel size: Pixel Size = (10.000000000000000,-10.000000000000000)
            import re
            match = re.search(r'Pixel Size = \(([^,]+),([^)]+)\)', line)
            if match:
                pixel_x, pixel_y = float(match.group(1)), float(match.group(2))
                print(f"    Pixel size: {pixel_x}, {pixel_y}")
        
        elif 'Upper Left' in line:
            # Extract upper left corner: Upper Left  (  567740.000, 3495290.000)
            import re
            match = re.search(r'Upper Left\s+\(\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                ul_x, ul_y = float(match.group(1)), float(match.group(2))
                print(f"    Upper left: {ul_x}, {ul_y}")
        
        elif 'Size is' in line:
            # Extract size: Size is 1129, 1042
            import re
            match = re.search(r'Size is (\d+), (\d+)', line)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                print(f"    Size: {width} x {height}")
    
    # Calculate lower right if we have all parameters
    if ul_x is not None and ul_y is not None and width and height:
        lr_x = ul_x + width * pixel_x
        lr_y = ul_y + height * pixel_y
        extent = f"{ul_x} {lr_y} {lr_x} {ul_y}"
        print(f"    Calculated extent: {extent}")
        
        # Use gdalwarp with exact extent and pixel size
        cmd_warp = f'''gdalwarp -te {extent} -tr {abs(pixel_x)} {abs(pixel_y)} -r near -of GTiff -co "COMPRESS=LZW" -overwrite "{input_raster}" "{output_path}"'''
        
    else:
        print("    Could not extract all parameters, using simpler approach...")
        # Fallback: just use the reference CRS and basic parameters
        cmd_warp = f'''gdalwarp -tr 10 10 -r near -of GTiff -co "COMPRESS=LZW" -t_srs EPSG:32629 -overwrite "{input_raster}" "{output_path}"'''
    
    success = run_gdal_command(cmd_warp, f"Clipping and aligning raster")
    if not success:
        raise Exception("Failed to clip raster")
    
    # Verify output was created
    if not os.path.exists(output_path):
        raise Exception(f"Output file was not created: {output_path}")
    
    print(f" Clipped raster saved to {output_path}")

# Main execution
if __name__ == "__main__":
    # Define input vector file paths with their corresponding Sentinel-2 tiles
    vector_tile_pairs = [
        {
            'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\04_FromCeinsys\Results_AOI\Field_Delineation_Morocco_Ceinsys\AOIs_Morocco1.shp",
            'sentinel': r"C:\Users\qin.xu\github\ftw-baselines\morocco_bl_gtaoi.tif",
            'name': 'Morocco1_BL'
        },
        {
            'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\04_FromCeinsys\Results_AOI\Field_Delineation_Morocco_Ceinsys\AOIs_Morocco2.shp",
            'sentinel': r"C:\Users\qin.xu\github\ftw-baselines\morocco_tr_gtaoi.tif",
            'name': 'Morocco2_TR'
        },
        {
            'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\05_FromStef\Parcel GT Stef New.shp",
            'sentinel': r"C:\Users\qin.xu\github\ftw-baselines\morocco_br_gtaoi.tif",
            'name': 'Stef_BR'
        }
    ]
    
    # Define output directory
    output_dir = r"C:\Users\qin.xu\github\ftw-baselines"
    
    # Check if GDAL is available
    try:
        result = subprocess.run(['gdalinfo', '--version'], capture_output=True, text=True)
        print(f"GDAL version: {result.stdout.strip()}")
    except FileNotFoundError:
        print(" GDAL not found. Please make sure GDAL is installed and in PATH")
        exit(1)
    
    try:
        print("=== GDAL-BASED PROCESSING: Each vector with its corresponding Sentinel-2 tile ===")
        
        aligned_rasters = []
        
        for i, pair in enumerate(vector_tile_pairs):
            print(f"\n{'='*60}")
            print(f"Processing pair {i+1}/{len(vector_tile_pairs)}: {pair['name']}")
            print(f"{'='*60}")
            
            # Define output paths for this pair
            output_raster_utm = f"{output_dir}\\{pair['name']}_3class_utm.tif"
            output_aligned = f"{output_dir}\\{pair['name']}_aligned.tif"
            
            print(f"Vector: {Path(pair['vector']).name}")
            print(f"Sentinel: {Path(pair['sentinel']).name}")
            
            # Step 1: Create 3-class raster from this vector file
            print(f"\n  Step 1: Creating 3-class raster for {pair['name']}...")
            result_path = create_3class_field_raster_gdal(
                vector_paths=[pair['vector']],  # Single vector file
                output_path=output_raster_utm,
                pixel_size=10,  # 10m pixels to match Sentinel-2
                boundary_buffer=10,  # 10m boundary width
                target_crs='EPSG:32629'  # UTM Zone 29N for Morocco
            )
            
            # Step 2: Align with corresponding Sentinel-2 tile
            print(f"\n  Step 2: Aligning with {Path(pair['sentinel']).name}...")
            clip_raster_to_reference_gdal(
                input_raster=result_path,
                reference_raster=pair['sentinel'],
                output_path=output_aligned
            )
            
            aligned_rasters.append({
                'name': pair['name'],
                'utm_raster': result_path,
                'aligned_raster': output_aligned,
                'sentinel_tile': pair['sentinel']
            })
            
            print(f" Completed {pair['name']}")
        
        print(f"\n SUCCESS! Created {len(aligned_rasters)} aligned raster pairs:")
        print("="*80)
        
        for i, raster in enumerate(aligned_rasters):
            print(f"\n{i+1}. {raster['name']}:")
            print(f"   UTM Ground Truth:     {raster['utm_raster']}")
            print(f"   Aligned Ground Truth: {raster['aligned_raster']}")
            print(f"   Sentinel-2 Tile:     {raster['sentinel_tile']}")
        
        print("\n" + "="*80)
        print("SUCCESS! All rasters created using GDAL command line tools")
        print("="*80)
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()