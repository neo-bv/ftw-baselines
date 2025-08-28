#This script is used for transfer vector to 3 class raster GT
import geopandas as gpd
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tempfile
import shutil

BASE_PATH = Path(os.environ.get('FTW_BASE_PATH', Path(__file__).parent.parent.parent))
print(f"Using base path: {BASE_PATH}")

def run_gdal_command(cmd, description=""):
    """Run a GDAL command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"GDAL command failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def create_3class_field_raster_gdal(vector_path, output_path, pixel_size=10, boundary_buffer=10, target_crs='EPSG:32629'):
    """Create a 3-class raster using GDAL command line tools for filtered polygons"""
    
    # Load filtered vector file (no need to merge since it's already filtered)
    print(f"Loading filtered polygons from: {vector_path}")
    gdf = gpd.read_file(vector_path)
    
    # Convert to target CRS
    gdf_utm = gdf.to_crs(target_crs)
    print(f"Loaded {len(gdf_utm)} filtered polygons")
    
    # Fix invalid geometries
    gdf_utm.geometry = gdf_utm.geometry.buffer(0)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save filtered shapefile in UTM
    filtered_shp = os.path.join(temp_dir, "filtered_fields.shp")
    gdf_utm.to_file(filtered_shp)
    
    # Create boundary shapefile
    boundaries_gdf = gdf_utm.copy()
    boundary_lines = gdf_utm.geometry.boundary
    boundary_polygons = boundary_lines.buffer(boundary_buffer)
    boundaries_gdf.geometry = boundary_polygons
    
    boundaries_shp = os.path.join(temp_dir, "boundaries.shp")
    boundaries_gdf.to_file(boundaries_shp)
    
    # Get extent and calculate dimensions
    bounds = gdf_utm.total_bounds
    extent = f"{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}"
    
    # Create temporary raster files
    fields_raster = os.path.join(temp_dir, "fields.tif")
    
    # Rasterize fields (value = 1)
    cmd_fields = f'''gdal_rasterize -a_srs {target_crs} -te {extent} -tr {pixel_size} {pixel_size} -burn 1 -ot Byte -of GTiff "{filtered_shp}" "{fields_raster}"'''
    
    if not run_gdal_command(cmd_fields, "Rasterizing filtered fields"):
        raise Exception("Failed to rasterize filtered fields")
    
    # Copy fields raster as base
    shutil.copy2(fields_raster, output_path)
    
    # Overlay boundaries (value = 2)
    cmd_overlay = f'''gdal_rasterize -burn 2 "{boundaries_shp}" "{str(output_path)}"'''
    
    if not run_gdal_command(cmd_overlay, "Overlaying boundaries"):
        # Fallback approach
        cmd_translate = f'''gdal_translate -of GTiff -co "COMPRESS=LZW" "{fields_raster}" "{output_path}"'''
        if run_gdal_command(cmd_translate, "Creating output"):
            run_gdal_command(cmd_overlay, "Overlaying boundaries (retry)")
    
    # Check if output exists
    if not os.path.exists(str(output_path)):
        raise Exception(f"Could not create output file: {output_path}")
    
    # Optional: Add color table
    color_file = os.path.join(temp_dir, "colors.txt")
    with open(color_file, 'w') as f:
        f.write("0 255 255 255 255\n")  # White - Background
        f.write("1 0 255 0 255\n")      # Green - Fields
        f.write("2 255 0 0 255\n")      # Red - Boundaries
    
    temp_colored = os.path.join(temp_dir, "colored.tif")
    cmd_color = f'''gdaldem color-relief "{output_path}" "{color_file}" "{temp_colored}" -alpha'''
    
    if run_gdal_command(cmd_color, "Adding color table"):
        colored_output = output_path.replace('.tif', '_colored.tif')
        shutil.copy2(temp_colored, colored_output)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return output_path

def clip_raster_to_reference_gdal(input_raster, reference_raster, output_path):
    """Clip and align a raster to match exactly with a reference raster"""
    
    # Check input files exist
    if not os.path.exists(str(input_raster)):
        raise Exception(f"Input raster not found: {input_raster}")
    if not os.path.exists(str(reference_raster)):
        raise Exception(f"Reference raster not found: {reference_raster}")
    
    # Get reference raster info
    cmd_info = f'gdalinfo "{reference_raster}"'
    result = subprocess.run(cmd_info, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to get info from reference raster: {result.stderr}")
    
    # Parse gdalinfo output
    info_lines = result.stdout.split('\n')
    pixel_x, pixel_y = 10, -10  # Default values
    ul_x = ul_y = None
    width = height = None
    
    for line in info_lines:
        if 'Pixel Size = ' in line:
            import re
            match = re.search(r'Pixel Size = \(([^,]+),([^)]+)\)', line)
            if match:
                pixel_x, pixel_y = float(match.group(1)), float(match.group(2))
        
        elif 'Upper Left' in line:
            import re
            match = re.search(r'Upper Left\s+\(\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                ul_x, ul_y = float(match.group(1)), float(match.group(2))
        
        elif 'Size is' in line:
            import re
            match = re.search(r'Size is (\d+), (\d+)', line)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
    
    # Calculate extent and run gdalwarp
    if ul_x is not None and ul_y is not None and width and height:
        lr_x = ul_x + width * pixel_x
        lr_y = ul_y + height * pixel_y
        extent = f"{ul_x} {lr_y} {lr_x} {ul_y}"
        
        cmd_warp = f'''gdalwarp -te {extent} -tr {abs(pixel_x)} {abs(pixel_y)} -r near -of GTiff -co "COMPRESS=LZW" -overwrite "{input_raster}" "{output_path}"'''
    else:
        cmd_warp = f'''gdalwarp -tr 10 10 -r near -of GTiff -co "COMPRESS=LZW" -t_srs EPSG:32629 -overwrite "{input_raster}" "{output_path}"'''
    
    if not run_gdal_command(cmd_warp, "Clipping and aligning raster"):
        raise Exception("Failed to clip raster")
    
    if not os.path.exists(str(output_path)):
        raise Exception(f"Output file was not created: {output_path}")

if __name__ == "__main__":
    # Configuration - Updated to use filtered polygons
    
    # OLD CONFIGURATION (commented out):
    # vector_tile_pairs = [
    #     {
    #         'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\04_FromCeinsys\Results_AOI\Field_Delineation_Morocco_Ceinsys\AOIs_Morocco1.shp",
    #         'sentinel': BASE_PATH / "morocco_bl_gtaoi.tif",
    #         'name': 'Morocco1_BL'
    #     },
    #     {
    #         'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\04_FromCeinsys\Results_AOI\Field_Delineation_Morocco_Ceinsys\AOIs_Morocco2.shp",
    #         'sentinel': BASE_PATH / "morocco_tr_gtaoi.tif",
    #         'name': 'Morocco2_TR'
    #     },
    #     {
    #         'vector': r"S:\E043 Crop classification and field delineation\04_FieldDelineation\TrainingData_Assignment\05_FromStef\Parcel GT Stef New.shp",
    #         'sentinel': BASE_PATH / "morocco_br_gtaoi.tif",
    #         'name': 'Stef_BR'
    #     }
    # ]
    
    # NEW CONFIGURATION - Using filtered polygons
    filtered_polygons_path = BASE_PATH / "Output"/"Threshold_0p02"/"morocco_active_fields_th0p02.shp"
    
    # Reference Sentinel-2 tiles for alignment (keep all three for different AOI regions)
    reference_tiles = [
        {
            'sentinel': BASE_PATH / "morocco_bl_gtaoi.tif",
            'name': 'Morocco1_BL'
        },
        {
            'sentinel': BASE_PATH / "morocco_tr_gtaoi.tif",
            'name': 'Morocco2_TR'
        },
        {
            'sentinel': BASE_PATH / "morocco_br_gtaoi.tif",
            'name': 'Stef_BR'
        }
    ]
    
    output_dir = BASE_PATH
    
    # Check GDAL installation
    try:
        result = subprocess.run(['gdalinfo', '--version'], capture_output=True, text=True)
        print(f"GDAL version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("GDAL not found. Please install GDAL and add to PATH.")
        exit(1)
    
    # Check if filtered polygons file exists
    if not os.path.exists(str(filtered_polygons_path)):
        print(f"Error: Filtered polygons file not found: {filtered_polygons_path}")
        print("Please run the NDVI filtering script first to generate the filtered polygons.")
        exit(1)
    
    try:
        print("Processing filtered polygons to create 3-class rasters...")
        print(f"Using filtered polygons: {filtered_polygons_path}")
        
        aligned_rasters = []
        
        # Create base 3-class raster from filtered polygons
        base_output_utm = output_dir / "morocco_filtered_3class_utm.tif"
        print(f"\nCreating base 3-class raster...")
        
        result_path = create_3class_field_raster_gdal(
            vector_path=filtered_polygons_path,
            output_path=base_output_utm,
            pixel_size=10,
            boundary_buffer=10,
            target_crs='EPSG:32629'
        )
        
        print(f"Base raster created: {result_path}")
        
        # Align with each Sentinel-2 tile for different AOI regions
        for i, tile in enumerate(reference_tiles):
            print(f"\nAligning with {tile['name']} ({i+1}/{len(reference_tiles)})")
            
            # Check if reference tile exists
            if not os.path.exists(str(tile['sentinel'])):
                print(f"Warning: Reference tile not found: {tile['sentinel']}")
                continue
            
            # Define output path
            output_aligned = output_dir / f"morocco_filtered_3class_{tile['name']}_aligned.tif"
            
            # Align with Sentinel-2 tile
            clip_raster_to_reference_gdal(
                input_raster=result_path,
                reference_raster=tile['sentinel'],
                output_path=output_aligned
            )
            
            aligned_rasters.append({
                'name': tile['name'],
                'aligned_raster': output_aligned,
                'sentinel_tile': tile['sentinel']
            })
            
            print(f"Completed alignment for {tile['name']}")
        
        print(f"\nProcessing complete!")
        print(f"Base UTM raster: {result_path}")
        print(f"Created {len(aligned_rasters)} aligned raster(s):")
        
        for i, raster in enumerate(aligned_rasters):
            print(f"{i+1}. {raster['name']}")
            print(f"   Aligned raster: {raster['aligned_raster']}")
            print(f"   Reference tile: {raster['sentinel_tile']}")
        
        print(f"\nRaster classes:")
        print(f"  0 = Background (white)")
        print(f"  1 = Field interior (green)")
        print(f"  2 = Field boundaries (red)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()