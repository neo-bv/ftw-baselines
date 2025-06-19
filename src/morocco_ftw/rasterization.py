#!/usr/bin/env python3
"""
Rasterization script for Morocco shapefiles
Converts shapefiles to raster format using GDAL/OGR
"""

import os
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import argparse
import logging

# Try to import rasterio, fall back to alternative method if it fails
try:
    import rasterio
    from rasterio import features
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping
    RASTERIO_AVAILABLE = True
    print("Using rasterio for rasterization")
except ImportError as e:
    print(f"Rasterio import failed: {e}")
    print("Falling back to alternative rasterization method...")
    RASTERIO_AVAILABLE = False
    try:
        from geocube.api.core import make_geocube
        GEOCUBE_AVAILABLE = True
        print("Using geocube for rasterization")
    except ImportError:
        GEOCUBE_AVAILABLE = False
        print("Geocube not available either. Please install: pip install geocube")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rasterize_shapefile_rasterio(shapefile_path, output_path, pixel_size=10, attribute_field=None, burn_value=1):
    """
    Rasterize a shapefile using rasterio
    """
    try:
        # Read shapefile
        logger.info(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.empty:
            logger.warning(f"Shapefile {shapefile_path} is empty!")
            return False
            
        logger.info(f"Loaded {len(gdf)} features from {shapefile_path}")
        logger.info(f"CRS: {gdf.crs}")
        
        # Get bounds
        bounds = gdf.total_bounds
        logger.info(f"Bounds: {bounds}")
        
        # Calculate raster dimensions
        width = int((bounds[2] - bounds[0]) / pixel_size)
        height = int((bounds[3] - bounds[1]) / pixel_size)
        
        logger.info(f"Output raster dimensions: {width} x {height}")
        
        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        # Prepare geometries and values for rasterization
        if attribute_field and attribute_field in gdf.columns:
            logger.info(f"Using attribute field '{attribute_field}' for burn values")
            shapes = [(mapping(geom), value) for geom, value in 
                     zip(gdf.geometry, gdf[attribute_field]) if geom is not None]
        else:
            logger.info(f"Using burn value: {burn_value}")
            shapes = [(mapping(geom), burn_value) for geom in gdf.geometry if geom is not None]
        
        if not shapes:
            logger.warning("No valid geometries found for rasterization!")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Rasterize
        logger.info(f"Rasterizing to: {output_path}")
        raster = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint16
        )
        
        # Write to file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(raster, 1)
        
        logger.info(f"Successfully created raster: {output_path}")
        logger.info(f"Raster stats - Min: {raster.min()}, Max: {raster.max()}, Non-zero pixels: {np.count_nonzero(raster)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {shapefile_path}: {str(e)}")
        return False

def rasterize_shapefile_geocube(shapefile_path, output_path, pixel_size=10, attribute_field=None, burn_value=1):
    """
    Rasterize a shapefile using geocube (alternative method)
    """
    try:
        # Read shapefile
        logger.info(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.empty:
            logger.warning(f"Shapefile {shapefile_path} is empty!")
            return False
            
        logger.info(f"Loaded {len(gdf)} features from {shapefile_path}")
        logger.info(f"CRS: {gdf.crs}")
        
        # Add burn value column if using default value
        if not attribute_field or attribute_field not in gdf.columns:
            gdf['burn_value'] = burn_value
            attribute_field = 'burn_value'
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use geocube to rasterize
        logger.info(f"Rasterizing to: {output_path}")
        out_grid = make_geocube(
            vector_data=gdf,
            measurements=[attribute_field],
            resolution=(-pixel_size, pixel_size),
            fill=0
        )
        
        # Save to file
        out_grid[attribute_field].rio.to_raster(output_path, compress='lzw')
        
        logger.info(f"Successfully created raster: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {shapefile_path}: {str(e)}")
        return False

def rasterize_shapefile_gdal_fallback(shapefile_path, output_path, pixel_size=10, attribute_field=None, burn_value=1):
    """
    Rasterize using GDAL command line as fallback
    """
    try:
        import subprocess
        
        # Read shapefile to get bounds and CRS
        logger.info(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.empty:
            logger.warning(f"Shapefile {shapefile_path} is empty!")
            return False
        
        logger.info(f"Loaded {len(gdf)} features from {shapefile_path}")
        
        # Get bounds
        bounds = gdf.total_bounds
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build gdal_rasterize command
        cmd = [
            'gdal_rasterize',
            '-burn', str(burn_value),
            '-tr', str(pixel_size), str(pixel_size),
            '-te', str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]),
            '-ot', 'UInt16',
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',
            str(shapefile_path),
            str(output_path)
        ]
        
        if attribute_field:
            cmd = cmd[:-2] + ['-a', attribute_field] + cmd[-2:]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully created raster: {output_path}")
            return True
        else:
            logger.error(f"GDAL command failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error with GDAL fallback: {str(e)}")
        return False

def rasterize_shapefile(shapefile_path, output_path, pixel_size=10, attribute_field=None, burn_value=1):
    """
    Rasterize a shapefile using the best available method
    """
    if RASTERIO_AVAILABLE:
        return rasterize_shapefile_rasterio(shapefile_path, output_path, pixel_size, attribute_field, burn_value)
    elif GEOCUBE_AVAILABLE:
        return rasterize_shapefile_geocube(shapefile_path, output_path, pixel_size, attribute_field, burn_value)
    else:
        logger.info("Trying GDAL command line as fallback...")
        return rasterize_shapefile_gdal_fallback(shapefile_path, output_path, pixel_size, attribute_field, burn_value)

def main():
    """Main function to rasterize all Morocco shapefiles"""
    
    # Define input shapefiles
    base_path = Path(r"C:\Users\qin.xu\github\ftw-baselines\data\morocco")
    shapefiles = {
        "parcel_gt": base_path / "Parcel GT Stef New.shp",
        "aoi_morocco1": base_path / "AOIs_Morocco1.shp",
        "aoi_morocco2": base_path / "AOIs_Morocco2.shp"
    }
    
    # Define output directory
    output_dir = Path(r"C:\Users\qin.xu\github\ftw-baselines\data\morocco\raster")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rasterization parameters for each file
    raster_configs = {
        "parcel_gt": {
            "pixel_size": 10,  # 10 meter pixels
            "attribute_field": None,  # You can specify a field name here if needed
            "burn_value": 1
        },
        "aoi_morocco1": {
            "pixel_size": 10,
            "attribute_field": None,
            "burn_value": 1
        },
        "aoi_morocco2": {
            "pixel_size": 10,
            "attribute_field": None,
            "burn_value": 1
        }
    }
    
    # Process each shapefile
    results = {}
    for name, shapefile_path in shapefiles.items():
        if not shapefile_path.exists():
            logger.error(f"Shapefile not found: {shapefile_path}")
            results[name] = False
            continue
        
        # Define output path
        output_path = output_dir / f"{name}.tif"
        
        # Get configuration
        config = raster_configs[name]
        
        # Rasterize
        success = rasterize_shapefile(
            str(shapefile_path),
            str(output_path),
            pixel_size=config["pixel_size"],
            attribute_field=config["attribute_field"],
            burn_value=config["burn_value"]
        )
        
        results[name] = success
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("RASTERIZATION SUMMARY")
    logger.info("="*50)
    
    success_count = 0
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{name}: {status}")
        if success:
            success_count += 1
    
    logger.info(f"\nProcessed {success_count}/{len(results)} files successfully")
    
    if success_count == len(results):
        logger.info("All files processed successfully!")
        return 0
    else:
        logger.warning("Some files failed to process. Check the logs above.")
        return 1

def rasterize_single_file():
    """Command line interface for rasterizing a single file"""
    parser = argparse.ArgumentParser(description='Rasterize a single shapefile')
    parser.add_argument('input_shapefile', help='Path to input shapefile')
    parser.add_argument('output_raster', help='Path to output raster file')
    parser.add_argument('--pixel-size', type=float, default=10, help='Pixel size (default: 10)')
    parser.add_argument('--attribute-field', help='Attribute field to use for burn values')
    parser.add_argument('--burn-value', type=int, default=1, help='Value to burn (default: 1)')
    
    args = parser.parse_args()
    
    success = rasterize_shapefile(
        args.input_shapefile,
        args.output_raster,
        pixel_size=args.pixel_size,
        attribute_field=args.attribute_field,
        burn_value=args.burn_value
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    # Check if command line arguments are provided for single file processing
    if len(sys.argv) > 1 and not sys.argv[1] in ['--help', '-h']:
        sys.exit(rasterize_single_file())
    else:
        # Process all Morocco files
        sys.exit(main())