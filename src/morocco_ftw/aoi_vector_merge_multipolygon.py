#!/usr/bin/env python3
"""
AOI Vector Masking and Merging Script (Updated for Parquet Support)
Clips multiple vector files (GPKG or Parquet) with all polygons in an AOI shapefile and merges them.
"""

import os
import sys
from pathlib import Path
import logging
import traceback
from pathlib import Path  # Add this import if not already there
BASE_PATH = Path(os.environ.get('FTW_BASE_PATH', Path(__file__).parent.parent.parent))
print(f"Using base path: {BASE_PATH}")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        #logging.FileHandler('aoi_vector_merge.log')
        logging.FileHandler(str(BASE_PATH / 'aoi_vector_merge.log'))
    ]
)
logger = logging.getLogger(__name__)

# Check imports
try:
    from osgeo import ogr, osr
    import geopandas as gpd
    import pandas as pd
    logger.info("GDAL and GeoPandas imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    print(f"ERROR: Failed to import required libraries: {e}")
    print("Please install required packages:")
    print("conda install gdal geopandas pandas pyarrow")
    print("or")
    print("pip install GDAL geopandas pandas pyarrow")
    sys.exit(1)

def detect_file_format(file_path):
    """Detect if file is parquet or vector format"""
    extension = Path(file_path).suffix.lower()
    if extension == '.parquet':
        return 'parquet'
    elif extension in ['.gpkg', '.shp', '.geojson']:
        return 'vector'
    else:
        logger.warning(f"Unknown file format: {extension}")
        return 'unknown'

def read_vector_file(file_path):
    """Read vector file (parquet or traditional vector format) using GeoPandas"""
    try:
        file_format = detect_file_format(file_path)
        
        if file_format == 'parquet':
            logger.info(f"Reading parquet file: {file_path}")
            gdf = gpd.read_parquet(file_path)
        elif file_format == 'vector':
            logger.info(f"Reading vector file: {file_path}")
            gdf = gpd.read_file(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        logger.info(f"Read {len(gdf)} features with CRS: {gdf.crs}")
        return gdf
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

def write_vector_file(gdf, output_path, file_format='gpkg'):
    """Write GeoDataFrame to file (parquet or vector format)"""
    try:
        if file_format == 'parquet' or output_path.endswith('.parquet'):
            logger.info(f"Writing parquet file: {output_path}")
            gdf.to_parquet(output_path)
        else:
            logger.info(f"Writing vector file: {output_path}")
            gdf.to_file(output_path, driver='GPKG')
        
        logger.info(f"Successfully wrote {len(gdf)} features to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing file {output_path}: {str(e)}")
        return False

def get_layer_crs(dataset, layer_index=0):
    """Get the CRS of a layer as an EPSG code"""
    layer = dataset.GetLayerByIndex(layer_index)
    srs = layer.GetSpatialRef()
    if srs:
        auth_name = srs.GetAuthorityName(None)
        auth_code = srs.GetAuthorityCode(None)
        if auth_name and auth_code:
            return f"{auth_name}:{auth_code}"
        else:
            return srs.GetName()
    return None

def transform_geometry(geom, source_srs, target_srs):
    """Transform geometry from source to target coordinate system"""
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    geom_copy = geom.Clone()
    geom_copy.Transform(transform)
    return geom_copy

def create_combined_aoi_geometry(aoi_layer, target_srs):
    """Create a combined geometry from all polygons in the AOI layer"""
    aoi_srs = aoi_layer.GetSpatialRef()
    combined_geom = None
    polygon_count = 0
    
    logger.info("Processing AOI polygons...")
    
    aoi_layer.ResetReading()
    
    for feature in aoi_layer:
        geom = feature.GetGeometryRef()
        if geom:
            # Transform geometry if needed
            if aoi_srs and target_srs and not aoi_srs.IsSame(target_srs):
                geom_transformed = transform_geometry(geom, aoi_srs, target_srs)
            else:
                geom_transformed = geom.Clone()
            
            # Combine with existing geometry
            if combined_geom is None:
                combined_geom = geom_transformed
            else:
                combined_geom = combined_geom.Union(geom_transformed)
            
            polygon_count += 1
            logger.info(f"Added AOI polygon {polygon_count}")
    
    logger.info(f"Combined {polygon_count} AOI polygons")
    
    if combined_geom:
        combined_area = combined_geom.GetArea()
        logger.info(f"Total AOI area: {combined_area:.2f} square units")
        
        envelope = combined_geom.GetEnvelope()
        logger.info(f"AOI extent: ({envelope[0]:.2f}, {envelope[2]:.2f}) to ({envelope[1]:.2f}, {envelope[3]:.2f})")
    
    return combined_geom

def clip_vector_with_aoi_geopandas(input_vector, aoi_shapefile, output_vector):
    """Clip a vector file using all polygons in an AOI shapefile using GeoPandas"""
    try:
        logger.info(f"Starting clip operation: {input_vector}")
        
        # Read AOI shapefile
        aoi_gdf = gpd.read_file(aoi_shapefile)
        logger.info(f"AOI CRS: {aoi_gdf.crs}")
        logger.info(f"AOI has {len(aoi_gdf)} polygons")
        
        # Read input vector (parquet or vector format)
        input_gdf = read_vector_file(input_vector)
        if input_gdf is None:
            return False
        
        logger.info(f"Input CRS: {input_gdf.crs}")
        original_count = len(input_gdf)
        
        # Reproject AOI to match input CRS if needed
        if aoi_gdf.crs != input_gdf.crs:
            logger.info(f"Reprojecting AOI from {aoi_gdf.crs} to {input_gdf.crs}")
            aoi_gdf = aoi_gdf.to_crs(input_gdf.crs)
        
        # Create combined AOI geometry
        combined_aoi_geom = aoi_gdf.unary_union
        logger.info(f"Combined AOI area: {combined_aoi_geom.area:.2f} square units")
        
        # Clip input vector with combined AOI
        logger.info(f"Clipping {original_count} features against combined AOI...")
        
        # Use spatial intersection
        clipped_gdf = gpd.clip(input_gdf, aoi_gdf)
        clipped_count = len(clipped_gdf)
        
        logger.info(f"Clipped {clipped_count} features out of {original_count} from {input_vector}")
        
        if clipped_count > 0:
            # Write output (determine format from extension)
            output_format = 'parquet' if output_vector.endswith('.parquet') else 'gpkg'
            success = write_vector_file(clipped_gdf, output_vector, output_format)
            return success
        else:
            logger.warning(f"No features intersected with AOI for {input_vector}")
            return False
        
    except Exception as e:
        logger.error(f"Error clipping {input_vector}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def merge_vectors_geopandas(input_vectors, output_vector):
    """Merge multiple vector files into a single output file using GeoPandas"""
    try:
        logger.info(f"Starting merge of {len(input_vectors)} files...")
        
        gdfs = []
        total_features = 0
        
        for input_vector in input_vectors:
            if not os.path.exists(input_vector):
                logger.warning(f"Input vector does not exist: {input_vector}")
                continue
            
            gdf = read_vector_file(input_vector)
            if gdf is None or len(gdf) == 0:
                logger.warning(f"Could not read or empty file: {input_vector}")
                continue
            
            logger.info(f"Read {len(gdf)} features from {input_vector}")
            gdfs.append(gdf)
            total_features += len(gdf)
        
        if not gdfs:
            logger.error("No valid input vectors found")
            return False
        
        # Ensure all GeoDataFrames have the same CRS
        target_crs = gdfs[0].crs
        for i, gdf in enumerate(gdfs[1:], 1):
            if gdf.crs != target_crs:
                logger.info(f"Reprojecting file {i} from {gdf.crs} to {target_crs}")
                gdfs[i] = gdf.to_crs(target_crs)
        
        # Concatenate all GeoDataFrames
        logger.info("Concatenating GeoDataFrames...")
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        merged_gdf.crs = target_crs
        
        logger.info(f"Total merged features: {len(merged_gdf)}")
        
        # Write output
        output_format = 'parquet' if output_vector.endswith('.parquet') else 'gpkg'
        success = write_vector_file(merged_gdf, output_vector, output_format)
        
        return success
        
    except Exception as e:
        logger.error(f"Error merging vectors: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function to process AOI masking and vector merging"""
    
    try:
        logger.info("Starting AOI Vector Masking and Merging Process")
        
        # Get paths
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        #base_path = os.path.dirname(os.path.dirname(script_dir))
        
        #logger.info(f"Script directory: {script_dir}")
        #logger.info(f"Base directory: {base_path}")
        
        # Input files - Updated for parquet files
        #aoi_shapefile = os.path.join(base_path, "SECTEURS.shp")
        aoi_shapefile = BASE_PATH / "SECTEURS.shp"

        # Parquet files (update these paths as needed)
        # input_vectors = [
        #     r"C:\Users\qin.xu\github\ftw-baselines\morocco_mosaic_morocco_CCBY_filtered0.02.parquet",
        #     r"C:\Users\qin.xu\github\ftw-baselines\morocco_tr_aoi_morocco_CCBY_filtered0.02.parquet",
        #     r"C:\Users\qin.xu\github\ftw-baselines\morocco_mid_mosaic_morocco_CCBY_filtered0.02.parquet"
        # ]
        input_vectors = [
        BASE_PATH / "morocco_mid_mosaic_filtered_inference_boundaries.gpkg",
        BASE_PATH / "morocco_mosaic_filtered_inference_boundaries.gpkg",
        BASE_PATH / "morocco_tr_aoi_filtered_inference_boundaries.gpkg"
    ]
        
        #temp_dir = os.path.join(base_path, "temp_clipped_morocco_filtered0.02_parquet")
        #final_output = os.path.join(base_path, "morocco_merged_morocco_CCBY_filtered0.02_ALL_AOI.parquet")  # Changed to parquet
        temp_dir = BASE_PATH / "temp_clipped_morocco_filtered"
        final_output = BASE_PATH / "morocco_merged_morocco_filtered_ALL_AOI.gpkg"

        # Log paths
        logger.info(f"AOI shapefile: {aoi_shapefile}")
        logger.info(f"Input vectors: {input_vectors}")
        logger.info(f"Temp directory: {temp_dir}")
        logger.info(f"Final output: {final_output}")
        
        # Create temporary directory
        logger.info("Creating temporary directory...")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Check AOI shapefile
        logger.info("Checking AOI shapefile...")
        if not os.path.exists(aoi_shapefile):
            logger.error(f"AOI shapefile not found: {aoi_shapefile}")
            return False
        else:
            aoi_ds = ogr.Open(aoi_shapefile, 0)
            if aoi_ds:
                aoi_layer = aoi_ds.GetLayer()
                aoi_count = aoi_layer.GetFeatureCount()
                logger.info(f"AOI shapefile found with {aoi_count} polygons: {aoi_shapefile}")
                del aoi_ds
            else:
                logger.error(f"Could not open AOI shapefile: {aoi_shapefile}")
                return False
        
        # Check input files
        for vector in input_vectors:
            if os.path.exists(vector):
                logger.info(f"Input vector found: {vector}")
            else:
                logger.warning(f"Input vector not found: {vector}")
        
        # Clip each vector with AOI polygons
        logger.info("Starting AOI clipping process")
        clipped_vectors = []
        
        for i, input_vector in enumerate(input_vectors):
            if not os.path.exists(input_vector):
                logger.warning(f"Input vector not found: {input_vector}")
                continue
                
            base_name = Path(input_vector).stem
            # Keep clipped files as parquet if input is parquet
            clipped_output = os.path.join(temp_dir, f"{base_name}_clipped_ALL_AOI.parquet")
            
            logger.info(f"Clipping {input_vector} with AOI polygons...")
            if clip_vector_with_aoi_geopandas(input_vector, aoi_shapefile, clipped_output):
                clipped_vectors.append(clipped_output)
                logger.info(f"Successfully clipped: {clipped_output}")
            else:
                logger.error(f"Failed to clip {input_vector}")
        
        if not clipped_vectors:
            logger.error("No vectors were successfully clipped")
            return False
        
        logger.info(f"Successfully clipped {len(clipped_vectors)} vectors")
        
        # Merge clipped vectors
        logger.info("Starting vector merging process")
        if merge_vectors_geopandas(clipped_vectors, final_output):
            logger.info(f"Successfully created merged output: {final_output}")
            
            # Verify output
            final_gdf = read_vector_file(final_output)
            if final_gdf is not None:
                logger.info(f"Final output contains {len(final_gdf)} features")
            
            return True
        else:
            logger.error("Failed to merge vectors")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in main(): {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    try:
        print("Starting AOI Vector Masking and Merging Script (Parquet Support)...")
        success = main()
        if success:
            print("Process completed successfully")
            logger.info("Process completed successfully")
            sys.exit(0)
        else:
            print("Process failed - check log file for details")
            logger.error("Process failed")
            sys.exit(1)
    except Exception as e:
        print(f"Critical error: {str(e)}")
        logger.error(f"Critical error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)