#!/usr/bin/env python3
"""
AOI Vector Masking and Merging Script - MULTI-POLYGON AOI VERSION
Clips multiple vector files with ALL polygons in an AOI shapefile and merges them.
Handles CRS transformations automatically.
"""

import os
import sys
from pathlib import Path
import logging
import traceback

# Set up logging to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('aoi_vector_merge_multipolygon.log')
    ]
)
logger = logging.getLogger(__name__)

# Check GDAL import
try:
    from osgeo import ogr, osr
    logger.info("GDAL imported successfully")
except ImportError as e:
    logger.error(f"Failed to import GDAL: {e}")
    print(f"ERROR: Failed to import GDAL: {e}")
    print("Please install GDAL: conda install gdal or pip install GDAL")
    sys.exit(1)

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
    """
    Create a combined geometry from ALL polygons in the AOI layer.
    Transform to target CRS if needed.
    """
    aoi_srs = aoi_layer.GetSpatialRef()
    combined_geom = None
    polygon_count = 0
    
    logger.info("Processing ALL AOI polygons...")
    
    # Reset reading to start from beginning
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
                # Union with existing geometry
                combined_geom = combined_geom.Union(geom_transformed)
            
            polygon_count += 1
            logger.info(f"Added AOI polygon {polygon_count}")
    
    logger.info(f"Combined {polygon_count} AOI polygons into single geometry")
    
    if combined_geom:
        combined_area = combined_geom.GetArea()
        logger.info(f"Total AOI area: {combined_area:.2f} square units")
        
        # Log extent
        envelope = combined_geom.GetEnvelope()
        logger.info(f"Combined AOI extent: ({envelope[0]:.2f}, {envelope[2]:.2f}) to ({envelope[1]:.2f}, {envelope[3]:.2f})")
    
    return combined_geom

def clip_vector_with_aoi(input_vector, aoi_shapefile, output_vector):
    """
    Clip a vector file using ALL polygons in an AOI shapefile with automatic CRS handling.
    
    Args:
        input_vector (str): Path to input vector file
        aoi_shapefile (str): Path to AOI shapefile
        output_vector (str): Path to output clipped vector file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting clip operation: {input_vector}")
        
        # Open AOI shapefile
        aoi_ds = ogr.Open(aoi_shapefile, 0)
        if not aoi_ds:
            logger.error(f"Could not open AOI file: {aoi_shapefile}")
            return False
        
        aoi_layer = aoi_ds.GetLayer()
        aoi_srs = aoi_layer.GetSpatialRef()
        aoi_crs = get_layer_crs(aoi_ds)
        aoi_feature_count = aoi_layer.GetFeatureCount()
        logger.info(f"AOI CRS: {aoi_crs}")
        logger.info(f"AOI has {aoi_feature_count} polygons")
        
        # Open input vector
        input_ds = ogr.Open(input_vector, 0)
        if not input_ds:
            logger.error(f"Could not open input vector: {input_vector}")
            return False
        
        input_layer = input_ds.GetLayer()
        input_srs = input_layer.GetSpatialRef()
        input_crs = get_layer_crs(input_ds)
        logger.info(f"Input CRS: {input_crs}")
        
        # Determine target CRS (use input vector's CRS)
        target_srs = input_srs
        target_crs = input_crs
        logger.info(f"Target CRS: {target_crs}")
        
        # Create combined AOI geometry from ALL polygons
        combined_aoi_geom = create_combined_aoi_geometry(aoi_layer, target_srs)
        if not combined_aoi_geom:
            logger.error("Failed to create combined AOI geometry")
            return False
        
        # Create output vector
        driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(output_vector):
            driver.DeleteDataSource(output_vector)
        
        output_ds = driver.CreateDataSource(output_vector)
        if not output_ds:
            logger.error(f"Could not create output vector: {output_vector}")
            return False
        
        # Get geometry type from input layer
        geom_type = input_layer.GetGeomType()
        
        # Create output layer with target CRS
        output_layer = output_ds.CreateLayer("clipped", target_srs, geom_type)
        
        # Copy field definitions from input layer
        input_defn = input_layer.GetLayerDefn()
        for i in range(input_defn.GetFieldCount()):
            field_defn = input_defn.GetFieldDefn(i)
            output_layer.CreateField(field_defn)
        
        # Reset reading for input layer
        input_layer.ResetReading()
        
        # Process each feature in input layer
        clipped_count = 0
        total_count = input_layer.GetFeatureCount()
        logger.info(f"Processing {total_count} features against combined AOI...")
        
        for feature in input_layer:
            geom = feature.GetGeometryRef()
            if geom and combined_aoi_geom.Intersects(geom):
                # Clip geometry with combined AOI
                clipped_geom = geom.Intersection(combined_aoi_geom)
                
                if clipped_geom and not clipped_geom.IsEmpty():
                    # Create new feature
                    new_feature = ogr.Feature(output_layer.GetLayerDefn())
                    new_feature.SetGeometry(clipped_geom)
                    
                    # Copy attributes
                    for i in range(input_defn.GetFieldCount()):
                        new_feature.SetField(i, feature.GetField(i))
                    
                    # Add feature to output layer
                    output_layer.CreateFeature(new_feature)
                    clipped_count += 1
                    
                    if clipped_count == 1:
                        logger.info(f"First intersection found! Clipped area: {clipped_geom.GetArea():.2f}")
                    elif clipped_count % 100 == 0:
                        logger.info(f"Processed {clipped_count} intersections...")
        
        logger.info(f"Clipped {clipped_count} features out of {total_count} from {input_vector}")
        
        # Clean up
        del output_layer, output_ds, input_ds, aoi_ds
        
        return clipped_count > 0  # Return False if no features were clipped
        
    except Exception as e:
        logger.error(f"Error clipping {input_vector}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def merge_vectors(input_vectors, output_vector):
    """
    Merge multiple vector files into a single output file.
    
    Args:
        input_vectors (list): List of input vector file paths
        output_vector (str): Path to output merged vector file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting merge of {len(input_vectors)} files...")
        
        # Create output vector
        driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(output_vector):
            driver.DeleteDataSource(output_vector)
        
        output_ds = driver.CreateDataSource(output_vector)
        if not output_ds:
            logger.error(f"Could not create output vector: {output_vector}")
            return False
        
        output_layer = None
        total_features = 0
        
        for i, input_vector in enumerate(input_vectors):
            if not os.path.exists(input_vector):
                logger.warning(f"Input vector does not exist: {input_vector}")
                continue
                
            # Open input vector
            input_ds = ogr.Open(input_vector, 0)
            if not input_ds:
                logger.warning(f"Could not open input vector: {input_vector}")
                continue
            
            input_layer = input_ds.GetLayer()
            input_feature_count = input_layer.GetFeatureCount()
            
            if input_feature_count == 0:
                logger.warning(f"Input vector has no features: {input_vector}")
                del input_ds
                continue
            
            # Create output layer from first valid input
            if output_layer is None:
                srs = input_layer.GetSpatialRef()
                geom_type = input_layer.GetGeomType()
                output_layer = output_ds.CreateLayer("merged", srs, geom_type)
                
                # Copy field definitions from first layer
                input_defn = input_layer.GetLayerDefn()
                for j in range(input_defn.GetFieldCount()):
                    field_defn = input_defn.GetFieldDefn(j)
                    output_layer.CreateField(field_defn)
                
                logger.info(f"Created output layer with CRS from {input_vector}")
            
            # Copy features
            feature_count = 0
            input_defn = input_layer.GetLayerDefn()
            
            for feature in input_layer:
                geom = feature.GetGeometryRef()
                if geom:
                    new_feature = ogr.Feature(output_layer.GetLayerDefn())
                    new_feature.SetGeometry(geom)
                    
                    # Copy attributes (handle potential field mismatches)
                    output_defn = output_layer.GetLayerDefn()
                    for j in range(min(input_defn.GetFieldCount(), output_defn.GetFieldCount())):
                        field_name = input_defn.GetFieldDefn(j).GetName()
                        if output_defn.GetFieldIndex(field_name) >= 0:
                            new_feature.SetField(field_name, feature.GetField(j))
                    
                    output_layer.CreateFeature(new_feature)
                    feature_count += 1
            
            logger.info(f"Merged {feature_count} features from {input_vector}")
            total_features += feature_count
            
            del input_ds
        
        logger.info(f"Total merged features: {total_features}")
        
        # Clean up
        del output_layer, output_ds
        
        return total_features > 0
        
    except Exception as e:
        logger.error(f"Error merging vectors: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function to process AOI masking and vector merging."""
    
    try:
        logger.info("=== Starting AOI Vector Masking and Merging Process (MULTI-POLYGON VERSION) ===")
        
        # Get the script directory and go up two levels to reach the base directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(os.path.dirname(script_dir))  # Go up from src/morocco_ftw to base
        
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Base directory: {base_path}")
        
        # Input files (all in the base directory)
        aoi_shapefile = os.path.join(base_path, "SECTEURS.shp")
        # input_vectors = [
        #     os.path.join(base_path, "morocco_gdal_boundaries.gpkg"),
        #     os.path.join(base_path, "morocco_mid_mosaic_boundaries.gpkg"),
        #     os.path.join(base_path, "morocco_tr_aoi_boundaries.gpkg")
        # ]
        input_vectors = [
            os.path.join(base_path, "morocco_mid_mosaic_morocco_trained.gpkg"),
            os.path.join(base_path, "morocco_mosaic_morocco_trained.gpkg"),
            os.path.join(base_path, "morocco_tr_aoi_morocco_trained.gpkg")
        ]
        # Output files (also in the base directory)
        # temp_dir = os.path.join(base_path, "temp_clipped_multipolygon")
        # final_output = os.path.join(base_path, "morocco_merged_boundaries_ALL_AOI.gpkg")
        temp_dir = os.path.join(base_path, "temp_clipped_morocco_trained")
        final_output = os.path.join(base_path, "morocco_merged_morocco_trained_ALL_AOI.gpkg")

        # Log all file paths
        logger.info(f"AOI shapefile: {aoi_shapefile}")
        logger.info(f"Input vectors: {input_vectors}")
        logger.info(f"Temp directory: {temp_dir}")
        logger.info(f"Final output: {final_output}")
        
        # Create temporary directory
        logger.info("Creating temporary directory...")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Check if files exist and log AOI polygon count
        logger.info("Checking AOI shapefile...")
        if not os.path.exists(aoi_shapefile):
            logger.error(f"AOI shapefile not found: {aoi_shapefile}")
            return False
        else:
            # Count AOI polygons
            aoi_ds = ogr.Open(aoi_shapefile, 0)
            if aoi_ds:
                aoi_layer = aoi_ds.GetLayer()
                aoi_count = aoi_layer.GetFeatureCount()
                logger.info(f"✓ AOI shapefile found with {aoi_count} polygons: {aoi_shapefile}")
                del aoi_ds
            else:
                logger.error(f"Could not open AOI shapefile: {aoi_shapefile}")
                return False
        
        for vector in input_vectors:
            if os.path.exists(vector):
                logger.info(f"✓ Input vector found: {vector}")
            else:
                logger.warning(f"✗ Input vector not found: {vector}")
        
        # Step 1: Clip each vector with ALL AOI polygons
        logger.info("=== Starting AOI clipping process (ALL POLYGONS) ===")
        clipped_vectors = []
        
        for i, input_vector in enumerate(input_vectors):
            if not os.path.exists(input_vector):
                logger.warning(f"Input vector not found: {input_vector}")
                continue
                
            # Generate output filename
            base_name = Path(input_vector).stem
            clipped_output = os.path.join(temp_dir, f"{base_name}_clipped_ALL_AOI.gpkg")
            
            logger.info(f"Clipping {input_vector} with ALL AOI polygons...")
            if clip_vector_with_aoi(input_vector, aoi_shapefile, clipped_output):
                clipped_vectors.append(clipped_output)
                logger.info(f"✓ Successfully clipped: {clipped_output}")
            else:
                logger.error(f"✗ Failed to clip {input_vector}")
        
        if not clipped_vectors:
            logger.error("No vectors were successfully clipped")
            return False
        
        logger.info(f"Successfully clipped {len(clipped_vectors)} vectors")
        
        # Step 2: Merge clipped vectors
        logger.info("=== Starting vector merging process ===")
        if merge_vectors(clipped_vectors, final_output):
            logger.info(f"✓ Successfully created merged output: {final_output}")
            
            # Verify output
            final_ds = ogr.Open(final_output, 0)
            if final_ds:
                final_layer = final_ds.GetLayer()
                final_count = final_layer.GetFeatureCount()
                logger.info(f"Final output contains {final_count} features")
                del final_ds
            
            return True
        else:
            logger.error("✗ Failed to merge vectors")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in main(): {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    try:
        print("Starting AOI Vector Masking and Merging Script (ALL AOI POLYGONS VERSION)...")
        success = main()
        if success:
            print("=== Process completed successfully! ===")
            print("Check the file: morocco_merged_boundaries_ALL_AOI.gpkg")
            logger.info("Process completed successfully!")
            sys.exit(0)
        else:
            print("=== Process failed! Check the log file for details ===")
            logger.error("Process failed!")
            sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        logger.error(f"CRITICAL ERROR: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)