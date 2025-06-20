#!/usr/bin/env python3
"""
AOI Vector Masking and Merging Script using GeoPandas for Parquet files
"""

import os
import sys
import logging

# Set up logging without Unicode characters
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('aoi_parquet_merge_geopandas.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        import geopandas as gpd
        import pandas as pd
        logger.info("GeoPandas imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import geopandas: {e}")
        print("Please install geopandas: conda install geopandas")
        return False
    
    base_path = r"C:\Users\qin.xu\github\ftw-baselines"
    
    # Input files
    aoi_shapefile = os.path.join(base_path, "SECTEURS.shp")
    input_parquets = [
        os.path.join(base_path, "morocco_mid_mosaic3class-inf.parquet"),
        os.path.join(base_path, "morocco_tr_aoi3class-inf.parquet"),
        os.path.join(base_path, "morocco_mosaic3class-inf.parquet")
    ]
    
    final_output = os.path.join(base_path, "morocco_merged_parquet_geopandas.gpkg")
    
    try:
        # Load AOI
        logger.info("Loading AOI shapefile...")
        aoi_gdf = gpd.read_file(aoi_shapefile)
        logger.info(f"AOI loaded: {len(aoi_gdf)} polygons, CRS: {aoi_gdf.crs}")
        
        # Combine all AOI polygons into one
        aoi_combined = aoi_gdf.unary_union
        aoi_combined_gdf = gpd.GeoDataFrame([1], geometry=[aoi_combined], crs=aoi_gdf.crs)
        
        merged_gdfs = []
        
        # Process each Parquet file
        for parquet_file in input_parquets:
            if not os.path.exists(parquet_file):
                logger.warning(f"File not found: {parquet_file}")
                continue
            
            logger.info(f"Processing: {parquet_file}")
            
            try:
                # Try to read as GeoParquet
                gdf = gpd.read_parquet(parquet_file)
                logger.info(f"Loaded {len(gdf)} features, CRS: {gdf.crs}")
                
                # Reproject AOI to match data CRS if needed
                if gdf.crs != aoi_gdf.crs:
                    aoi_reprojected = aoi_combined_gdf.to_crs(gdf.crs)
                    logger.info(f"Reprojected AOI from {aoi_gdf.crs} to {gdf.crs}")
                else:
                    aoi_reprojected = aoi_combined_gdf
                
                # Clip data with AOI
                clipped = gpd.clip(gdf, aoi_reprojected)
                logger.info(f"Clipped result: {len(clipped)} features")
                
                if len(clipped) > 0:
                    merged_gdfs.append(clipped)
                else:
                    logger.warning(f"No features after clipping: {parquet_file}")
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                
                # Try reading as regular parquet and look for geometry
                try:
                    df = pd.read_parquet(parquet_file)
                    logger.info(f"Read as pandas DataFrame: {df.shape}")
                    logger.info(f"Columns: {list(df.columns)}")
                    
                    # Look for geometry columns
                    geom_cols = [col for col in df.columns if 'geom' in col.lower()]
                    if geom_cols:
                        logger.info(f"Found geometry columns: {geom_cols}")
                    
                except Exception as e2:
                    logger.error(f"Could not read as pandas either: {e2}")
        
        # Merge all clipped data
        if merged_gdfs:
            logger.info(f"Merging {len(merged_gdfs)} datasets...")
            final_gdf = gpd.pd.concat(merged_gdfs, ignore_index=True)
            logger.info(f"Final merged dataset: {len(final_gdf)} features")
            
            # Save result
            final_gdf.to_file(final_output, driver="GPKG")
            logger.info(f"Saved to: {final_output}")
            
            return True
        else:
            logger.error("No data to merge")
            return False
    
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("SUCCESS: Check morocco_merged_parquet_geopandas.gpkg")
    else:
        print("FAILED: Check the log file")