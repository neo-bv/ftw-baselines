import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
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

output_dir = "Output/"
NDVI_THRESHOLD = 0.15  # Avoid bare soil
NDVI_DIFFERENCE_THRESHOLD = 0.1  # Minimum difference between windows

def calculate_ndvi(red, nir):
    """Calculate NDVI from red and NIR bands"""
    red = red.astype('float32')
    nir = nir.astype('float32')
    
    denominator = nir + red
    ndvi = np.divide(
        (nir - red),
        denominator,
        out=np.full_like(red, np.nan, dtype='float32'),
        where=(denominator != 0)
    )
    return np.clip(ndvi, -1, 1)

def save_ndvi_image(ndvi_array, profile, output_path):
    """Save NDVI array as GeoTIFF"""
    profile_ndvi = profile.copy()
    profile_ndvi.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw'  # Add compression to reduce file size
    })
    
    with rasterio.open(output_path, 'w', **profile_ndvi) as dst:
        dst.write(ndvi_array, 1)
    print(f"  Saved NDVI image: {output_path}")

def process_aoi(vector_path, sentinel_path, aoi_name):
    """Process one AOI pair"""
    print(f"Processing {aoi_name}...")
    
    # Check files exist
    if not os.path.exists(vector_path) or not os.path.exists(sentinel_path):
        print(f"  Skipping {aoi_name} - missing files")
        return None, None
    
    # Load polygons
    gdf = gpd.read_file(vector_path)
    original_crs = gdf.crs  # Store original CRS
    print(f"  Loaded {len(gdf)} polygons, CRS: {original_crs}")
    
    # Load Sentinel-2 and calculate NDVI
    with rasterio.open(sentinel_path) as src:
        bands = src.read()
        
        # Assuming band order: R_A, G_A, B_A, NIR_A, R_B, G_B, B_B, NIR_B
        red_a = bands[0]    # Red Window A
        nir_a = bands[3]    # NIR Window A
        red_b = bands[4]    # Red Window B
        nir_b = bands[7]    # NIR Window B
        
        ndvi_a = calculate_ndvi(red_a, nir_a)
        ndvi_b = calculate_ndvi(red_b, nir_b)
        
        # Save NDVI images for this AOI
        ndvi_dir = os.path.join(output_dir, "NDVI_Images")
        os.makedirs(ndvi_dir, exist_ok=True)
        
        ndvi_a_path = os.path.join(ndvi_dir, f"{aoi_name}_NDVI_WindowA.tif")
        ndvi_b_path = os.path.join(ndvi_dir, f"{aoi_name}_NDVI_WindowB.tif")
        ndvi_diff_path = os.path.join(ndvi_dir, f"{aoi_name}_NDVI_Difference.tif")
        
        # Calculate NDVI difference
        ndvi_difference = np.abs(ndvi_a - ndvi_b)
        
        # Save NDVI images
        save_ndvi_image(ndvi_a, src.profile, ndvi_a_path)
        save_ndvi_image(ndvi_b, src.profile, ndvi_b_path)
        save_ndvi_image(ndvi_difference, src.profile, ndvi_diff_path)
        
        # Reproject polygons if needed for processing
        gdf_for_processing = gdf.copy()
        if gdf.crs != src.crs:
            gdf_for_processing = gdf.to_crs(src.crs)
        
        # Save NDVI to temp files for masking
        profile = src.profile.copy()
        profile.update({'count': 1, 'dtype': 'float32', 'nodata': np.nan})
        
        temp_a = f"temp_ndvi_a_{aoi_name}.tif"
        temp_b = f"temp_ndvi_b_{aoi_name}.tif"
        
        with rasterio.open(temp_a, 'w', **profile) as dst:
            dst.write(ndvi_a, 1)
        with rasterio.open(temp_b, 'w', **profile) as dst:
            dst.write(ndvi_b, 1)
        
        # Extract NDVI for each polygon
        results = []
        
        for idx, row in tqdm(gdf_for_processing.iterrows(), total=len(gdf_for_processing), desc=f"{aoi_name} polygons"):
            geom = [row.geometry]
            
            try:
                # Window A NDVI
                with rasterio.open(temp_a) as src_a:
                    masked_a, _ = mask(src_a, geom, crop=True, nodata=np.nan)
                    valid_a = masked_a[0][~np.isnan(masked_a[0])]
                    ndvi_a_mean = np.mean(valid_a) if len(valid_a) > 0 else np.nan
                    ndvi_a_std = np.std(valid_a) if len(valid_a) > 0 else np.nan
                    ndvi_a_min = np.min(valid_a) if len(valid_a) > 0 else np.nan
                    ndvi_a_max = np.max(valid_a) if len(valid_a) > 0 else np.nan
                
                # Window B NDVI
                with rasterio.open(temp_b) as src_b:
                    masked_b, _ = mask(src_b, geom, crop=True, nodata=np.nan)
                    valid_b = masked_b[0][~np.isnan(masked_b[0])]
                    ndvi_b_mean = np.mean(valid_b) if len(valid_b) > 0 else np.nan
                    ndvi_b_std = np.std(valid_b) if len(valid_b) > 0 else np.nan
                    ndvi_b_min = np.min(valid_b) if len(valid_b) > 0 else np.nan
                    ndvi_b_max = np.max(valid_b) if len(valid_b) > 0 else np.nan
                
                # Apply new filter conditions:
                # 1. NDVI > 0.15 in both windows (avoid bare soil)
                # 2. Difference in NDVI > 0.1 (temporal change)
                ndvi_diff = abs(ndvi_a_mean - ndvi_b_mean) if (not np.isnan(ndvi_a_mean) and not np.isnan(ndvi_b_mean)) else 0
                
                passed_filter = (ndvi_a_mean > NDVI_THRESHOLD and 
                               ndvi_b_mean > NDVI_THRESHOLD and
                               ndvi_diff > NDVI_DIFFERENCE_THRESHOLD and
                               not np.isnan(ndvi_a_mean) and 
                               not np.isnan(ndvi_b_mean))
                
                # Use ORIGINAL geometry (not reprojected)
                original_geom = gdf.iloc[idx].geometry
                
                results.append({
                    'aoi_name': aoi_name,
                    'polygon_id': f"{aoi_name}_{idx}",
                    'ndvi_a_mean': ndvi_a_mean,
                    'ndvi_a_std': ndvi_a_std,
                    'ndvi_a_min': ndvi_a_min,
                    'ndvi_a_max': ndvi_a_max,
                    'ndvi_b_mean': ndvi_b_mean,
                    'ndvi_b_std': ndvi_b_std,
                    'ndvi_b_min': ndvi_b_min,
                    'ndvi_b_max': ndvi_b_max,
                    'ndvi_diff': ndvi_diff,
                    'passed_filter': passed_filter,
                    'recommendation': 'KEEP' if passed_filter else 'EXCLUDE',
                    'geometry': original_geom
                })
                
            except Exception as e:
                # Use ORIGINAL geometry (not reprojected)
                original_geom = gdf.iloc[idx].geometry
                
                results.append({
                    'aoi_name': aoi_name,
                    'polygon_id': f"{aoi_name}_{idx}",
                    'ndvi_a_mean': np.nan,
                    'ndvi_a_std': np.nan,
                    'ndvi_a_min': np.nan,
                    'ndvi_a_max': np.nan,
                    'ndvi_b_mean': np.nan,
                    'ndvi_b_std': np.nan,
                    'ndvi_b_min': np.nan,
                    'ndvi_b_max': np.nan,
                    'ndvi_diff': 0,
                    'passed_filter': False,
                    'recommendation': 'EXCLUDE',
                    'geometry': original_geom
                })
        
        # Clean up temp files
        for temp_file in [temp_a, temp_b]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return pd.DataFrame(results), original_crs

def main():
    print("Morocco NDVI Field Filter with Image Output")
    print("Conditions: NDVI > 0.15 in both windows AND difference > 0.1")
    print("=" * 60)
    
    all_results = []
    output_crs = None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each AOI
    for pair in vector_tile_pairs:
        result_df, crs = process_aoi(pair['vector'], pair['sentinel'], pair['name'])
        if result_df is not None:
            all_results.append(result_df)
            if output_crs is None:
                output_crs = crs  # Use CRS from first valid AOI
    
    if not all_results:
        print("No data processed successfully.")
        return
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Summary
    total = len(combined_df)
    kept = combined_df['passed_filter'].sum()
    excluded = total - kept
    
    print(f"\nResults Summary:")
    print(f"  Total polygons: {total}")
    print(f"  KEEP (NDVI_A > 0.15 AND NDVI_B > 0.15 AND |NDVI_A - NDVI_B| > 0.1): {kept}")
    print(f"  EXCLUDE: {excluded}")
    print(f"  Success rate: {kept/total*100:.1f}%")
    
    # Show average NDVI difference for kept vs excluded
    kept_data = combined_df[combined_df['passed_filter']]
    excluded_data = combined_df[~combined_df['passed_filter']]
    
    if len(kept_data) > 0:
        avg_diff_kept = kept_data['ndvi_diff'].mean()
        avg_ndvi_a_kept = kept_data['ndvi_a_mean'].mean()
        avg_ndvi_b_kept = kept_data['ndvi_b_mean'].mean()
        print(f"  Average NDVI difference (KEEP): {avg_diff_kept:.3f}")
        print(f"  Average NDVI Window A (KEEP): {avg_ndvi_a_kept:.3f}")
        print(f"  Average NDVI Window B (KEEP): {avg_ndvi_b_kept:.3f}")
    
    if len(excluded_data) > 0:
        avg_diff_excluded = excluded_data['ndvi_diff'].mean()
        print(f"  Average NDVI difference (EXCLUDE): {avg_diff_excluded:.3f}")
    
    # Results by AOI
    print(f"\nBy AOI:")
    for aoi in combined_df['aoi_name'].unique():
        aoi_data = combined_df[combined_df['aoi_name'] == aoi]
        aoi_kept = aoi_data['passed_filter'].sum()
        print(f"  {aoi}: {aoi_kept}/{len(aoi_data)} kept ({aoi_kept/len(aoi_data)*100:.1f}%)")
    
    # Save results
    # All results
    gdf_all = gpd.GeoDataFrame(combined_df, geometry='geometry', crs=output_crs)
    gdf_all.to_file(os.path.join(output_dir, 'morocco_ndvi_filtered.shp'))
    gdf_all.drop('geometry', axis=1).to_csv(os.path.join(output_dir, 'morocco_ndvi_results.csv'), index=False)
    
    # Keep only active fields
    gdf_keep = gdf_all[gdf_all['recommendation'] == 'KEEP'].copy()
    gdf_keep.to_file(os.path.join(output_dir, 'morocco_active_fields.shp'))
    gdf_keep.drop('geometry', axis=1).to_csv(os.path.join(output_dir, 'morocco_active_fields.csv'), index=False)
    
    print(f"\nFiles saved to {output_dir}:")
    print("  Shapefiles and CSVs:")
    print("    - morocco_ndvi_filtered.shp (all polygons with filter results)")
    print("    - morocco_active_fields.shp (only KEEP polygons)")
    print("    - morocco_ndvi_results.csv (all results)")
    print("    - morocco_active_fields.csv (active fields only)")
    print("  NDVI Images:")
    print("    - NDVI_Images/{AOI_name}_NDVI_WindowA.tif")
    print("    - NDVI_Images/{AOI_name}_NDVI_WindowB.tif")
    print("    - NDVI_Images/{AOI_name}_NDVI_Difference.tif")
    
    print(f"\nDone! Use the KEEP polygons for FTW training.")
    print("Check the NDVI_Images folder for visual analysis of NDVI values.")

if __name__ == "__main__":
    main()