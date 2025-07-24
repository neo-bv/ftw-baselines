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
THRESHOLDS_TO_TEST = [0.01, 0.02, 0.05, 0.1]  # Different NDVI difference thresholds to test

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

def process_aoi(vector_path, sentinel_path, aoi_name, save_images=True):
    """Process one AOI pair and return raw polygon data"""
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
        
        # Save NDVI images for this AOI (only once, not for each threshold)
        if save_images:
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
                
                # Calculate NDVI difference (will be used for threshold testing)
                ndvi_diff = abs(ndvi_a_mean - ndvi_b_mean) if (not np.isnan(ndvi_a_mean) and not np.isnan(ndvi_b_mean)) else 0
                
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
                    'geometry': original_geom
                })
        
        # Clean up temp files
        for temp_file in [temp_a, temp_b]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return pd.DataFrame(results), original_crs

def apply_threshold_filter(combined_df, threshold):
    """Apply threshold filter to the combined dataframe"""
    # Create a copy to avoid modifying original
    df_filtered = combined_df.copy()
    
    # Apply filter condition
    passed_filter = (df_filtered['ndvi_diff'] > threshold) & \
                   (~df_filtered['ndvi_a_mean'].isna()) & \
                   (~df_filtered['ndvi_b_mean'].isna())
    
    df_filtered['passed_filter'] = passed_filter
    df_filtered['recommendation'] = df_filtered['passed_filter'].map({True: 'KEEP', False: 'EXCLUDE'})
    df_filtered['threshold_used'] = threshold
    
    return df_filtered

def save_threshold_results(df_filtered, threshold, output_crs):
    """Save results for a specific threshold"""
    threshold_str = f"{threshold:.2f}".replace('.', 'p')
    threshold_dir = os.path.join(output_dir, f"Threshold_{threshold_str}")
    os.makedirs(threshold_dir, exist_ok=True)
    
    # All results for this threshold
    gdf_all = gpd.GeoDataFrame(df_filtered, geometry='geometry', crs=output_crs)
    gdf_all.to_file(os.path.join(threshold_dir, f'morocco_ndvi_filtered_th{threshold_str}.shp'))
    gdf_all.drop('geometry', axis=1).to_csv(os.path.join(threshold_dir, f'morocco_ndvi_results_th{threshold_str}.csv'), index=False)
    
    # Keep only active fields for this threshold
    gdf_keep = gdf_all[gdf_all['recommendation'] == 'KEEP'].copy()
    if len(gdf_keep) > 0:
        gdf_keep.to_file(os.path.join(threshold_dir, f'morocco_active_fields_th{threshold_str}.shp'))
        gdf_keep.drop('geometry', axis=1).to_csv(os.path.join(threshold_dir, f'morocco_active_fields_th{threshold_str}.csv'), index=False)
    
    return threshold_dir

def print_threshold_summary(df_filtered, threshold):
    """Print summary for a specific threshold"""
    total = len(df_filtered)
    kept = df_filtered['passed_filter'].sum()
    excluded = total - kept
    
    print(f"\n--- Threshold {threshold:.2f} Results ---")
    print(f"  Total polygons: {total}")
    print(f"  KEEP (|NDVI_A - NDVI_B| > {threshold:.2f}): {kept}")
    print(f"  EXCLUDE: {excluded}")
    print(f"  Success rate: {kept/total*100:.1f}%")
    
    # Show average NDVI difference for kept vs excluded
    kept_data = df_filtered[df_filtered['passed_filter']]
    excluded_data = df_filtered[~df_filtered['passed_filter']]
    
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
    
    # Results by AOI for this threshold
    print(f"  By AOI:")
    for aoi in df_filtered['aoi_name'].unique():
        aoi_data = df_filtered[df_filtered['aoi_name'] == aoi]
        aoi_kept = aoi_data['passed_filter'].sum()
        print(f"    {aoi}: {aoi_kept}/{len(aoi_data)} kept ({aoi_kept/len(aoi_data)*100:.1f}%)")

def main():
    print("Morocco NDVI Field Filter - Multiple Threshold Testing")
    print(f"Testing thresholds: {THRESHOLDS_TO_TEST}")
    print("=" * 60)
    
    all_results = []
    output_crs = None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each AOI (only once - extract raw NDVI data)
    for pair in vector_tile_pairs:
        result_df, crs = process_aoi(pair['vector'], pair['sentinel'], pair['name'], save_images=True)
        if result_df is not None:
            all_results.append(result_df)
            if output_crs is None:
                output_crs = crs  # Use CRS from first valid AOI
    
    if not all_results:
        print("No data processed successfully.")
        return
    
    # Combine raw results (without any threshold filtering yet)
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create summary comparison table
    summary_data = []
    
    # Test each threshold
    for threshold in THRESHOLDS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"TESTING THRESHOLD: {threshold:.2f}")
        print(f"{'='*60}")
        
        # Apply threshold filter
        df_filtered = apply_threshold_filter(combined_df, threshold)
        
        # Print summary for this threshold
        print_threshold_summary(df_filtered, threshold)
        
        # Save results for this threshold
        threshold_dir = save_threshold_results(df_filtered, threshold, output_crs)
        
        # Add to summary comparison
        total = len(df_filtered)
        kept = df_filtered['passed_filter'].sum()
        summary_data.append({
            'threshold': threshold,
            'total_polygons': total,
            'kept': kept,
            'excluded': total - kept,
            'success_rate_percent': kept/total*100 if total > 0 else 0,
            'avg_ndvi_diff_kept': df_filtered[df_filtered['passed_filter']]['ndvi_diff'].mean() if kept > 0 else np.nan,
            'avg_ndvi_diff_excluded': df_filtered[~df_filtered['passed_filter']]['ndvi_diff'].mean() if (total - kept) > 0 else np.nan
        })
        
        print(f"  Results saved to: {threshold_dir}")
    
    # Create and save summary comparison table
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'threshold_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print final comparison table
    print(f"\n{'='*80}")
    print("THRESHOLD COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Threshold':<12} {'Total':<8} {'Kept':<8} {'Excluded':<10} {'Success %':<12} {'Avg Diff (Keep)':<16}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['threshold']:<12.2f} {row['total_polygons']:<8} {row['kept']:<8} {row['excluded']:<10} "
              f"{row['success_rate_percent']:<12.1f} {row['avg_ndvi_diff_kept']:<16.3f}")
    
    print(f"\nFiles saved to {output_dir}:")
    print("  NDVI Images:")
    print("    - NDVI_Images/{AOI_name}_NDVI_WindowA.tif")
    print("    - NDVI_Images/{AOI_name}_NDVI_WindowB.tif") 
    print("    - NDVI_Images/{AOI_name}_NDVI_Difference.tif")
    print("  Threshold Results:")
    for threshold in THRESHOLDS_TO_TEST:
        threshold_str = f"{threshold:.2f}".replace('.', 'p')
        print(f"    - Threshold_{threshold_str}/morocco_*_th{threshold_str}.*")
    print("  Summary:")
    print(f"    - threshold_comparison_summary.csv")
    
    print(f"\nDone! Compare the results from different thresholds to choose the best one for FTW training.")

if __name__ == "__main__":
    main()