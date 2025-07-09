#!/usr/bin/env python3
"""
GDAL polygonization script for raster data.
"""

import os
import sys
import time

def check_gdal_installation():
    """Check if GDAL is properly installed."""
    try:
        from osgeo import gdal, ogr, osr
        print(f"GDAL version: {gdal.VersionInfo()}")
        return True
    except ImportError as e:
        print(f"GDAL not found: {e}")
        print("Install GDAL with: conda install -c conda-forge gdal")
        return False

def polygonize_raster(input_raster, output_vector, filter_value=1):
    """Polygonize raster using GDAL Python bindings."""
    
    from osgeo import gdal, ogr, osr
    
    print(f"Input: {input_raster}")
    print(f"Output: {output_vector}")
    
    gdal.UseExceptions()
    
    try:
        # Open source raster
        src_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if src_ds is None:
            raise Exception(f"Could not open {input_raster}")
        
        print(f"Raster size: {src_ds.RasterXSize} x {src_ds.RasterYSize}")
        
        srcband = src_ds.GetRasterBand(1)
        
        # Create memory layer for polygonization
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('temp')
        
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())
        
        mem_layer = mem_ds.CreateLayer('polygons', srs=srs)
        
        field_defn = ogr.FieldDefn('DN', ogr.OFTInteger)
        mem_layer.CreateField(field_defn)
        
        print("Running polygonization...")
        start_time = time.time()
        
        gdal.Polygonize(srcband, None, mem_layer, 0, [], callback=gdal.TermProgress_nocb)
        
        print(f"Completed in {time.time() - start_time:.1f} seconds")
        
        # Create output file
        out_driver = ogr.GetDriverByName('GPKG')
        if os.path.exists(output_vector):
            out_driver.DeleteDataSource(output_vector)
        
        out_ds = out_driver.CreateDataSource(output_vector)
        out_layer = out_ds.CreateLayer('field_boundaries', srs=srs)
        
        # Add output fields
        id_field = ogr.FieldDefn('id', ogr.OFTString)
        out_layer.CreateField(id_field)
        
        area_field = ogr.FieldDefn('area_ha', ogr.OFTReal)
        out_layer.CreateField(area_field)
        
        perimeter_field = ogr.FieldDefn('perimeter_m', ogr.OFTReal)
        out_layer.CreateField(perimeter_field)
        
        # Filter and copy features
        print(f"Filtering polygons (DN = {filter_value})...")
        
        feature_count = 0
        filtered_count = 0
        
        mem_layer.ResetReading()
        for feature in mem_layer:
            feature_count += 1
            
            dn_value = feature.GetField('DN')
            if dn_value != filter_value:
                continue
                
            geom = feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                continue
            
            # Calculate area and perimeter
            area_sq_m = geom.GetArea()
            area_ha = area_sq_m * 0.0001
            perimeter_m = geom.Boundary().Length()
            
            # Skip very small polygons
            if area_ha < 0.01:
                continue
            
            # Create output feature
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetGeometry(geom)
            out_feature.SetField('id', str(filtered_count + 1))
            out_feature.SetField('area_ha', round(area_ha, 4))
            out_feature.SetField('perimeter_m', round(perimeter_m, 2))
            
            out_layer.CreateFeature(out_feature)
            filtered_count += 1
            
            if filtered_count % 1000 == 0:
                print(f"Processed {filtered_count} features...")
        
        print(f"Total polygons: {feature_count}")
        print(f"Filtered polygons: {filtered_count}")
        
        # Cleanup
        src_ds = None
        mem_ds = None
        out_ds = None
        
        print(f"Output saved to: {output_vector}")
        return True
        
    except Exception as e:
        print(f"Error during polygonization: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python gdal_polygonize.py <input_raster> [output_vector]")
        return 1
    
    input_raster = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_vector = sys.argv[2]
    else:
        base_name = os.path.splitext(input_raster)[0]
        output_vector = f"{base_name}_boundaries.gpkg"
    
    if not os.path.exists(input_raster):
        print(f"Error: Input file {input_raster} does not exist")
        return 1
    
    if not check_gdal_installation():
        return 1
    
    success = polygonize_raster(input_raster, output_vector, filter_value=1)
    
    if success:
        print("Polygonization completed successfully")
        return 0
    else:
        print("Polygonization failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())