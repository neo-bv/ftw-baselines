#!/usr/bin/env python3
"""
Debug script to check what happened during the merging process
"""

import os
from osgeo import ogr, osr

def check_file_contents(filepath, file_description):
    """Check contents of a geospatial file"""
    print(f"\n=== Checking {file_description} ===")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File does not exist!")
        return False
    
    file_size = os.path.getsize(filepath)
    print(f"📁 Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    try:
        ds = ogr.Open(filepath, 0)
        if not ds:
            print("❌ Cannot open with GDAL/OGR")
            return False
        
        print(f"✅ Opened successfully")
        
        for i in range(ds.GetLayerCount()):
            layer = ds.GetLayerByIndex(i)
            layer_name = layer.GetName()
            feature_count = layer.GetFeatureCount()
            
            print(f"📊 Layer '{layer_name}': {feature_count} features")
            
            # Get geometry info
            geom_type = layer.GetGeomType()
            geom_name = ogr.GeometryTypeToName(geom_type)
            print(f"🗺️  Geometry: {geom_name}")
            
            # Get CRS
            srs = layer.GetSpatialRef()
            if srs:
                auth_name = srs.GetAuthorityName(None)
                auth_code = srs.GetAuthorityCode(None)
                if auth_name and auth_code:
                    print(f"🌍 CRS: {auth_name}:{auth_code}")
                else:
                    print(f"🌍 CRS: {srs.GetName()}")
            else:
                print("⚠️  No CRS defined")
            
            # Get extent
            try:
                extent = layer.GetExtent()
                print(f"📏 Extent: ({extent[0]:.2f}, {extent[2]:.2f}) to ({extent[1]:.2f}, {extent[3]:.2f})")
            except:
                print("⚠️  Cannot get extent")
            
            # Sample first feature if exists
            if feature_count > 0:
                layer.ResetReading()
                feature = layer.GetNextFeature()
                if feature:
                    geom = feature.GetGeometryRef()
                    if geom:
                        print(f"🎯 Sample geometry: {geom.GetGeometryName()}")
                        if hasattr(geom, 'GetArea'):
                            try:
                                area = geom.GetArea()
                                print(f"   Area: {area:.6f}")
                            except:
                                pass
        
        del ds
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_aoi_intersection():
    """Test if AOI intersects with input files"""
    print(f"\n{'='*60}")
    print("TESTING AOI INTERSECTION WITH INPUT FILES")
    print(f"{'='*60}")
    
    base_path = r"C:\Users\qin.xu\github\ftw-baselines"
    aoi_file = os.path.join(base_path, "SECTEURS.shp")
    
    input_files = [
        "morocco_gdal_boundaries.gpkg",
        "morocco_mid_mosaic_boundaries.gpkg", 
        "morocco_tr_aoi_boundaries.gpkg"
    ]
    
    # First check AOI
    if not check_file_contents(aoi_file, "AOI Shapefile"):
        return
    
    # Check each input file
    for input_file in input_files:
        filepath = os.path.join(base_path, input_file)
        check_file_contents(filepath, f"Input: {input_file}")
    
    # Now test intersection
    print(f"\n{'='*60}")
    print("TESTING SPATIAL INTERSECTION")
    print(f"{'='*60}")
    
    try:
        # Open AOI
        aoi_ds = ogr.Open(aoi_file, 0)
        if not aoi_ds:
            print("❌ Cannot open AOI file")
            return
        
        aoi_layer = aoi_ds.GetLayer()
        aoi_feature = aoi_layer.GetNextFeature()
        if not aoi_feature:
            print("❌ No features in AOI")
            return
        
        aoi_geom = aoi_feature.GetGeometryRef()
        if not aoi_geom:
            print("❌ No geometry in AOI")
            return
        
        print(f"✅ AOI geometry loaded: {aoi_geom.GetGeometryName()}")
        aoi_area = aoi_geom.GetArea()
        print(f"🏁 AOI area: {aoi_area:.6f}")
        
        # Test each input file against AOI
        for input_file in input_files:
            filepath = os.path.join(base_path, input_file)
            if not os.path.exists(filepath):
                continue
                
            print(f"\n--- Testing {input_file} ---")
            
            input_ds = ogr.Open(filepath, 0)
            if not input_ds:
                print(f"❌ Cannot open {input_file}")
                continue
            
            input_layer = input_ds.GetLayer()
            intersecting_features = 0
            total_features = input_layer.GetFeatureCount()
            
            print(f"🔍 Checking {total_features} features for intersection...")
            
            for feature in input_layer:
                geom = feature.GetGeometryRef()
                if geom and aoi_geom.Intersects(geom):
                    intersecting_features += 1
                    if intersecting_features == 1:  # Show details for first intersection
                        intersection = geom.Intersection(aoi_geom)
                        if intersection and not intersection.IsEmpty():
                            print(f"✅ First intersection found!")
                            print(f"   Intersection type: {intersection.GetGeometryName()}")
                            try:
                                int_area = intersection.GetArea()
                                print(f"   Intersection area: {int_area:.6f}")
                            except:
                                pass
            
            print(f"📊 Result: {intersecting_features}/{total_features} features intersect with AOI")
            
            if intersecting_features == 0:
                print("⚠️  NO INTERSECTION FOUND - This explains why no features were clipped!")
            
            del input_ds
        
        del aoi_ds
        
    except Exception as e:
        print(f"❌ Error during intersection test: {str(e)}")

def check_temp_files():
    """Check if temporary clipped files exist and have content"""
    print(f"\n{'='*60}")
    print("CHECKING TEMPORARY CLIPPED FILES")
    print(f"{'='*60}")
    
    base_path = r"C:\Users\qin.xu\github\ftw-baselines"
    temp_dir = os.path.join(base_path, "temp_clipped")
    
    if not os.path.exists(temp_dir):
        print(f"❌ Temp directory does not exist: {temp_dir}")
        return
    
    print(f"📁 Temp directory: {temp_dir}")
    
    # List all files in temp directory
    try:
        files = os.listdir(temp_dir)
        print(f"📄 Files in temp directory: {len(files)}")
        
        for file in files:
            if file.endswith('.gpkg'):
                filepath = os.path.join(temp_dir, file)
                check_file_contents(filepath, f"Temp file: {file}")
                
    except Exception as e:
        print(f"❌ Error checking temp files: {str(e)}")

if __name__ == "__main__":
    print("DEBUGGING MOROCCO VECTOR MERGE PROCESS")
    print("="*60)
    
    # Check the final output first
    final_output = r"C:\Users\qin.xu\github\ftw-baselines\morocco_merged_boundaries.gpkg"
    check_file_contents(final_output, "Final merged output")
    
    # Test AOI and input file intersections
    test_aoi_intersection()
    
    # Check temporary files
    check_temp_files()
    
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}")
    print("If no intersections were found:")
    print("1. Check coordinate systems - AOI and input files might be in different CRS")
    print("2. Check if AOI covers the right geographic area")
    print("3. Visualize AOI and input files separately in QGIS first")
    print("4. Consider transforming coordinate systems to match")
