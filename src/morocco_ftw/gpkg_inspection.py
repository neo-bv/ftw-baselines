#!/usr/bin/env python3
"""
Quick script to inspect GPKG file contents and validity
"""

import os
from osgeo import ogr, osr

def inspect_gpkg(gpkg_path):
    """Inspect a GPKG file and report its contents"""
    
    print(f"Inspecting: {gpkg_path}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(gpkg_path):
        print(f"❌ File does not exist: {gpkg_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(gpkg_path)
    print(f"📁 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    try:
        # Open the file
        ds = ogr.Open(gpkg_path, 0)
        if not ds:
            print("❌ Could not open file with GDAL/OGR")
            return False
        
        print(f"✅ File opened successfully")
        
        # Get number of layers
        layer_count = ds.GetLayerCount()
        print(f"📊 Number of layers: {layer_count}")
        
        if layer_count == 0:
            print("⚠️  No layers found in file")
            return False
        
        # Inspect each layer
        for i in range(layer_count):
            layer = ds.GetLayerByIndex(i)
            layer_name = layer.GetName()
            feature_count = layer.GetFeatureCount()
            
            print(f"\n--- Layer {i+1}: {layer_name} ---")
            print(f"🔢 Feature count: {feature_count}")
            
            # Get geometry type
            geom_type = layer.GetGeomType()
            geom_name = ogr.GeometryTypeToName(geom_type)
            print(f"🗺️  Geometry type: {geom_name}")
            
            # Get spatial reference
            srs = layer.GetSpatialRef()
            if srs:
                print(f"🌍 Spatial Reference: {srs.GetAuthorityName(None)}:{srs.GetAuthorityCode(None)}")
            else:
                print("⚠️  No spatial reference found")
            
            # Get extent
            try:
                extent = layer.GetExtent()
                print(f"📏 Extent: {extent}")
                print(f"   Min X: {extent[0]:.6f}, Max X: {extent[1]:.6f}")
                print(f"   Min Y: {extent[2]:.6f}, Max Y: {extent[3]:.6f}")
            except:
                print("⚠️  Could not get extent")
            
            # Get field information
            layer_defn = layer.GetLayerDefn()
            field_count = layer_defn.GetFieldCount()
            print(f"📋 Number of fields: {field_count}")
            
            if field_count > 0:
                print("   Fields:")
                for j in range(field_count):
                    field_defn = layer_defn.GetFieldDefn(j)
                    field_name = field_defn.GetName()
                    field_type = field_defn.GetTypeName()
                    print(f"     - {field_name} ({field_type})")
            
            # Sample a few features
            if feature_count > 0:
                print("   Sample features:")
                layer.ResetReading()
                for k, feature in enumerate(layer):
                    if k >= 3:  # Only show first 3 features
                        break
                    
                    geom = feature.GetGeometryRef()
                    if geom:
                        print(f"     Feature {k+1}: {geom.GetGeometryName()}")
                        if hasattr(geom, 'GetPointCount'):
                            try:
                                point_count = geom.GetPointCount()
                                print(f"       Points: {point_count}")
                            except:
                                pass
                    else:
                        print(f"     Feature {k+1}: No geometry")
        
        del ds
        print(f"\n✅ File appears to be valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting file: {str(e)}")
        return False

if __name__ == "__main__":
    # Check the merged output file
    gpkg_file = r"C:\Users\qin.xu\github\ftw-baselines\morocco_merged_boundaries.gpkg"
    
    inspect_gpkg(gpkg_file)
    
    print("\n" + "="*50)
    print("QGIS Loading Tips:")
    print("1. Try 'Layer' → 'Add Layer' → 'Add Vector Layer'")
    print("2. Browse to the GPKG file")
    print("3. If multiple layers, select which one(s) to load")
    print("4. Check the CRS/projection matches your project")
    print("5. Right-click layer → 'Zoom to Layer' to see data")
    print("6. Check symbology if features appear but aren't visible")