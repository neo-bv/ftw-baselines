#!/usr/bin/env python3
"""
Test script to check geospatial package installation
"""

def test_imports():
    """Test importing all required packages"""
    
    packages = [
        ('geopandas', 'gpd'),
        ('rasterio', 'rasterio'),
        ('fiona', 'fiona'),
        ('shapely', 'shapely'),
        ('pyogrio', 'pyogrio')
    ]
    
    results = {}
    
    for package_name, import_name in packages:
        try:
            if import_name == 'gpd':
                import geopandas as gpd
                version = gpd.__version__
            elif import_name == 'rasterio':
                import rasterio
                version = rasterio.__version__
            elif import_name == 'fiona':
                import fiona
                version = fiona.__version__
            elif import_name == 'shapely':
                import shapely
                version = shapely.__version__
            elif import_name == 'pyogrio':
                import pyogrio
                version = pyogrio.__version__
            
            results[package_name] = f"✓ OK (v{version})"
            print(f"{package_name}: ✓ OK (v{version})")
            
        except Exception as e:
            results[package_name] = f"✗ FAILED: {str(e)}"
            print(f"{package_name}: ✗ FAILED: {str(e)}")
    
    return results

def test_shapefile_reading():
    """Test reading a shapefile"""
    import os
    
    shapefile_path = r"data\morocco\Parcel GT Stef New.shp"
    
    if not os.path.exists(shapefile_path):
        print(f"\nShapefile test: ✗ File not found: {shapefile_path}")
        return False
    
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shapefile_path)
        print(f"\nShapefile test: ✓ Successfully read {len(gdf)} features")
        print(f"CRS: {gdf.crs}")
        print(f"Bounds: {gdf.total_bounds}")
        return True
        
    except Exception as e:
        print(f"\nShapefile test: ✗ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing geospatial package installation...")
    print("=" * 50)
    
    # Test imports
    import_results = test_imports()
    
    # Test shapefile reading if imports work
    if all("✓ OK" in result for result in import_results.values()):
        test_shapefile_reading()
    else:
        print("\nSkipping shapefile test due to import failures")
    
    print("\n" + "=" * 50)
    print("Test complete!")