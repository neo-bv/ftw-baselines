# Morocco Field Boundary Segmentation with Fields of The World (FTW)
## This project includes NDVI-based filtering to identify active agricultural fields and fine-tuning of pre-trained FTW models on Morocco-specific data.

The Morocco FTW project addresses the challenge of agricultural field boundary detection in Morocco by:

1. NDVI Filtering: Using temporal NDVI analysis to identify active agricultural fields
2. Data Preprocessing: Converting filtered field polygons to training-ready format
3. Model Training: Fine-tuning pre-trained FTW models on filtered Morocco data
4. Inference: Running predictions on Sentinel-2 imagery
5. Post-processing: Converting predictions to vector format and merging results

The following scripts usage are below:
src/morocco_ftw/
**mosaic_sentinel.py**               # Create Sentinel-2 mosaics

**ndvi_check.py**                    # NDVI-based field filtering

**3class_raster_prepare.py**         # Convert filtered polygons to 3-class rasters

**preprocessing_for_training.py**    # Create training patches from rasters

**organize_morocco_data.py**         # Structure data for FTW training format

**create_chips.py**                  # Generate metadata for training chips

**calculate_class_weight_morocco.py** # Calculate class weights used for training

**train_morocco.py**                 # Fine-tune FTW model on Morocco data

**weight_extraction_inference.py**   # Run inference on Sentinel-2 images

**gdal_polygonize_windows.py**       # Convert predictions to polygons

**aoi_vector_merge_multipolygon.py** # Merge and clip results with AOI

**geopandas_parquet_merge.py**       # Alternative merging approach

To set up the environment, the following command should conduct:
```bash
conda create -n ftw python=3.9 -y

conda activate ftw

conda install -c conda-forge gdal -y

where gdal

conda list gdal

conda info --envs

dir C:\Users\qin.xu\AppData\Local\anaconda3\envs\ftw\Library\bin\gdal*.dll

conda install -c conda-forge rasterio pyproj -y

pip list

python

python -m pip uninstall pip setuptools

python -m pip install --upgrade pip

pip install -upgrade pip

pip install pip

pip install -U pip

pip install --upgrade setuptools

pip install ftw-tools

pip install ftw-tools stackstac rioxarray pyarrow
python -c "import rasterio; print(rasterio.__version__)"
```

# Usage
## 1. Create mosaics for Sentinel-2 imagery and NDVI-Based Field Filtering 
python src/morocco_ftw/mosaic_sentinel.py
python src/morocco_ftw/ndvi_check.py
## 2. Raster Ground Truth Generation
python src/morocco_ftw/3class_raster_prepare.py
## 3. Training data preparation and organization
python src/morocco_ftw/preprocessing_for_training.py
python src/morocco_ftw/organize_morocco_data.py
python src/morocco_ftw/create_chips.py
## 4. Model training
python src/morocco_ftw/train_morocco.py
Training configuration is in morocco_config.yaml
Before training, should set up "set KMP_DUPLICATE_LIB_OK=TRUE"
## 5. Inference
python src/morocco_ftw/weight_extraction_inference.py
## 6. Post-processing
python src/morocco_ftw/gdal_polygonize_windows.py input_prediction.tif output_polygons.gpkg
python src/morocco_ftw/aoi_vector_merge_multipolygon.py



