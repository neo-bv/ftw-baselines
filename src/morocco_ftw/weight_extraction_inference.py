#This script is used for using trained model to make predictions on three sentinel2 images
#!/usr/bin/env python3
"""
Extract weights from checkpoint and run inference on Sentinel-2 images
Updated to use the new filtered Morocco training model
"""

import os
import time
import numpy as np
import torch
import rasterio
from rasterio.enums import ColorInterp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set backend to avoid GUI issues
os.environ['MPLBACKEND'] = 'Agg'

class SimpleRasterDataset(Dataset):
    """Simple dataset for loading raster patches"""
    
    def __init__(self, raster_path, patch_size=512, stride=None, padding=64):
        self.raster_path = raster_path
        self.patch_size = patch_size
        self.padding = padding
        
        if stride is None:
            self.stride = patch_size - padding * 2
        else:
            self.stride = stride
            
        # Open raster and get info
        with rasterio.open(raster_path) as src:
            self.height, self.width = src.shape
            self.profile = src.profile
            self.transform = src.transform
            self.tags = src.tags()
            
        # Calculate patch grid
        self.patches = []
        for y in range(0, self.height, self.stride):
            for x in range(0, self.width, self.stride):
                # Ensure patch doesn't go beyond image bounds
                patch_height = min(self.patch_size, self.height - y)
                patch_width = min(self.patch_size, self.width - x)
                
                if patch_height >= 64 and patch_width >= 64:  # Minimum patch size
                    self.patches.append((x, y, patch_width, patch_height))
                    
        print(f"Created {len(self.patches)} patches for {self.width}x{self.height} image")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        x, y, patch_width, patch_height = self.patches[idx]
        
        # Read patch from raster
        with rasterio.open(self.raster_path) as src:
            # Create window
            window = rasterio.windows.Window(x, y, patch_width, patch_height)
            
            # Read data
            data = src.read(window=window)  # Shape: (bands, height, width)
            
            # Pad to patch_size if needed
            if patch_height < self.patch_size or patch_width < self.patch_size:
                padded_data = np.zeros((data.shape[0], self.patch_size, self.patch_size), dtype=data.dtype)
                padded_data[:, :patch_height, :patch_width] = data
                data = padded_data
        
        # Convert to tensor and normalize
        data = torch.from_numpy(data).float() / 3000.0
        
        return {
            'image': data,
            'window_info': (x, y, patch_width, patch_height),
            'patch_idx': idx
        }

def extract_model_weights(checkpoint_path):
    """Extract and clean model weights from Lightning checkpoint"""
    print(f"Extracting weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    hparams = checkpoint['hyper_parameters']
    print(f"Model config: {hparams['model']}, backbone: {hparams['backbone']}")
    print(f"Classes: {hparams['num_classes']}, Channels: {hparams['in_channels']}")
    
    # Extract state dict and remove 'model.' prefix from keys
    state_dict = checkpoint['state_dict']
    cleaned_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('model.'):
            clean_key = key[6:]  # Remove 'model.' prefix
            cleaned_state_dict[clean_key] = value
    
    print(f"Extracted {len(cleaned_state_dict)} weight tensors")
    return cleaned_state_dict, hparams

def create_model_from_weights(state_dict, hparams):
    """Create model and load the extracted weights"""
    try:
        import segmentation_models_pytorch as smp
        
        if hparams['model'] == 'unet':
            model = smp.Unet(
                encoder_name=hparams['backbone'],
                encoder_weights=None,
                in_channels=hparams['in_channels'],
                classes=hparams['num_classes'],
            )
        else:
            raise ValueError(f"Model {hparams['model']} not supported")
        
        # Load the cleaned state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                print(f"Keys: {missing_keys}")
        
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                print(f"Keys: {unexpected_keys}")
        
        if not missing_keys and not unexpected_keys:
            print("All weights loaded successfully")
        elif len(missing_keys) < 10:
            print("Model loaded with minor issues")
        else:
            print("Warning: Too many missing keys - model may not work properly")
            return None
            
        model.eval()
        return model
        
    except ImportError:
        print("Error: segmentation_models_pytorch not available")
        return None

def run_inference_on_image(image_path, model_path, output_path, batch_size=2, patch_size=512, padding=64):
    """Run inference on a single image"""
    
    print(f"Processing: {image_path}")
    print(f"Output: {output_path}")
    
    # Check inputs
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
        
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    # Extract weights and create model
    state_dict, hparams = extract_model_weights(model_path)
    model = create_model_from_weights(state_dict, hparams)
    
    if model is None:
        return False
    
    # Create dataset and dataloader
    dataset = SimpleRasterDataset(image_path, patch_size=patch_size, padding=padding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get image info for output
    with rasterio.open(image_path) as src:
        height, width = src.shape
        profile = src.profile
        transform = src.transform
        tags = src.tags()
    
    # Initialize output array
    output_mask = np.zeros((height, width), dtype=np.uint8)
    stride = patch_size - padding * 2
    
    print(f"Running inference on {len(dataset)} patches...")
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing patches"):
            images = batch['image']
            window_infos = batch['window_info']
            
            # Forward pass
            predictions = model(images)
            predictions = predictions.argmax(dim=1).cpu().numpy()
            
            # Place predictions in output mask
            for i, (x, y, patch_width, patch_height) in enumerate(zip(*window_infos)):
                x, y = int(x), int(y)
                patch_width, patch_height = int(patch_width), int(patch_height)
                
                # Calculate output region (excluding padding)
                out_x1 = x + padding
                out_y1 = y + padding
                out_x2 = min(x + patch_width - padding, width)
                out_y2 = min(y + patch_height - padding, height)
                
                # Calculate prediction region (excluding padding)
                pred_x1 = padding
                pred_y1 = padding
                pred_x2 = pred_x1 + (out_x2 - out_x1)
                pred_y2 = pred_y1 + (out_y2 - out_y1)
                
                # Make sure we don't go out of bounds
                if out_x1 < width and out_y1 < height and out_x2 > out_x1 and out_y2 > out_y1:
                    output_mask[out_y1:out_y2, out_x1:out_x2] = predictions[i][pred_y1:pred_y2, pred_x1:pred_x2]
    
    # Save output
    profile.update({
        'driver': 'GTiff',
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'nodata': 0,
        'blockxsize': 512,
        'blockysize': 512,
        'tiled': True,
        'interleave': 'pixel'
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.update_tags(**tags)
        dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
        dst.colorinterp = [ColorInterp.palette]
        dst.write(output_mask, 1)
    
    print(f"Saved predictions to: {output_path}")
    return True

def test_checkpoint_content(checkpoint_path):
    """Test what's in the checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded successfully")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
            print(f"Model weight keys: {len(model_keys)}")
            print(f"First few model keys: {model_keys[:5]}")
            
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def main():
    """Main inference function"""
    print("Morocco Model Inference - Filtered Data Training")
    print("-" * 60)
    
    # Model paths (commented old ones for reference)
    # OLD MODELS (commented out for reference):
    # Original trained model:
    # model_path = r"logs\Morocco-FTW\lightning_logs\version_4\checkpoints\epoch=2-val_loss=0.67.ckpt"
    # Fine-tuned model (old):
    # model_path = r"logs\Morocco-FTW-Transfer\lightning_logs\version_4\checkpoints\epoch=100-val_loss=0.42.ckpt"
    
    # NEW FILTERED DATA MODEL (best checkpoint from your training):
    model_path = r"logs\Morocco-FTW-Filtered\lightning_logs\version_2\checkpoints\epoch=104-val_loss=0.41.ckpt"
    
    # Test checkpoint first
    print("Testing checkpoint file...")
    if not test_checkpoint_content(model_path):
        return
    
    # Same Sentinel-2 images as before
    images = [
        "morocco_mosaic.tif",
        "morocco_mid_mosaic.tif", 
        "morocco_tr_aoi.tif"
    ]
    
    # Process each image
    success_count = 0
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # Generate output name with clear filtered model indication
        base_name = os.path.splitext(img_path)[0]
        # OLD OUTPUT PATHS (commented out for reference):
        # output_path = f"{base_name}_morocco_trained_extracted.tif"
        # output_path = f"{base_name}_morocco_finetuned.tif"
        
        # NEW OUTPUT PATH for filtered model:
        output_path = f"{base_name}_morocco_filtered.tif"  # Clear naming for filtered model
        
        # Run inference
        start_time = time.time()
        success = run_inference_on_image(
            image_path=img_path,
            model_path=model_path,
            output_path=output_path,
            batch_size=2,
            patch_size=256,
            padding=32
        )
        
        if success:
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.1f} seconds")
            success_count += 1
        else:
            print(f"Failed to process {img_path}")
    
    print(f"Inference completed - {success_count} images processed")
    print("\nOutput files with filtered model:")
    for f in ["morocco_mosaic_morocco_filtered.tif", 
              "morocco_mid_mosaic_morocco_filtered.tif",
              "morocco_tr_aoi_morocco_filtered.tif"]:
        if os.path.exists(f):
            print(f"  ✓ {f}")

if __name__ == "__main__":
    main()