#!/usr/bin/env python3
"""
Morocco FTW Model Fine-Tuning Script - Filtered Data
Adapts pre-trained FTW model to filtered Morocco-specific data
"""

import multiprocessing
import os
import glob
from ftw_cli.model import fit
from pathlib import Path  
BASE_PATH = Path(os.environ.get('FTW_BASE_PATH', Path(__file__).parent.parent.parent))
print(f"Using base path: {BASE_PATH}")

def main():
    print("Starting Morocco model fine-tuning with filtered data...")
    print("-" * 60)
    
    # Configuration setup
    pretrained_model = BASE_PATH / "3_Class_CCBY_FTW_Pretrained.ckpt"
    config_file = BASE_PATH/ "morocco_config.yaml"
    
    # Verify required files exist
    if not os.path.exists(pretrained_model):
        print(f"Error: Missing pre-trained model file - {pretrained_model}")
        print("\nTo download it, run:")
        print("wget https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/3_Class_CCBY_FTW_Pretrained.ckpt")
        return 1
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found - {config_file}")
        print("Please ensure morocco_config.yaml exists in this directory")
        return 1
    
    # Verify Morocco data directory exists
    morocco_data_path = BASE_PATH / "data" / "ftw" / "morocco"
    if not os.path.exists(morocco_data_path):
        print(f"Error: Morocco data directory not found - {morocco_data_path}")
        print("Please ensure the filtered Morocco data is available at the specified path")
        return 1
    
    print(f"Using pre-trained model: {pretrained_model}")
    print(f"Using config file: {config_file}")
    print(f"Using filtered Morocco data from: {morocco_data_path}")
    
    # Training details
    print("\nTraining configuration:")
    print(f"- Loading weights from: {pretrained_model}")
    print(f"- Configuration from: {config_file}")
    print(f"- Data source: Filtered Morocco dataset")
    print(f"- Learning rate set to: 1e-4")
    print(f"- Training for max 150 epochs")
    print(f"- Logs will be saved to: logs/Morocco-FTW-Filtered/")
    
    try:
        print("\nBeginning fine-tuning process with filtered Morocco data...")
        
        # Core training call
        fit(
            config=config_file,
            ckpt_path=pretrained_model,
            cli_args=[]
        )
        
        print("\nFine-tuning with filtered data complete!")
        print("\nNext steps:")
        print("1. Review training logs in BASE_PATH / "logs" / "Morocco-FTW-Filtered" / "lightning_logs"")
        print("2. Locate best model in the checkpoints directory")
        print("3. Use for inference on Morocco images")
        print("4. Compare performance with original model and unfiltered data")
        
        # Show generated checkpoints
        checkpoints = glob.glob(str(BASE_PATH / "logs" / "Morocco-FTW-Filtered" / "lightning_logs" / "*" / "checkpoints" / "*.ckpt"))
        if checkpoints:
            print("\nGenerated model checkpoints:")
            for ckpt in checkpoints:
                print(f"- {ckpt}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your filtered data format and paths")
        print("2. Check config file values")
        print("3. Ensure model file isn't corrupted")
        print("4. Verify Morocco directory structure matches FTW expected format")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    exit(main())