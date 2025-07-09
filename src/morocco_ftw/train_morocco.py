#This script is used for training


    #!/usr/bin/env python3

import multiprocessing
from ftw_cli.model import fit

def main():
    print("Starting Morocco FTW Training...")
    
    # Calculate and display dataset statistics
    import os
    import glob
    
    train_masks = glob.glob("data/ftw/morocco/label_masks/semantic_3class/*.tif")
    print(f"Total patches: {len(train_masks)} (28 train, 3 val, 3 test)")
    
    print("Running fit command")
    print("CLI arguments: ['fit', '--config=morocco_config.yaml']")
    
    # Call the fit function with your config
    fit(
        config="morocco_config.yaml",
        ckpt_path=None,
        cli_args=[]
    )
    
    print("Training completed!")

if __name__ == "__main__":
    # This is crucial for Windows multiprocessing
    multiprocessing.freeze_support()
    main()