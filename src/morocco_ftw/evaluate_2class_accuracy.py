#!/usr/bin/env python3
"""
Evaluate 2-class model predictions against 3-class ground truth
Calculates all 5 FTW metrics: Pixel IoU, Pixel Precision, Pixel Recall, Object Precision, Object Recall
Simplified version using only numpy and scipy
"""

import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
from scipy import ndimage

def convert_3class_to_2class(gt_3class):
    """Convert 3-class GT (0=bg, 1=field, 2=boundary) to 2-class (0=bg, 1=field+boundary)"""
    gt_2class = np.zeros_like(gt_3class)
    gt_2class[gt_3class > 0] = 1  # Combine field and boundary as "field"
    return gt_2class

def calculate_object_level_metrics(gt_array, pred_array, iou_threshold=0.5):
    """Calculate object-level precision and recall using scipy ndimage"""
    
    # Label connected components (fields) using scipy
    gt_labeled, gt_n_objects = ndimage.label(gt_array)
    pred_labeled, pred_n_objects = ndimage.label(pred_array)
    
    if gt_n_objects == 0 or pred_n_objects == 0:
        return 0.0, 0.0  # No objects to compare
    
    # Find object properties using scipy
    gt_objects = ndimage.find_objects(gt_labeled)
    pred_objects = ndimage.find_objects(pred_labeled)
    
    # Match predicted objects to ground truth objects
    matched_gt = set()
    matched_pred = set()
    
    for pred_idx in range(1, pred_n_objects + 1):  # Object labels start from 1
        pred_mask = (pred_labeled == pred_idx)
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(1, gt_n_objects + 1):
            if gt_idx in matched_gt:
                continue
                
            gt_mask = (gt_labeled == gt_idx)
            
            # Calculate IoU between this predicted object and GT object
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_gt_idx > 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
    
    # Calculate object-level metrics
    n_matched = len(matched_pred)
    
    object_precision = n_matched / pred_n_objects if pred_n_objects > 0 else 0
    object_recall = n_matched / gt_n_objects if gt_n_objects > 0 else 0
    
    return object_precision, object_recall

def calculate_2class_metrics(gt_array, pred_array, area_name):
    """Calculate all FTW metrics for 2-class comparison"""
    
    # Convert 3-class GT to 2-class
    gt_2class = convert_3class_to_2class(gt_array)
    
    # Flatten arrays for pixel-level metrics
    gt_flat = gt_2class.flatten()
    pred_flat = pred_array.flatten()
    
    # Calculate pixel-level confusion matrix elements
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    
    # Pixel-level metrics
    pixel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    pixel_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    pixel_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Object-level metrics
    object_precision, object_recall = calculate_object_level_metrics(gt_2class, pred_array)
    
    return {
        'area': area_name,
        'pixel_iou': pixel_iou,
        'pixel_precision': pixel_precision,
        'pixel_recall': pixel_recall,
        'object_precision': object_precision,
        'object_recall': object_recall,
        'pixel_accuracy': pixel_accuracy,
        'total_pixels': len(gt_flat),
        'field_pixels_gt': np.sum(gt_flat),
        'field_pixels_pred': np.sum(pred_flat)
    }

def crop_to_match(pred_src, gt_src):
    """Crop prediction to match ground truth extent"""
    from rasterio.warp import reproject, Resampling
    
    gt_array = gt_src.read(1)
    pred_cropped = np.zeros_like(gt_array)
    
    reproject(
        source=rasterio.band(pred_src, 1),
        destination=pred_cropped,
        src_transform=pred_src.transform,
        src_crs=pred_src.crs,
        dst_transform=gt_src.transform,
        dst_crs=gt_src.crs,
        resampling=Resampling.nearest
    )
    
    return pred_cropped

def evaluate_2class_model():
    """Evaluate 2-class model predictions"""
    
    print("2-Class Model Evaluation")
    print("=" * 30)
    
    # Your 2-class prediction files and corresponding 3-class ground truth
    eval_pairs = [
        ("morocco_mosaic-inf.tif",
         "morocco_filtered_3class_Morocco1_BL_aligned.tif",
         "Morocco_Mosaic"),
        ("morocco_mid_mosaic-inf.tif",
         "morocco_filtered_3class_Stef_BR_aligned.tif",
         "Mid_Mosaic"),
        ("morocco_tr_aoi-inf.tif", 
         "morocco_filtered_3class_Morocco2_TR_aligned.tif",
         "TR_AOI")
    ]
    
    base_dir = Path(r"C:\Users\qin.xu\github\ftw-baselines")
    results = []
    
    for pred_file, gt_file, area_name in eval_pairs:
        
        pred_path = base_dir / pred_file
        gt_path = base_dir / gt_file
        
        print(f"\nEvaluating {area_name}...")
        
        if not pred_path.exists():
            print(f"  Prediction not found: {pred_path}")
            continue
            
        if not gt_path.exists():
            print(f"  Ground truth not found: {gt_path}")
            continue
        
        try:
            with rasterio.open(pred_path) as pred_src, rasterio.open(gt_path) as gt_src:
                gt_array = gt_src.read(1)
                
                if pred_src.shape != gt_src.shape:
                    print(f"  Cropping prediction to match GT extent")
                    pred_array = crop_to_match(pred_src, gt_src)
                else:
                    pred_array = pred_src.read(1)
                
                print(f"  GT shape: {gt_array.shape}, Pred shape: {pred_array.shape}")
                print(f"  GT classes: {np.unique(gt_array)}")
                print(f"  Pred classes: {np.unique(pred_array)}")
            
            metrics = calculate_2class_metrics(gt_array, pred_array, area_name)
            
            if metrics:
                print(f"  Pixel IoU: {metrics['pixel_iou']:.4f}")
                print(f"  Pixel Precision: {metrics['pixel_precision']:.4f}")
                print(f"  Pixel Recall: {metrics['pixel_recall']:.4f}")
                print(f"  Object Precision: {metrics['object_precision']:.4f}")
                print(f"  Object Recall: {metrics['object_recall']:.4f}")
                results.append(metrics)
            
        except Exception as e:
            print(f"  Error processing {area_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\nNo successful evaluations!")
        return
    
    # Summary
    print(f"\n2-Class Model Results")
    print("-" * 45)
    
    df = pd.DataFrame(results)
    print(f"{'Area':<15} {'Pixel IoU':<10} {'Pixel Prec':<11} {'Pixel Rec':<10} {'Obj Prec':<9} {'Obj Rec':<8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['area']:<15} {row['pixel_iou']:<10.4f} {row['pixel_precision']:<11.4f} {row['pixel_recall']:<10.4f} {row['object_precision']:<9.4f} {row['object_recall']:<8.4f}")
    
    # Calculate averages
    if len(results) > 1:
        avg_pixel_iou = df['pixel_iou'].mean()
        avg_pixel_prec = df['pixel_precision'].mean()
        avg_pixel_rec = df['pixel_recall'].mean()
        avg_obj_prec = df['object_precision'].mean()
        avg_obj_rec = df['object_recall'].mean()
        
        print("-" * 70)
        print(f"{'AVERAGE':<15} {avg_pixel_iou:<10.4f} {avg_pixel_prec:<11.4f} {avg_pixel_rec:<10.4f} {avg_obj_prec:<9.4f} {avg_obj_rec:<8.4f}")
    
    # Save results
    df.to_csv('morocco_2class_results.csv', index=False)
    print(f"\nResults saved to morocco_2class_results.csv")
    
    return df

def compare_all_models():
    """Compare all three models with complete metrics"""
    
    print(f"\n" + "="*70)
    print("COMPLETE MODEL COMPARISON - ALL 5 METRICS")
    print("="*70)
    
    print(f"""
📊 Official FTW Test Results (3 test areas combined):

3-Class Baseline:
  Pixel IoU: 0.0653 | Pixel Precision: 0.2925 | Pixel Recall: 0.0776
  Object Precision: 0.0040 | Object Recall: 0.0048

Your Fine-tuned 3-Class Model:
  Pixel IoU: 0.1540 | Pixel Precision: 0.2757 | Pixel Recall: 0.2587
  Object Precision: 0.0649 | Object Recall: 0.0905

2-Class Baseline (from script above):
  [Results will be shown after running the script]
    """)

if __name__ == "__main__":
    
    # Run 2-class evaluation
    results_2class = evaluate_2class_model()
    
    # Show comparison
    compare_all_models()
    
    print(f"\nNotes:")
    print("- 2-class model tested on converted 2-class ground truth")
    print("- 3-class models tested on original 3-class ground truth") 
    print("- Comparison shows relative performance, not absolute")