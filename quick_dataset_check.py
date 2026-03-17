#!/usr/bin/env python3
"""
Quick Dataset Checker - Handles list-type RFI data
"""

import os
import glob
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# UPDATE THESE PATHS!
# ============================================================================
FRB_H5_DIR = "FRB/realtime_data/FRBs/"
RFI_PKL_PATH = "FRB/RFIs/LOFAR_Full_RFI_dataset.pkl"

print("="*80)
print("QUICK DATASET VALIDATION")
print("="*80)

# ============================================================================
# 1. COUNT FRB FILES
# ============================================================================
print("\n1. FRB FILES:")
print("-"*80)
frb_files = sorted(glob.glob(os.path.join(FRB_H5_DIR, "*.h5")))
num_frb_files = len(frb_files)
print(f"Found {num_frb_files} FRB H5 files")

if num_frb_files > 0:
    print(f"\nFirst 3 files:")
    for i, f in enumerate(frb_files[:3]):
        print(f"  {i+1}. {os.path.basename(f)}")

# ============================================================================
# 2. INSPECT RFI PICKLE
# ============================================================================
print("\n2. RFI PICKLE:")
print("-"*80)
with open(RFI_PKL_PATH, 'rb') as f:
    rfi_data = pickle.load(f)

print(f"Type: {type(rfi_data)}")
print(f"Length: {len(rfi_data)}")

# Handle different structures
rfi_array = None
num_rfi = 0

if isinstance(rfi_data, list):
    print(f"\n✓ RFI is a LIST with {len(rfi_data)} elements")
    
    if len(rfi_data) > 0:
        first = rfi_data[0]
        print(f"  First element type: {type(first)}")
        
        if isinstance(first, np.ndarray):
            print(f"  First element shape: {first.shape}")
            print(f"  → List of numpy arrays")
            
            # Try to convert to single array
            try:
                rfi_array = np.array(rfi_data)
                print(f"  ✓ Converted to array: {rfi_array.shape}")
                num_rfi = len(rfi_array)
            except:
                try:
                    rfi_array = np.stack(rfi_data)
                    print(f"  ✓ Stacked to array: {rfi_array.shape}")
                    num_rfi = len(rfi_array)
                except Exception as e:
                    print(f"  ❌ Could not convert: {e}")
                    num_rfi = len(rfi_data)
        
        elif isinstance(first, dict):
            print(f"  First element keys: {list(first.keys())}")
            print(f"  → List of dictionaries")
            num_rfi = len(rfi_data)
        
        else:
            print(f"  → Unknown structure")
            num_rfi = len(rfi_data)

elif isinstance(rfi_data, dict):
    print(f"\n✓ RFI is a DICT with keys: {list(rfi_data.keys())}")
    for key in ['data', 'images', 'X']:
        if key in rfi_data and isinstance(rfi_data[key], np.ndarray):
            rfi_array = rfi_data[key]
            num_rfi = len(rfi_array)
            print(f"  Using key '{key}': shape={rfi_array.shape}")
            break

elif isinstance(rfi_data, np.ndarray):
    print(f"\n✓ RFI is a numpy array")
    rfi_array = rfi_data
    num_rfi = len(rfi_array)
    print(f"  Shape: {rfi_array.shape}")

# ============================================================================
# 3. CLASS IMBALANCE CHECK
# ============================================================================
print("\n3. CLASS IMBALANCE:")
print("-"*80)
total = num_frb_files + num_rfi
print(f"Total samples: {total:,}")
print(f"  FRB: {num_frb_files:,} ({100*num_frb_files/total:.2f}%)")
print(f"  RFI: {num_rfi:,} ({100*num_rfi/total:.2f}%)")

if num_frb_files > 0 and num_rfi > 0:
    ratio = max(num_frb_files, num_rfi) / min(num_frb_files, num_rfi)
    majority = "RFI" if num_rfi > num_frb_files else "FRB"
    
    print(f"\nImbalance: {ratio:.1f}:1 ({majority} is majority)")
    
    if ratio > 100:
        print("\n🔴 CRITICAL: EXTREME IMBALANCE (>100:1)")
        print(f"   Baseline accuracy by always predicting {majority}: {max(num_frb_files, num_rfi)/total*100:.1f}%")
        print("   → This is why your model predicts everything as one class!")
    elif ratio > 10:
        print("\n⚠️  SEVERE IMBALANCE (>10:1)")
        print("   → Must use class weights or resampling")
    elif ratio > 3:
        print("\n⚠️  MODERATE IMBALANCE (>3:1)")
    else:
        print("\n✓ Acceptable balance")

# ============================================================================
# 4. INSPECT SAMPLE FRB FILE
# ============================================================================
if num_frb_files > 0:
    print("\n4. SAMPLE FRB FILE STRUCTURE:")
    print("-"*80)
    with h5py.File(frb_files[0], 'r') as f:
        print(f"File: {os.path.basename(frb_files[0])}")
        print(f"Keys: {list(f.keys())}")
        
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data = f[key][:]
                print(f"\n{key}:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
                print(f"  Mean:  {data.mean():.4f}")
                print(f"  Zeros: {100*(data==0).sum()/data.size:.1f}%")
                
                if len(data.shape) == 4:
                    print(f"  ⚠️  Batch of {data.shape[0]} samples detected!")

# ============================================================================
# 5. VISUAL COMPARISON (if we can load samples)
# ============================================================================
if num_frb_files > 0 and rfi_array is not None:
    print("\n5. LOADING SAMPLE IMAGES FOR VISUALIZATION:")
    print("-"*80)
    
    # Load 1 FRB sample
    with h5py.File(frb_files[0], 'r') as f:
        for key in ['data_dm_time', 'dm_time', 'data']:
            if key in f:
                frb_img = f[key][:]
                if len(frb_img.shape) == 4:
                    frb_img = frb_img[0]
                break
    
    # Load 1 RFI sample
    if len(rfi_array.shape) >= 3:
        rfi_img = rfi_array[0]
    else:
        rfi_img = rfi_array
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(frb_img.squeeze(), aspect='auto', cmap='viridis')
    axes[0].set_title('FRB Sample', fontweight='bold', fontsize=14, color='blue')
    axes[0].axis('off')
    
    axes[1].imshow(rfi_img.squeeze(), aspect='auto', cmap='viridis')
    axes[1].set_title('RFI Sample', fontweight='bold', fontsize=14, color='red')
    axes[1].axis('off')
    
    plt.suptitle('Visual Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved sample_comparison.png")
    plt.show()

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n6. RECOMMENDATIONS:")
print("="*80)

if total > 0 and num_frb_files > 0 and num_rfi > 0:
    ratio = max(num_frb_files, num_rfi) / min(num_frb_files, num_rfi)
    
    if ratio > 10:
        print("\n🔧 REQUIRED FIX: Handle Class Imbalance")
        print("-"*80)
        print("\nOption 1: Use Class Weights (Recommended)")
        weight_minority = ratio
        print(f"  class_weight = {{0: 1.0, 1: {weight_minority:.1f}}}")
        print("  model.fit(..., class_weight=class_weight)")
        
        print("\nOption 2: Undersample Majority Class")
        target = min(num_frb_files, num_rfi) * 3
        print(f"  Keep only ~{target} samples from majority class")
        
        print("\nOption 3: Oversample Minority Class")
        print("  Use SMOTE or data augmentation")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
