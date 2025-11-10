"""
Script untuk memeriksa dataset sebelum training
Jalankan: python check_dataset.py
"""
import os
import torch
from src.datasets.restoration_dataset import RestorationDataset

def check_dataset():
    print("="*60)
    print("Checking dataset integrity...")
    print("="*60)
    
    # Setup paths
    output_root = "outputs"
    targets = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    
    train_list = []
    for t in targets[:3]:  # Check first 3 datasets
        base = os.path.join(output_root, t)
        images_dir = os.path.join(base, "images")
        depth_dir = os.path.join(base, "dense", "depth_maps")
        if os.path.isdir(images_dir) and os.path.isdir(depth_dir):
            train_list.append((images_dir, depth_dir))
    
    if not train_list:
        print("❌ No dataset found!")
        return
    
    # Create dataset
    ds = RestorationDataset(pairs=train_list, augment=True)
    print(f"\n✅ Dataset loaded with {len(ds)} samples")
    
    # Check first 5 samples
    print("\nChecking samples...")
    for i in range(min(5, len(ds))):
        try:
            sample = ds[i]
            print(f"\nSample {i}:")
            for k, v in sample.items():
                is_finite = torch.isfinite(v).all().item()
                vmin = v.min().item()
                vmax = v.max().item()
                status = "✅" if is_finite and -1.1 <= vmin <= 1.1 and -1.1 <= vmax <= 1.1 else "❌"
                print(f"  {status} {k:15s} | shape: {str(v.shape):20s} | finite: {is_finite} | range: [{vmin:.3f}, {vmax:.3f}]")
        except Exception as e:
            print(f"❌ Error on sample {i}: {e}")
    
    print("\n" + "="*60)
    print("Dataset check completed!")
    print("="*60)

if __name__ == "__main__":
    check_dataset()
