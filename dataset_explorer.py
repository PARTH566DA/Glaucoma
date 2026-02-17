"""
Dataset Explorer for Chakshu Glaucoma Dataset
This script helps explore and validate the dataset structure.
"""

import os
import cv2
import sys
from collections import defaultdict

base_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/"

def count_files(directory):
    """Count files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')])

def explore_dataset():
    """Explore the complete dataset structure and provide statistics."""
    
    print("=" * 80)
    print("CHAKSHU GLAUCOMA DATASET EXPLORER")
    print("=" * 80)
    
    devices = ["Bosch", "Forus", "Remidio"]
    splits = ["Train", "Test"]
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for split in splits:
        print(f"\n{'='*80}")
        print(f"{split.upper()} DATASET")
        print(f"{'='*80}")
        
        for device in devices:
            # Count original images
            img_dir = os.path.join(base_path, split, '1.0_Original_Fundus_Images', device)
            img_count = count_files(img_dir)
            stats[split][device] = img_count
            
            # Count disc masks
            disc_dir = os.path.join(base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Disc', 'STAPLE')
            disc_count = count_files(disc_dir)
            
            # Count cup masks
            cup_dir = os.path.join(base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Cup', 'STAPLE')
            cup_count = count_files(cup_dir)
            
            print(f"\n{device}:")
            print(f"  - Original Images: {img_count}")
            print(f"  - Disc Masks (STAPLE): {disc_count}")
            print(f"  - Cup Masks (STAPLE): {cup_count}")
            
            if img_count != disc_count or img_count != cup_count:
                print(f"  ⚠️  WARNING: Mismatch in counts!")
            else:
                print(f"  ✓ All counts match")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_train = sum(stats['Train'].values())
    total_test = sum(stats['Test'].values())
    total = total_train + total_test
    
    print(f"\nTotal Images:")
    print(f"  - Train: {total_train}")
    print(f"  - Test: {total_test}")
    print(f"  - Total: {total}")
    
    print(f"\nBreakdown by Device:")
    for device in devices:
        train_count = stats['Train'][device]
        test_count = stats['Test'][device]
        device_total = train_count + test_count
        print(f"  - {device}: {device_total} (Train: {train_count}, Test: {test_count})")

def check_image_dimensions():
    """Check dimensions of images from each device."""
    print(f"\n{'='*80}")
    print("IMAGE DIMENSIONS CHECK")
    print(f"{'='*80}")
    
    devices = ["Bosch", "Forus", "Remidio"]
    splits = ["Train", "Test"]
    
    for split in splits:
        print(f"\n{split.upper()} Dataset:")
        for device in devices:
            img_dir = os.path.join(base_path, split, '1.0_Original_Fundus_Images', device)
            if os.path.isdir(img_dir):
                files = [f for f in os.listdir(img_dir) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    # Check first image
                    img_path = os.path.join(img_dir, files[0])
                    img = cv2.imread(img_path)
                    if img is not None:
                        print(f"  {device}: {img.shape[1]}x{img.shape[0]} (W x H) - {img.shape[2]} channels")
                        print(f"    Sample: {files[0]}")

def load_sample_data(split="Train", device="Remidio", image_idx=0):
    """Load a sample image and its masks."""
    print(f"\n{'='*80}")
    print(f"LOADING SAMPLE DATA: {split}/{device}")
    print(f"{'='*80}")
    
    # Get list of images
    img_dir = os.path.join(base_path, split, '1.0_Original_Fundus_Images', device)
    if not os.path.isdir(img_dir):
        print(f"Error: Directory not found: {img_dir}")
        return None
    
    files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        print(f"Error: No images found in {img_dir}")
        return None
    
    if image_idx >= len(files):
        image_idx = 0
    
    image_filename = files[image_idx]
    print(f"\nSelected image: {image_filename} (index {image_idx}/{len(files)-1})")
    
    # Load image
    img_path = os.path.join(img_dir, image_filename)
    fundus_image = cv2.imread(img_path)
    if fundus_image is None:
        print(f"Error: Could not load image: {img_path}")
        return None
    
    # Load masks
    mask_basename = os.path.splitext(image_filename)[0]
    disc_dir = os.path.join(base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Disc', 'STAPLE')
    cup_dir = os.path.join(base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE', device, 'Cup', 'STAPLE')
    
    # Load disc mask
    disc_mask = None
    disc_path = os.path.join(disc_dir, mask_basename + '.png')
    if os.path.isfile(disc_path):
        disc_mask = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
    
    # Load cup mask
    cup_mask = None
    cup_path = os.path.join(cup_dir, mask_basename + '.png')
    if os.path.isfile(cup_path):
        cup_mask = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"\nResults:")
    print(f"  - Fundus Image: {fundus_image.shape}")
    if disc_mask is not None:
        print(f"  - Disc Mask: {disc_mask.shape}")
        print(f"  - Disc Mask Range: [{disc_mask.min()}, {disc_mask.max()}]")
    else:
        print(f"  - Disc Mask: NOT FOUND")
    
    if cup_mask is not None:
        print(f"  - Cup Mask: {cup_mask.shape}")
        print(f"  - Cup Mask Range: [{cup_mask.min()}, {cup_mask.max()}]")
    else:
        print(f"  - Cup Mask: NOT FOUND")
    
    return {
        'image': fundus_image,
        'disc_mask': disc_mask,
        'cup_mask': cup_mask,
        'filename': image_filename
    }

def main():
    """Main function."""
    print("\n")
    
    # Explore dataset structure
    explore_dataset()
    
    # Check image dimensions
    check_image_dimensions()
    
    # Load sample data
    print("\n")
    for device in ["Bosch", "Forus", "Remidio"]:
        load_sample_data("Train", device, 0)
    
    print(f"\n{'='*80}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
