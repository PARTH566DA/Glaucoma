"""
Data Loader Utility for Chakshu Glaucoma Dataset
Provides easy-to-use functions for loading images and masks from the dataset.
"""

import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

class ChakshuDataLoader:
    """Data loader for Chakshu Glaucoma Dataset."""
    
    def __init__(self, base_path: str = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/"):
        """Initialize the data loader.
        
        Args:
            base_path: Base path to the dataset directory
        """
        self.base_path = base_path
        self.devices = ["Bosch", "Forus", "Remidio"]
        self.splits = ["Train", "Test"]
        
    def get_image_list(self, split: str = "Train", device: str = "Remidio") -> List[str]:
        """Get list of all image filenames for a device and split.
        
        Args:
            split: Dataset split ("Train" or "Test")
            device: Device name ("Bosch", "Forus", or "Remidio")
            
        Returns:
            List of image filenames
        """
        img_dir = os.path.join(self.base_path, split, '1.0_Original_Fundus_Images', device)
        if not os.path.isdir(img_dir):
            return []
        
        files = [f for f in os.listdir(img_dir) 
                if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        return sorted(files)
    
    def load_image(self, filename: str, split: str = "Train", device: str = "Remidio", 
                   color_mode: str = "RGB") -> Optional[np.ndarray]:
        """Load a fundus image.
        
        Args:
            filename: Image filename
            split: Dataset split ("Train" or "Test")
            device: Device name ("Bosch", "Forus", or "Remidio")
            color_mode: Color mode ("RGB", "BGR", or "GRAY")
            
        Returns:
            Image as numpy array or None if not found
        """
        img_dir = os.path.join(self.base_path, split, '1.0_Original_Fundus_Images', device)
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.isfile(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None
        
        if color_mode == "GRAY":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
            if img is not None and color_mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def load_masks(self, filename: str, split: str = "Train", device: str = "Remidio",
                   mask_type: str = "STAPLE") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load disc and cup masks for an image.
        
        Args:
            filename: Image filename (without extension or with it)
            split: Dataset split ("Train" or "Test")
            device: Device name ("Bosch", "Forus", or "Remidio")
            mask_type: Mask type ("STAPLE", "Mean", "Median", "Majority")
            
        Returns:
            Tuple of (disc_mask, cup_mask) as numpy arrays, or None if not found
        """
        mask_basename = os.path.splitext(filename)[0]
        
        # Load disc mask
        disc_dir = os.path.join(self.base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE',
                               device, 'Disc', mask_type)
        disc_path = os.path.join(disc_dir, mask_basename + '.png')
        disc_mask = None
        if os.path.isfile(disc_path):
            disc_mask = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
        
        # Load cup mask
        cup_dir = os.path.join(self.base_path, split, '5.0_OD_OC_Mean_Median_Majority_STAPLE',
                              device, 'Cup', mask_type)
        cup_path = os.path.join(cup_dir, mask_basename + '.png')
        cup_mask = None
        if os.path.isfile(cup_path):
            cup_mask = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
        
        return disc_mask, cup_mask
    
    def load_sample(self, filename: str, split: str = "Train", device: str = "Remidio",
                   color_mode: str = "RGB", mask_type: str = "STAPLE") -> Dict:
        """Load an image and its corresponding masks.
        
        Args:
            filename: Image filename
            split: Dataset split ("Train" or "Test")
            device: Device name ("Bosch", "Forus", or "Remidio")
            color_mode: Color mode for image ("RGB", "BGR", or "GRAY")
            mask_type: Mask type ("STAPLE", "Mean", "Median", "Majority")
            
        Returns:
            Dictionary containing 'image', 'disc_mask', 'cup_mask', and 'filename'
        """
        image = self.load_image(filename, split, device, color_mode)
        disc_mask, cup_mask = self.load_masks(filename, split, device, mask_type)
        
        return {
            'image': image,
            'disc_mask': disc_mask,
            'cup_mask': cup_mask,
            'filename': filename,
            'split': split,
            'device': device
        }
    
    def load_all_samples(self, split: str = "Train", device: str = "Remidio",
                        color_mode: str = "RGB", mask_type: str = "STAPLE",
                        limit: Optional[int] = None) -> List[Dict]:
        """Load all samples for a given split and device.
        
        Args:
            split: Dataset split ("Train" or "Test")
            device: Device name ("Bosch", "Forus", or "Remidio")
            color_mode: Color mode for images ("RGB", "BGR", or "GRAY")
            mask_type: Mask type ("STAPLE", "Mean", "Median", "Majority")
            limit: Maximum number of samples to load (None for all)
            
        Returns:
            List of sample dictionaries
        """
        filenames = self.get_image_list(split, device)
        
        if limit is not None:
            filenames = filenames[:limit]
        
        samples = []
        for i, filename in enumerate(filenames):
            if (i + 1) % 100 == 0:
                print(f"Loading {split}/{device}: {i+1}/{len(filenames)}")
            
            sample = self.load_sample(filename, split, device, color_mode, mask_type)
            
            # Only add if image loaded successfully
            if sample['image'] is not None:
                samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {split}/{device}")
        return samples
    
    def calculate_cdr(self, disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
        """Calculate Cup-to-Disc Ratio (CDR).
        
        Args:
            disc_mask: Binary disc mask
            cup_mask: Binary cup mask
            
        Returns:
            CDR value (cup area / disc area)
        """
        if disc_mask is None or cup_mask is None:
            return -1.0
        
        disc_area = np.sum(disc_mask > 0)
        cup_area = np.sum(cup_mask > 0)
        
        if disc_area == 0:
            return -1.0
        
        return cup_area / disc_area


def demo():
    """Demonstration of the data loader."""
    print("=" * 80)
    print("CHAKSHU DATA LOADER DEMO")
    print("=" * 80)
    
    # Initialize loader
    loader = ChakshuDataLoader()
    
    # Example 1: Load a single sample
    print("\n1. Loading a single sample...")
    sample = loader.load_sample("IMG_2431.JPG", split="Train", device="Remidio")
    
    if sample['image'] is not None:
        print(f"   Loaded: {sample['filename']}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Disc mask shape: {sample['disc_mask'].shape if sample['disc_mask'] is not None else 'None'}")
        print(f"   Cup mask shape: {sample['cup_mask'].shape if sample['cup_mask'] is not None else 'None'}")
        
        # Calculate CDR
        if sample['disc_mask'] is not None and sample['cup_mask'] is not None:
            cdr = loader.calculate_cdr(sample['disc_mask'], sample['cup_mask'])
            print(f"   Cup-to-Disc Ratio (CDR): {cdr:.4f}")
    
    # Example 2: Get list of images
    print("\n2. Getting list of images...")
    for device in loader.devices:
        for split in loader.splits:
            images = loader.get_image_list(split, device)
            print(f"   {split}/{device}: {len(images)} images")
    
    # Example 3: Load multiple samples
    print("\n3. Loading multiple samples (5 from Train/Bosch)...")
    samples = loader.load_all_samples(split="Train", device="Bosch", limit=5)
    
    for i, sample in enumerate(samples):
        cdr = loader.calculate_cdr(sample['disc_mask'], sample['cup_mask'])
        print(f"   [{i+1}] {sample['filename']}: CDR = {cdr:.4f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo()
