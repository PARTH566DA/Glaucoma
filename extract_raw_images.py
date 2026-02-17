#!/usr/bin/env python3
"""
Feature Extraction for Raw Glaucoma Images
Extracts features from raw fundus images without pre-existing masks.
Uses OpenCV-based segmentation to detect optic disc and cup.
"""

import cv2
import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, Tuple, List


class RawImageFeatureExtractor:
    """Extract glaucoma-related features from raw fundus images."""
    
    def __init__(self, raw_images_path: str):
        """Initialize feature extractor for raw images.
        
        Args:
            raw_images_path: Path to the raw images directory
        """
        self.raw_images_path = raw_images_path
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fundus image for segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_optic_disc(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect optic disc region using brightness-based segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (disc_mask, bounding_box)
        """
        # Preprocess
        enhanced = self.preprocess_image(image)
        
        # The optic disc is typically the brightest region
        # Apply threshold to detect bright regions
        _, bright_regions = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_regions = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)
        bright_regions = cv2.morphologyEx(bright_regions, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(bright_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no bright region found, use center region as fallback
            h, w = enhanced.shape
            disc_mask = np.zeros_like(enhanced)
            cv2.circle(disc_mask, (w//2, h//2), min(w, h)//8, 255, -1)
            bbox = (w//2 - min(w, h)//8, h//2 - min(w, h)//8, min(w, h)//4, min(w, h)//4)
            return disc_mask, bbox
        
        # Get the largest bright region (likely optic disc)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        disc_mask = np.zeros_like(enhanced)
        cv2.drawContours(disc_mask, [largest_contour], -1, 255, -1)
        
        # Get bounding box
        bbox = cv2.boundingRect(largest_contour)
        
        return disc_mask, bbox
    
    def detect_optic_cup(self, image: np.ndarray, disc_mask: np.ndarray, 
                        disc_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect optic cup within the disc region.
        
        Args:
            image: Input BGR image
            disc_mask: Optic disc mask
            disc_bbox: Disc bounding box (x, y, w, h)
            
        Returns:
            Tuple of (cup_mask, bounding_box)
        """
        # Extract disc region
        x, y, w, h = disc_bbox
        disc_region = image[y:y+h, x:x+w].copy()
        disc_mask_region = disc_mask[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(disc_region, cv2.COLOR_BGR2GRAY)
        
        # The cup is typically the brightest part within the disc
        # Apply adaptive threshold
        _, cup_region = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cup_region = cv2.morphologyEx(cup_region, cv2.MORPH_CLOSE, kernel)
        cup_region = cv2.morphologyEx(cup_region, cv2.MORPH_OPEN, kernel)
        
        # Mask with disc mask
        cup_region = cv2.bitwise_and(cup_region, disc_mask_region)
        
        # Create full-size cup mask
        cup_mask = np.zeros_like(disc_mask)
        cup_mask[y:y+h, x:x+w] = cup_region
        
        # Find contours for bounding box
        contours, _ = cv2.findContours(cup_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(largest_contour)
            cup_bbox = (x + cup_x, y + cup_y, cup_w, cup_h)
        else:
            # Fallback: assume cup is ~60% of disc size in center
            cup_w, cup_h = int(w * 0.6), int(h * 0.6)
            cup_x, cup_y = x + (w - cup_w) // 2, y + (h - cup_h) // 2
            cup_bbox = (cup_x, cup_y, cup_w, cup_h)
            
            # Create a circular cup mask
            center = (x + w//2, y + h//2)
            radius = min(cup_w, cup_h) // 2
            cv2.circle(cup_mask, center, radius, 255, -1)
        
        return cup_mask, cup_bbox
    
    def calculate_area(self, mask: np.ndarray) -> int:
        """Calculate area of a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Area in pixels
        """
        return np.sum(mask > 128)
    
    def extract_features_from_image(self, filename: str) -> Optional[Dict]:
        """Extract all features from a raw image.
        
        Args:
            filename: Image filename
            
        Returns:
            Dictionary of features or None if processing failed
        """
        # Load image
        image_path = os.path.join(self.raw_images_path, filename)
        
        if not os.path.isfile(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {filename}")
            return None
        
        try:
            # Detect optic disc
            disc_mask, disc_bbox = self.detect_optic_disc(image)
            
            # Detect optic cup
            cup_mask, cup_bbox = self.detect_optic_cup(image, disc_mask, disc_bbox)
            
            # Calculate areas
            disc_area = self.calculate_area(disc_mask)
            cup_area = self.calculate_area(cup_mask)
            rim_area = disc_area - cup_area
            
            # Extract dimensions from bounding boxes
            disc_x, disc_y, disc_width, disc_height = disc_bbox
            cup_x, cup_y, cup_width, cup_height = cup_bbox
            
            # Calculate CDR metrics
            # ACDR: Area Cup-to-Disc Ratio
            acdr = cup_area / disc_area if disc_area > 0 else 0.0
            
            # VCDR: Vertical Cup-to-Disc Ratio
            vcdr = cup_height / disc_height if disc_height > 0 else 0.0
            
            # HCDR: Horizontal Cup-to-Disc Ratio
            hcdr = cup_width / disc_width if disc_width > 0 else 0.0
            
            # Determine glaucoma prediction based on CDR thresholds
            # Clinical thresholds: VCDR > 0.6 or ACDR > 0.4 suggests glaucoma
            if acdr > 0.4 or vcdr > 0.6:
                glaucoma_decision = "GLAUCOMA POSITIVE"
                risk_level = "HIGH"
            elif acdr > 0.3 or vcdr > 0.5:
                glaucoma_decision = "GLAUCOMA SUSPECT"
                risk_level = "MEDIUM"
            else:
                glaucoma_decision = "NORMAL"
                risk_level = "LOW"
            
            # Create feature dictionary
            features = {
                'Image_Filename': filename,
                'Patient_ID': os.path.splitext(filename)[0],
                'Disc_Area': disc_area,
                'Cup_Area': cup_area,
                'Rim_Area': rim_area,
                'Cup_Height': cup_height,
                'Cup_Width': cup_width,
                'Disc_Height': disc_height,
                'Disc_Width': disc_width,
                'ACDR': round(acdr, 4),
                'VCDR': round(vcdr, 4),
                'HCDR': round(hcdr, 4),
                'Glaucoma_Prediction': glaucoma_decision,
                'Risk_Level': risk_level,
                'Processing_Status': 'SUCCESS'
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return {
                'Image_Filename': filename,
                'Patient_ID': os.path.splitext(filename)[0],
                'Processing_Status': 'FAILED',
                'Error': str(e)
            }
    
    def extract_all_features(self, output_csv: str = "raw_images_features.csv") -> pd.DataFrame:
        """Extract features from all raw images.
        
        Args:
            output_csv: Output CSV filename
            
        Returns:
            DataFrame with all extracted features
        """
        print("=" * 80)
        print("EXTRACTING FEATURES FROM RAW GLAUCOMA IMAGES")
        print("=" * 80)
        print(f"Image directory: {self.raw_images_path}")
        
        # Get list of images
        image_files = [f for f in os.listdir(self.raw_images_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                      and not f.startswith('.')]
        image_files = sorted(image_files)
        
        print(f"Found {len(image_files)} images to process")
        print("=" * 80)
        
        features_list = []
        
        for i, filename in enumerate(image_files):
            print(f"Processing [{i+1}/{len(image_files)}]: {filename}", end="")
            
            features = self.extract_features_from_image(filename)
            
            if features is not None:
                features_list.append(features)
                if features.get('Processing_Status') == 'SUCCESS':
                    print(f" ✓ ACDR={features['ACDR']:.3f}, VCDR={features['VCDR']:.3f} - {features['Glaucoma_Prediction']}")
                else:
                    print(" ✗ FAILED")
            else:
                print(" ✗ SKIPPED")
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Successfully processed: {len([f for f in features_list if f.get('Processing_Status') == 'SUCCESS'])}")
        print(f"Failed: {len([f for f in features_list if f.get('Processing_Status') != 'SUCCESS'])}")
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(self.raw_images_path), output_csv)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to: {output_path}")
        
        # Print summary statistics
        if 'Glaucoma_Prediction' in df.columns:
            print("\n" + "=" * 80)
            print("GLAUCOMA PREDICTION SUMMARY")
            print("=" * 80)
            prediction_counts = df['Glaucoma_Prediction'].value_counts()
            for prediction, count in prediction_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {prediction:25s}: {count:3d} ({percentage:5.1f}%)")
            
            # CDR Statistics
            if 'ACDR' in df.columns and 'VCDR' in df.columns:
                successful_df = df[df['Processing_Status'] == 'SUCCESS']
                if len(successful_df) > 0:
                    print("\n" + "=" * 80)
                    print("CDR STATISTICS")
                    print("=" * 80)
                    print(f"  ACDR - Mean: {successful_df['ACDR'].mean():.4f}, "
                          f"Std: {successful_df['ACDR'].std():.4f}, "
                          f"Min: {successful_df['ACDR'].min():.4f}, "
                          f"Max: {successful_df['ACDR'].max():.4f}")
                    print(f"  VCDR - Mean: {successful_df['VCDR'].mean():.4f}, "
                          f"Std: {successful_df['VCDR'].std():.4f}, "
                          f"Min: {successful_df['VCDR'].min():.4f}, "
                          f"Max: {successful_df['VCDR'].max():.4f}")
        
        return df


def main():
    """Main function."""
    # Path to raw images
    raw_images_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Glaucoma/raw images(glaucoma+ve))"
    
    # Check if path exists
    if not os.path.isdir(raw_images_path):
        print(f"Error: Directory not found: {raw_images_path}")
        return
    
    # Initialize extractor
    extractor = RawImageFeatureExtractor(raw_images_path)
    
    # Extract features and save to CSV
    df = extractor.extract_all_features(output_csv="raw_glaucoma_images_features.csv")
    
    print("\n" + "=" * 80)
    print("SAMPLE OF EXTRACTED FEATURES (First 10 rows)")
    print("=" * 80)
    if len(df) > 0:
        # Display key columns
        display_cols = ['Image_Filename', 'ACDR', 'VCDR', 'HCDR', 
                       'Glaucoma_Prediction', 'Risk_Level']
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNote: These images are from 'raw images(glaucoma+ve)' folder,")
    print("indicating they are known glaucoma positive cases.")
    print("The predictions are based on automated CDR analysis.")


if __name__ == "__main__":
    main()
