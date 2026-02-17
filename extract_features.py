"""
Feature Extraction Script for Chakshu Glaucoma Dataset
Extracts features from fundus images matching the Remidio.csv feature set:
- Disc Area, Cup Area, Rim Area
- Cup Height, Cup Width, Disc Height, Disc Width
- ACDR (Area CDR), VCDR (Vertical CDR), HCDR (Horizontal CDR)
"""

import cv2
import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, Tuple
from data_loader import ChakshuDataLoader


class FeatureExtractor:
    """Extract glaucoma-related features from fundus images and masks."""
    
    def __init__(self, base_path: str = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/"):
        """Initialize feature extractor.
        
        Args:
            base_path: Base path to the dataset directory
        """
        self.base_path = base_path
        self.loader = ChakshuDataLoader(base_path)
    
    def get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (x, y, width, height)
        """
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0, 0, 0)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def calculate_area(self, mask: np.ndarray) -> int:
        """Calculate area of a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Area in pixels
        """
        if mask is None:
            return 0
        return np.sum(mask > 128)
    
    def extract_features(self, filename: str, split: str = "Train", 
                        device: str = "Remidio", mask_type: str = "STAPLE") -> Optional[Dict]:
        """Extract all features from an image.
        
        Args:
            filename: Image filename
            split: Dataset split ("Train" or "Test")
            device: Device name
            mask_type: Type of mask to use
            
        Returns:
            Dictionary of features or None if masks not available
        """
        # Load image and masks
        sample = self.loader.load_sample(filename, split, device, mask_type=mask_type)
        
        if sample['image'] is None:
            print(f"Warning: Could not load image {filename}")
            return None
        
        disc_mask = sample['disc_mask']
        cup_mask = sample['cup_mask']
        
        if disc_mask is None or cup_mask is None:
            print(f"Warning: Masks not available for {filename}")
            return None
        
        # Calculate areas
        disc_area = self.calculate_area(disc_mask)
        cup_area = self.calculate_area(cup_mask)
        rim_area = disc_area - cup_area
        
        # Get bounding boxes
        disc_x, disc_y, disc_width, disc_height = self.get_bounding_box(disc_mask)
        cup_x, cup_y, cup_width, cup_height = self.get_bounding_box(cup_mask)
        
        # Calculate CDR metrics
        # ACDR: Area Cup-to-Disc Ratio
        acdr = cup_area / disc_area if disc_area > 0 else 0.0
        
        # VCDR: Vertical Cup-to-Disc Ratio
        vcdr = cup_height / disc_height if disc_height > 0 else 0.0
        
        # HCDR: Horizontal Cup-to-Disc Ratio
        hcdr = cup_width / disc_width if disc_width > 0 else 0.0
        
        # Determine glaucoma decision based on CDR
        # Typically: ACDR > 0.4 or VCDR > 0.6 suggests glaucoma suspect
        if acdr > 0.4 or vcdr > 0.6:
            glaucoma_decision = "GLAUCOMA  SUSUPECT"  # Note: matching CSV typo
        else:
            glaucoma_decision = "NORMAL"
        
        # Create feature dictionary
        features = {
            'Images': filename,
            'Disc Area': disc_area,
            'Cup Area': cup_area,
            'Rim Area': rim_area,
            'Cup Height': cup_height,
            'Cup Width': cup_width,
            'Disc Height': disc_height,
            'Disc Width': disc_width,
            'ACDR': round(acdr, 3),
            'VCDR': round(vcdr, 3),
            'HCDR': round(hcdr, 3),
            'Glaucoma Decision': glaucoma_decision
        }
        
        return features
    
    def extract_all_features(self, split: str = "Train", device: str = "Remidio",
                            mask_type: str = "STAPLE", output_csv: Optional[str] = None) -> pd.DataFrame:
        """Extract features for all images in a dataset.
        
        Args:
            split: Dataset split ("Train" or "Test")
            device: Device name
            mask_type: Type of mask to use
            output_csv: Path to save output CSV (optional)
            
        Returns:
            DataFrame with all extracted features
        """
        print("=" * 80)
        print(f"EXTRACTING FEATURES: {split}/{device}")
        print("=" * 80)
        
        # Get list of images
        image_list = self.loader.get_image_list(split, device)
        print(f"Found {len(image_list)} images")
        
        features_list = []
        
        for i, filename in enumerate(image_list):
            if (i + 1) % 10 == 0:
                print(f"Processing: {i+1}/{len(image_list)}")
            
            features = self.extract_features(filename, split, device, mask_type)
            
            if features is not None:
                features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        print(f"\nSuccessfully extracted features from {len(features_list)} images")
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved features to: {output_csv}")
        
        return df
    
    def compare_with_csv(self, extracted_df: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
        """Compare extracted features with reference CSV.
        
        Args:
            extracted_df: DataFrame with extracted features
            reference_csv: Path to reference CSV file
            
        Returns:
            DataFrame with comparison statistics
        """
        print("\n" + "=" * 80)
        print("COMPARING WITH REFERENCE CSV")
        print("=" * 80)
        
        # Load reference CSV
        ref_df = pd.read_csv(reference_csv)
        print(f"Reference CSV has {len(ref_df)} entries")
        
        # Normalize image names for comparison
        # Remove the suffix "-IMG_XXXX-1.tif" pattern if present
        def normalize_name(name):
            # Convert to string and get base name
            name = str(name)
            # Remove extensions
            base = os.path.splitext(name)[0]
            # For reference CSV: "17521.tif-17521-1" -> "17521"
            # For extracted: "17521.jpg" -> "17521"
            if '.tif-' in base or '.jpg-' in base or '.png-' in base:
                # Pattern like "17521.tif-17521-1"
                base = base.split('-')[0].split('.')[0]
            return base.lower()
        
        extracted_df['normalized_name'] = extracted_df['Images'].apply(normalize_name)
        ref_df['normalized_name'] = ref_df['Images'].apply(normalize_name)
        
        # Merge dataframes
        merged = pd.merge(
            extracted_df, 
            ref_df, 
            on='normalized_name', 
            suffixes=('_extracted', '_reference'),
            how='inner'
        )
        
        print(f"Matched {len(merged)} images")
        
        if len(merged) == 0:
            print("Warning: No matching images found!")
            return pd.DataFrame()
        
        # Calculate differences for numeric columns
        numeric_cols = ['Disc Area', 'Cup Area', 'Rim Area', 
                       'Cup Height', 'Cup Width', 'Disc Height', 'Disc Width',
                       'ACDR', 'VCDR', 'HCDR']
        
        comparison_stats = []
        
        for col in numeric_cols:
            col_ext = f"{col}_extracted"
            col_ref = f"{col}_reference"
            
            if col_ext in merged.columns and col_ref in merged.columns:
                diff = merged[col_ext] - merged[col_ref]
                abs_diff = np.abs(diff)
                rel_diff = (abs_diff / merged[col_ref].replace(0, np.nan)) * 100
                
                stats = {
                    'Feature': col,
                    'Mean Absolute Difference': abs_diff.mean(),
                    'Max Absolute Difference': abs_diff.max(),
                    'Mean Relative Difference (%)': rel_diff.mean(),
                    'Correlation': merged[col_ext].corr(merged[col_ref])
                }
                comparison_stats.append(stats)
        
        comparison_df = pd.DataFrame(comparison_stats)
        
        print("\nComparison Statistics:")
        print(comparison_df.to_string(index=False))
        
        # Check glaucoma decision agreement
        if 'Glaucoma Decision_extracted' in merged.columns and 'Glaucoma Decision_reference' in merged.columns:
            agreement = (merged['Glaucoma Decision_extracted'] == merged['Glaucoma Decision_reference']).sum()
            total = len(merged)
            print(f"\nGlaucoma Decision Agreement: {agreement}/{total} ({100*agreement/total:.1f}%)")
        
        return comparison_df


def main():
    """Main function for feature extraction."""
    print("=" * 80)
    print("CHAKSHU FEATURE EXTRACTION")
    print("=" * 80)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract features for Remidio Train dataset
    output_csv = os.path.join(extractor.base_path, "extracted_features_remidio_train.csv")
    df = extractor.extract_all_features(
        split="Train",
        device="Remidio",
        mask_type="STAPLE",
        output_csv=output_csv
    )
    
    print("\n" + "=" * 80)
    print("SAMPLE OF EXTRACTED FEATURES")
    print("=" * 80)
    print(df.head(10).to_string())
    
    # Compare with reference CSV
    reference_csv = os.path.join(
        extractor.base_path,
        "Train/6.0_Glaucoma_Decision/Majority/Remidio.csv"
    )
    
    if os.path.exists(reference_csv):
        comparison_df = extractor.compare_with_csv(df, reference_csv)
    else:
        print(f"\nWarning: Reference CSV not found at {reference_csv}")
    
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nExtracted features saved to: {output_csv}")
    print(f"Total images processed: {len(df)}")
    print("\nYou can now use these features for:")
    print("  - Machine learning model training")
    print("  - Glaucoma risk analysis")
    print("  - Statistical analysis")
    print("  - Visualization and comparison")


if __name__ == "__main__":
    main()
