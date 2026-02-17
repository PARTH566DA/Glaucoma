#!/usr/bin/env python3
"""
Quick Feature Extraction - Single Command
Extract features from a specific image or all images
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_features import FeatureExtractor

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 quick_extract.py <image_filename>  # Extract single image")
        print("  python3 quick_extract.py --all              # Extract all images")
        print("  python3 quick_extract.py --help             # Show this help")
        print("\nExamples:")
        print("  python3 quick_extract.py IMG_2431.JPG")
        print("  python3 quick_extract.py 17521.jpg")
        print("  python3 quick_extract.py --all")
        sys.exit(0)
    
    arg = sys.argv[1]
    
    extractor = FeatureExtractor()
    
    if arg == "--all":
        print("Extracting features for all images...")
        df = extractor.extract_all_features(
            split="Train",
            device="Remidio",
            mask_type="STAPLE",
            output_csv="extracted_features_remidio_train.csv"
        )
        print(f"\n✓ Successfully extracted {len(df)} images")
        print(f"✓ Saved to: extracted_features_remidio_train.csv")
        
    else:
        # Single image
        filename = arg
        print(f"Extracting features for: {filename}")
        
        features = extractor.extract_features(
            filename=filename,
            split="Train",
            device="Remidio",
            mask_type="STAPLE"
        )
        
        if features:
            print("\n" + "=" * 60)
            print("EXTRACTED FEATURES")
            print("=" * 60)
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.3f}")
                else:
                    print(f"  {key:20s}: {value}")
            print("=" * 60)
        else:
            print("✗ Failed to extract features (image or masks not found)")

if __name__ == "__main__":
    main()
