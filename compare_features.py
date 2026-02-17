"""
Compare extracted features with reference CSV
"""

import pandas as pd
import numpy as np
import os

def normalize_name(name):
    """Normalize image name for comparison."""
    name = str(name)
    # Remove extensions
    base = os.path.splitext(name)[0]
    # For reference CSV: "17521.tif-17521-1" -> "17521"
    # For extracted: "17521.jpg" -> "17521"
    if '.tif-' in base or '.jpg-' in base or '.png-' in base:
        # Pattern like "17521.tif-17521-1"
        base = base.split('-')[0].split('.')[0]
    return base.lower()

# Load datasets
base_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/"
extracted_csv = os.path.join(base_path, "extracted_features_remidio_train.csv")
reference_csv = os.path.join(base_path, "Train/6.0_Glaucoma_Decision/Majority/Remidio.csv")

print("Loading CSVs...")
extracted_df = pd.read_csv(extracted_csv)
ref_df = pd.read_csv(reference_csv)

print(f"Extracted: {len(extracted_df)} entries")
print(f"Reference: {len(ref_df)} entries")

# Normalize names
extracted_df['normalized_name'] = extracted_df['Images'].apply(normalize_name)
ref_df['normalized_name'] = ref_df['Images'].apply(normalize_name)

print("\nSample normalized names (extracted):")
print(extracted_df[['Images', 'normalized_name']].head())
print("\nSample normalized names (reference):")
print(ref_df[['Images', 'normalized_name']].head())

# Merge dataframes
merged = pd.merge(
    extracted_df, 
    ref_df, 
    on='normalized_name', 
    suffixes=('_extracted', '_reference'),
    how='inner'
)

print(f"\nMatched {len(merged)} images")

if len(merged) > 0:
    # Calculate differences for numeric columns
    numeric_cols = ['Disc Area', 'Cup Area', 'Rim Area', 
                   'Cup Height', 'Cup Width', 'Disc Height', 'Disc Width',
                   'ACDR', 'VCDR', 'HCDR']
    
    comparison_stats = []
    
    print("\n" + "=" * 100)
    print("FEATURE COMPARISON STATISTICS")
    print("=" * 100)
    
    for col in numeric_cols:
        col_ext = f"{col}_extracted"
        col_ref = f"{col}_reference"
        
        if col_ext in merged.columns and col_ref in merged.columns:
            diff = merged[col_ext] - merged[col_ref]
            abs_diff = np.abs(diff)
            rel_diff = (abs_diff / merged[col_ref].replace(0, np.nan)) * 100
            
            stats = {
                'Feature': col,
                'Mean Abs Diff': f"{abs_diff.mean():.2f}",
                'Max Abs Diff': f"{abs_diff.max():.2f}",
                'Mean Rel Diff (%)': f"{rel_diff.mean():.2f}",
                'Correlation': f"{merged[col_ext].corr(merged[col_ref]):.4f}"
            }
            comparison_stats.append(stats)
    
    comparison_df = pd.DataFrame(comparison_stats)
    print(comparison_df.to_string(index=False))
    
    # Check glaucoma decision agreement
    if 'Glaucoma Decision_extracted' in merged.columns and 'Glaucoma Decision_reference' in merged.columns:
        agreement = (merged['Glaucoma Decision_extracted'] == merged['Glaucoma Decision_reference']).sum()
        total = len(merged)
        print(f"\n{'=' * 100}")
        print(f"Glaucoma Decision Agreement: {agreement}/{total} ({100*agreement/total:.1f}%)")
        print("=" * 100)
    
    # Show some sample comparisons
    print("\n" + "=" * 100)
    print("SAMPLE COMPARISONS (First 5 matches)")
    print("=" * 100)
    
    sample_cols = ['Images_extracted', 'Disc Area_extracted', 'Disc Area_reference',
                   'ACDR_extracted', 'ACDR_reference', 
                   'Glaucoma Decision_extracted', 'Glaucoma Decision_reference']
    
    available_cols = [col for col in sample_cols if col in merged.columns]
    print(merged[available_cols].head().to_string())
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
else:
    print("\nNo matches found! Check the normalization logic.")
