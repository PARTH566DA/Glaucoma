"""
Feature Visualization and Analysis Script
Visualizes and analyzes extracted features from the Chakshu dataset
"""

import pandas as pd
import numpy as np
import cv2
import os
from data_loader import ChakshuDataLoader

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def analyze_features(csv_path):
    """Analyze and visualize features from CSV."""
    
    print_section("LOADING FEATURES")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images with features")
    
    # Basic statistics
    print_section("BASIC STATISTICS")
    numeric_cols = ['Disc Area', 'Cup Area', 'Rim Area', 
                   'Cup Height', 'Cup Width', 'Disc Height', 'Disc Width',
                   'ACDR', 'VCDR', 'HCDR']
    
    stats = df[numeric_cols].describe()
    print(stats.to_string())
    
    # Glaucoma classification distribution
    print_section("GLAUCOMA CLASSIFICATION DISTRIBUTION")
    decision_counts = df['Glaucoma Decision'].value_counts()
    print(decision_counts)
    print(f"\nPercentage:")
    for decision, count in decision_counts.items():
        print(f"  {decision}: {100*count/len(df):.1f}%")
    
    # CDR analysis
    print_section("CUP-TO-DISC RATIO (CDR) ANALYSIS")
    
    print("\nACDR (Area CDR) Distribution:")
    print(f"  Min:  {df['ACDR'].min():.3f}")
    print(f"  25%:  {df['ACDR'].quantile(0.25):.3f}")
    print(f"  Mean: {df['ACDR'].mean():.3f}")
    print(f"  75%:  {df['ACDR'].quantile(0.75):.3f}")
    print(f"  Max:  {df['ACDR'].max():.3f}")
    
    print("\nVCDR (Vertical CDR) Distribution:")
    print(f"  Min:  {df['VCDR'].min():.3f}")
    print(f"  25%:  {df['VCDR'].quantile(0.25):.3f}")
    print(f"  Mean: {df['VCDR'].mean():.3f}")
    print(f"  75%:  {df['VCDR'].quantile(0.75):.3f}")
    print(f"  Max:  {df['VCDR'].max():.3f}")
    
    print("\nHCDR (Horizontal CDR) Distribution:")
    print(f"  Min:  {df['HCDR'].min():.3f}")
    print(f"  25%:  {df['HCDR'].quantile(0.25):.3f}")
    print(f"  Mean: {df['HCDR'].mean():.3f}")
    print(f"  75%:  {df['HCDR'].quantile(0.75):.3f}")
    print(f"  Max:  {df['HCDR'].max():.3f}")
    
    # Risk stratification
    print_section("RISK STRATIFICATION")
    
    # Based on ACDR
    low_risk = len(df[df['ACDR'] < 0.3])
    borderline = len(df[(df['ACDR'] >= 0.3) & (df['ACDR'] <= 0.4)])
    high_risk = len(df[df['ACDR'] > 0.4])
    
    print(f"Based on ACDR thresholds:")
    print(f"  Low Risk (ACDR < 0.3):      {low_risk:3d} ({100*low_risk/len(df):5.1f}%)")
    print(f"  Borderline (0.3-0.4):       {borderline:3d} ({100*borderline/len(df):5.1f}%)")
    print(f"  High Risk (ACDR > 0.4):     {high_risk:3d} ({100*high_risk/len(df):5.1f}%)")
    
    # Based on VCDR
    low_risk_v = len(df[df['VCDR'] < 0.5])
    borderline_v = len(df[(df['VCDR'] >= 0.5) & (df['VCDR'] <= 0.6)])
    high_risk_v = len(df[df['VCDR'] > 0.6])
    
    print(f"\nBased on VCDR thresholds:")
    print(f"  Low Risk (VCDR < 0.5):      {low_risk_v:3d} ({100*low_risk_v/len(df):5.1f}%)")
    print(f"  Borderline (0.5-0.6):       {borderline_v:3d} ({100*borderline_v/len(df):5.1f}%)")
    print(f"  High Risk (VCDR > 0.6):     {high_risk_v:3d} ({100*high_risk_v/len(df):5.1f}%)")
    
    # Find extreme cases
    print_section("EXTREME CASES")
    
    print("\nLowest ACDR (5 cases):")
    lowest_acdr = df.nsmallest(5, 'ACDR')[['Images', 'ACDR', 'VCDR', 'HCDR', 'Glaucoma Decision']]
    print(lowest_acdr.to_string(index=False))
    
    print("\nHighest ACDR (5 cases):")
    highest_acdr = df.nlargest(5, 'ACDR')[['Images', 'ACDR', 'VCDR', 'HCDR', 'Glaucoma Decision']]
    print(highest_acdr.to_string(index=False))
    
    # Correlation analysis
    print_section("CORRELATION ANALYSIS")
    
    cdr_cols = ['ACDR', 'VCDR', 'HCDR']
    correlation_matrix = df[cdr_cols].corr()
    print("\nCorrelation between CDR metrics:")
    print(correlation_matrix.to_string())
    
    # Area correlations
    print("\nCorrelation between areas:")
    area_cols = ['Disc Area', 'Cup Area', 'Rim Area']
    area_correlation = df[area_cols].corr()
    print(area_correlation.to_string())
    
    return df


def visualize_samples(df, num_samples=5):
    """Visualize sample images with their features."""
    
    print_section(f"VISUALIZING {num_samples} SAMPLE IMAGES")
    
    loader = ChakshuDataLoader()
    output_dir = "feature_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Select diverse samples
    samples_to_viz = []
    
    # Get one with low CDR
    low_cdr = df.nsmallest(1, 'ACDR').iloc[0]
    samples_to_viz.append(("Low_CDR", low_cdr))
    
    # Get one with medium CDR
    median_idx = len(df) // 2
    medium_cdr = df.sort_values('ACDR').iloc[median_idx]
    samples_to_viz.append(("Medium_CDR", medium_cdr))
    
    # Get one with high CDR
    high_cdr = df.nlargest(1, 'ACDR').iloc[0]
    samples_to_viz.append(("High_CDR", high_cdr))
    
    # Get two glaucoma suspects if available
    suspects = df[df['Glaucoma Decision'] == 'GLAUCOMA  SUSUPECT']
    if len(suspects) >= 2:
        for i, (_, row) in enumerate(suspects.head(2).iterrows()):
            samples_to_viz.append((f"Glaucoma_Suspect_{i+1}", row))
    
    for name, row in samples_to_viz[:num_samples]:
        filename = row['Images']
        
        # Load image and masks
        sample = loader.load_sample(filename, split="Train", device="Remidio")
        
        if sample['image'] is None or sample['disc_mask'] is None or sample['cup_mask'] is None:
            print(f"  Skipping {filename} - data not available")
            continue
        
        # Create visualization
        image = sample['image']
        disc_mask = sample['disc_mask']
        cup_mask = sample['cup_mask']
        
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        h, w = image_bgr.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w, new_h = int(w * scale), int(h * scale)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h))
            disc_mask = cv2.resize(disc_mask, (new_w, new_h))
            cup_mask = cv2.resize(cup_mask, (new_w, new_h))
        
        # Create overlay
        overlay = image_bgr.copy()
        
        # Add disc contour (green)
        disc_binary = (disc_mask > 128).astype(np.uint8)
        disc_contours, _ = cv2.findContours(disc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, disc_contours, -1, (0, 255, 0), 3)
        
        # Add cup contour (red)
        cup_binary = (cup_mask > 128).astype(np.uint8)
        cup_contours, _ = cv2.findContours(cup_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cup_contours, -1, (0, 0, 255), 3)
        
        # Add text with features
        text_lines = [
            f"File: {filename}",
            f"ACDR: {row['ACDR']:.3f}",
            f"VCDR: {row['VCDR']:.3f}",
            f"HCDR: {row['HCDR']:.3f}",
            f"Decision: {row['Glaucoma Decision']}"
        ]
        
        y = 30
        for line in text_lines:
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30
        
        # Save image
        output_path = os.path.join(output_dir, f"{name}_{row['ACDR']:.3f}.jpg")
        cv2.imwrite(output_path, overlay)
        print(f"  Saved: {output_path}")
    
    print(f"\nVisualization complete. Images saved to: {output_dir}/")


def main():
    """Main function for feature analysis."""
    
    print("=" * 80)
    print("CHAKSHU FEATURE ANALYSIS")
    print("=" * 80)
    
    # Analyze features
    csv_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/extracted_features_remidio_train.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Features CSV not found at {csv_path}")
        print("Please run extract_features.py first!")
        return
    
    df = analyze_features(csv_path)
    
    # Visualize samples
    visualize_samples(df, num_samples=5)
    
    print_section("ANALYSIS COMPLETE")
    print("\nYou can now:")
    print("  1. Review the statistics above")
    print("  2. Check visualizations in feature_visualizations/")
    print("  3. Use the CSV for machine learning")
    print("  4. Perform additional custom analysis")


if __name__ == "__main__":
    main()
