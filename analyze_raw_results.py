#!/usr/bin/env python3
"""
Analysis Script for Raw Image Feature Extraction Results
Provides detailed statistics and insights from the extracted features.
"""

import pandas as pd
import numpy as np
import os


def analyze_results(csv_path: str):
    """Analyze the feature extraction results."""
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("DETAILED ANALYSIS OF RAW GLAUCOMA IMAGE FEATURES")
    print("=" * 80)
    
    # Basic statistics
    print(f"\nTotal Images Processed: {len(df)}")
    print(f"Successfully Processed: {len(df[df['Processing_Status'] == 'SUCCESS'])}")
    print(f"Failed: {len(df[df['Processing_Status'] != 'SUCCESS'])}")
    
    # Filter successful processing
    df_success = df[df['Processing_Status'] == 'SUCCESS'].copy()
    
    if len(df_success) == 0:
        print("\nNo successful extractions to analyze.")
        return
    
    # Glaucoma Prediction Summary
    print("\n" + "=" * 80)
    print("GLAUCOMA PREDICTION BREAKDOWN")
    print("=" * 80)
    
    prediction_summary = df_success['Glaucoma_Prediction'].value_counts().sort_index()
    for prediction, count in prediction_summary.items():
        percentage = (count / len(df_success)) * 100
        print(f"  {prediction:25s}: {count:4d} images ({percentage:5.1f}%)")
    
    # Risk Level Summary
    print("\n" + "=" * 80)
    print("RISK LEVEL DISTRIBUTION")
    print("=" * 80)
    
    risk_summary = df_success['Risk_Level'].value_counts().sort_index()
    for risk, count in risk_summary.items():
        percentage = (count / len(df_success)) * 100
        print(f"  {risk:15s}: {count:4d} images ({percentage:5.1f}%)")
    
    # CDR Statistics
    print("\n" + "=" * 80)
    print("CUP-TO-DISC RATIO (CDR) STATISTICS")
    print("=" * 80)
    
    cdr_metrics = ['ACDR', 'VCDR', 'HCDR']
    
    print(f"\n{'Metric':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 60)
    
    for metric in cdr_metrics:
        if metric in df_success.columns:
            values = df_success[metric]
            print(f"{metric:<10} {values.mean():>9.4f} {values.std():>9.4f} "
                  f"{values.min():>9.4f} {values.max():>9.4f} {values.median():>9.4f}")
    
    # Area Statistics
    print("\n" + "=" * 80)
    print("OPTIC DISC AND CUP AREA STATISTICS (in pixels)")
    print("=" * 80)
    
    area_metrics = ['Disc_Area', 'Cup_Area', 'Rim_Area']
    
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 63)
    
    for metric in area_metrics:
        if metric in df_success.columns:
            values = df_success[metric]
            print(f"{metric:<15} {values.mean():>11.1f} {values.std():>11.1f} "
                  f"{values.min():>11.1f} {values.max():>11.1f}")
    
    # Dimension Statistics
    print("\n" + "=" * 80)
    print("OPTIC DISC AND CUP DIMENSION STATISTICS (in pixels)")
    print("=" * 80)
    
    dim_metrics = ['Disc_Height', 'Disc_Width', 'Cup_Height', 'Cup_Width']
    
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 63)
    
    for metric in dim_metrics:
        if metric in df_success.columns:
            values = df_success[metric]
            print(f"{metric:<15} {values.mean():>11.1f} {values.std():>11.1f} "
                  f"{values.min():>11.1f} {values.max():>11.1f}")
    
    # Clinical Insights
    print("\n" + "=" * 80)
    print("CLINICAL INSIGHTS")
    print("=" * 80)
    
    # High risk cases based on VCDR > 0.7 or ACDR > 0.5
    high_vcdr = df_success[df_success['VCDR'] > 0.7]
    high_acdr = df_success[df_success['ACDR'] > 0.5]
    
    print(f"\nCases with VCDR > 0.7 (High Risk): {len(high_vcdr)} ({len(high_vcdr)/len(df_success)*100:.1f}%)")
    print(f"Cases with ACDR > 0.5 (High Risk): {len(high_acdr)} ({len(high_acdr)/len(df_success)*100:.1f}%)")
    
    # Very high risk cases
    very_high_vcdr = df_success[df_success['VCDR'] > 0.8]
    very_high_acdr = df_success[df_success['ACDR'] > 0.7]
    
    print(f"\nCases with VCDR > 0.8 (Very High Risk): {len(very_high_vcdr)} ({len(very_high_vcdr)/len(df_success)*100:.1f}%)")
    print(f"Cases with ACDR > 0.7 (Very High Risk): {len(very_high_acdr)} ({len(very_high_acdr)/len(df_success)*100:.1f}%)")
    
    # Patients with highest risk
    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST RISK CASES (by ACDR)")
    print("=" * 80)
    
    top_acdr = df_success.nlargest(10, 'ACDR')[['Patient_ID', 'ACDR', 'VCDR', 'HCDR', 'Glaucoma_Prediction', 'Risk_Level']]
    print(top_acdr.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST RISK CASES (by VCDR)")
    print("=" * 80)
    
    top_vcdr = df_success.nlargest(10, 'VCDR')[['Patient_ID', 'ACDR', 'VCDR', 'HCDR', 'Glaucoma_Prediction', 'Risk_Level']]
    print(top_vcdr.to_string(index=False))
    
    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION BETWEEN CDR METRICS")
    print("=" * 80)
    
    cdr_corr = df_success[['ACDR', 'VCDR', 'HCDR']].corr()
    print(cdr_corr.to_string())
    
    # Summary of findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    glaucoma_positive = len(df_success[df_success['Glaucoma_Prediction'] == 'GLAUCOMA POSITIVE'])
    glaucoma_suspect = len(df_success[df_success['Glaucoma_Prediction'] == 'GLAUCOMA SUSPECT'])
    normal = len(df_success[df_success['Glaucoma_Prediction'] == 'NORMAL'])
    
    print(f"""
1. Dataset Overview:
   - Total images analyzed: {len(df_success)}
   - Folder name indicates these are known glaucoma positive cases
   
2. Automated Prediction Results:
   - Glaucoma Positive: {glaucoma_positive} ({glaucoma_positive/len(df_success)*100:.1f}%)
   - Glaucoma Suspect: {glaucoma_suspect} ({glaucoma_suspect/len(df_success)*100:.1f}%)
   - Normal: {normal} ({normal/len(df_success)*100:.1f}%)
   
3. CDR Analysis:
   - Average ACDR: {df_success['ACDR'].mean():.4f} (threshold: 0.4 for glaucoma)
   - Average VCDR: {df_success['VCDR'].mean():.4f} (threshold: 0.6 for glaucoma)
   - Average HCDR: {df_success['HCDR'].mean():.4f}
   
4. Model Performance Note:
   - The automated predictions are based on CDR thresholds
   - {glaucoma_positive + glaucoma_suspect} of {len(df_success)} ({(glaucoma_positive + glaucoma_suspect)/len(df_success)*100:.1f}%) 
     cases flagged as glaucoma positive or suspect
   - This indicates the model is identifying high-risk features in most images
   
5. Recommendations:
   - Images predicted as NORMAL ({normal} cases) may need manual review
   - High ACDR values (>{df_success['ACDR'].quantile(0.75):.3f}) indicate severe cases
   - Consider correlation with clinical diagnoses for validation
    """)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {csv_path}")
    print("You can now use this data for:")
    print("  - Clinical validation studies")
    print("  - Machine learning model training")
    print("  - Statistical analysis and research")
    print("  - Patient risk stratification")


def main():
    """Main function."""
    csv_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Glaucoma/raw_glaucoma_images_features.csv"
    analyze_results(csv_path)


if __name__ == "__main__":
    main()
