#!/usr/bin/env python3
"""
Quick Query Tool for Feature Extraction Results
Easily search and view patient data from the CSV.
"""

import pandas as pd
import sys
import os


def load_data():
    """Load the CSV data."""
    csv_path = "/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Glaucoma/raw_glaucoma_images_features.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None
    
    return pd.read_csv(csv_path)


def display_patient(df, patient_id):
    """Display detailed information for a specific patient."""
    # Convert to int if it's a string
    try:
        patient_id = int(patient_id)
    except ValueError:
        pass
    
    patient = df[df['Patient_ID'] == patient_id]
    
    if len(patient) == 0:
        print(f"No patient found with ID: {patient_id}")
        return
    
    patient = patient.iloc[0]
    
    print("=" * 80)
    print(f"PATIENT REPORT: {patient_id}")
    print("=" * 80)
    
    print(f"\nImage File: {patient['Image_Filename']}")
    print(f"Processing Status: {patient['Processing_Status']}")
    
    if patient['Processing_Status'] == 'SUCCESS':
        print("\n--- ANATOMICAL MEASUREMENTS ---")
        print(f"  Optic Disc Area:     {patient['Disc_Area']:>10.0f} pixels")
        print(f"  Optic Cup Area:      {patient['Cup_Area']:>10.0f} pixels")
        print(f"  Neuroretinal Rim:    {patient['Rim_Area']:>10.0f} pixels")
        print(f"  Disc Dimensions:     {patient['Disc_Width']:.0f} x {patient['Disc_Height']:.0f} pixels (W x H)")
        print(f"  Cup Dimensions:      {patient['Cup_Width']:.0f} x {patient['Cup_Height']:.0f} pixels (W x H)")
        
        print("\n--- CUP-TO-DISC RATIOS ---")
        print(f"  ACDR (Area):         {patient['ACDR']:>10.4f}  {'⚠️ HIGH' if patient['ACDR'] > 0.4 else '✓ Normal'}")
        print(f"  VCDR (Vertical):     {patient['VCDR']:>10.4f}  {'⚠️ HIGH' if patient['VCDR'] > 0.6 else '✓ Normal'}")
        print(f"  HCDR (Horizontal):   {patient['HCDR']:>10.4f}")
        
        print("\n--- CLINICAL ASSESSMENT ---")
        print(f"  Prediction:          {patient['Glaucoma_Prediction']}")
        print(f"  Risk Level:          {patient['Risk_Level']}")
        
        # Provide interpretation
        print("\n--- INTERPRETATION ---")
        if patient['Glaucoma_Prediction'] == 'GLAUCOMA POSITIVE':
            print("  ⚠️  High risk indicators detected")
            print("  📋 Recommendation: Immediate ophthalmologist consultation")
        elif patient['Glaucoma_Prediction'] == 'GLAUCOMA SUSPECT':
            print("  ⚠️  Moderate risk indicators detected")
            print("  📋 Recommendation: Regular monitoring and follow-up")
        else:
            print("  ✓  No significant glaucoma indicators")
            print("  📋 Recommendation: Routine annual eye examination")
    
    print("=" * 80)


def list_patients_by_prediction(df, prediction_type):
    """List all patients with a specific prediction."""
    filtered = df[df['Glaucoma_Prediction'] == prediction_type]
    
    print("=" * 80)
    print(f"PATIENTS WITH PREDICTION: {prediction_type}")
    print("=" * 80)
    print(f"Total: {len(filtered)} patients\n")
    
    if len(filtered) > 0:
        display_cols = ['Patient_ID', 'ACDR', 'VCDR', 'HCDR', 'Risk_Level']
        print(filtered[display_cols].to_string(index=False))
    
    print("=" * 80)


def list_high_risk_patients(df):
    """List patients with highest CDR values."""
    print("=" * 80)
    print("HIGH RISK PATIENTS (ACDR > 0.7 OR VCDR > 0.8)")
    print("=" * 80)
    
    high_risk = df[(df['ACDR'] > 0.7) | (df['VCDR'] > 0.8)]
    
    if len(high_risk) > 0:
        display_cols = ['Patient_ID', 'ACDR', 'VCDR', 'Glaucoma_Prediction', 'Risk_Level']
        print(high_risk.sort_values('ACDR', ascending=False)[display_cols].to_string(index=False))
        print(f"\nTotal High Risk Cases: {len(high_risk)}")
    else:
        print("No high risk cases found.")
    
    print("=" * 80)


def show_statistics(df):
    """Show summary statistics."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Patients: {len(df)}")
    
    print("\nPrediction Distribution:")
    for pred, count in df['Glaucoma_Prediction'].value_counts().items():
        print(f"  {pred:25s}: {count:3d} ({count/len(df)*100:5.1f}%)")
    
    print("\nCDR Statistics:")
    print(f"  ACDR: {df['ACDR'].mean():.4f} ± {df['ACDR'].std():.4f}")
    print(f"  VCDR: {df['VCDR'].mean():.4f} ± {df['VCDR'].std():.4f}")
    print(f"  HCDR: {df['HCDR'].mean():.4f} ± {df['HCDR'].std():.4f}")
    
    print("=" * 80)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("=" * 80)
        print("PATIENT DATA QUERY TOOL")
        print("=" * 80)
        print("\nUsage:")
        print("  python3 query_results.py <patient_id>         # Show specific patient")
        print("  python3 query_results.py --positive           # List glaucoma positive")
        print("  python3 query_results.py --suspect            # List glaucoma suspect")
        print("  python3 query_results.py --normal             # List normal cases")
        print("  python3 query_results.py --high-risk          # List high risk cases")
        print("  python3 query_results.py --stats              # Show statistics")
        print("\nExamples:")
        print("  python3 query_results.py 036")
        print("  python3 query_results.py 533")
        print("  python3 query_results.py --positive")
        print("  python3 query_results.py --high-risk")
        sys.exit(0)
    
    df = load_data()
    if df is None:
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "--positive":
        list_patients_by_prediction(df, "GLAUCOMA POSITIVE")
    elif arg == "--suspect":
        list_patients_by_prediction(df, "GLAUCOMA SUSPECT")
    elif arg == "--normal":
        list_patients_by_prediction(df, "NORMAL")
    elif arg == "--high-risk":
        list_high_risk_patients(df)
    elif arg == "--stats":
        show_statistics(df)
    else:
        # Assume it's a patient ID
        display_patient(df, arg)


if __name__ == "__main__":
    main()
