# Raw Glaucoma Images Feature Extraction

## Overview
This document describes the feature extraction process performed on the raw glaucoma-positive fundus images.

## Files Created

### 1. `extract_raw_images.py`
Main feature extraction script that:
- Loads raw fundus images from the `raw images(glaucoma+ve))` folder
- Automatically detects optic disc and cup using OpenCV-based segmentation
- Extracts clinical features (areas, dimensions, CDR ratios)
- Predicts glaucoma status based on CDR thresholds
- Saves results to CSV file

### 2. `analyze_raw_results.py`
Analysis script that provides detailed statistics and insights from the extracted features.

### 3. `raw_glaucoma_images_features.csv`
Output CSV file containing extracted features for all 134 images.

## Dataset Information

**Source Folder:** `raw images(glaucoma+ve))`
- Total Images: 134
- Image Format: JPG
- Known Status: Glaucoma Positive Cases

## Extracted Features

For each image, the following features are extracted:

### Identification
- `Image_Filename`: Original filename
- `Patient_ID`: Patient identifier (extracted from filename)

### Anatomical Measurements
- `Disc_Area`: Optic disc area (pixels)
- `Cup_Area`: Optic cup area (pixels)
- `Rim_Area`: Neuroretinal rim area (pixels)
- `Disc_Height`: Optic disc vertical dimension (pixels)
- `Disc_Width`: Optic disc horizontal dimension (pixels)
- `Cup_Height`: Optic cup vertical dimension (pixels)
- `Cup_Width`: Optic cup horizontal dimension (pixels)

### Clinical Ratios
- `ACDR`: Area Cup-to-Disc Ratio
- `VCDR`: Vertical Cup-to-Disc Ratio
- `HCDR`: Horizontal Cup-to-Disc Ratio

### Prediction Results
- `Glaucoma_Prediction`: Automated prediction (GLAUCOMA POSITIVE / GLAUCOMA SUSPECT / NORMAL)
- `Risk_Level`: Risk classification (HIGH / MEDIUM / LOW)
- `Processing_Status`: Processing status (SUCCESS / FAILED)

## Prediction Methodology

The automated prediction uses clinically established CDR thresholds:

### High Risk (GLAUCOMA POSITIVE)
- ACDR > 0.4 OR
- VCDR > 0.6

### Medium Risk (GLAUCOMA SUSPECT)
- ACDR > 0.3 OR
- VCDR > 0.5

### Low Risk (NORMAL)
- Below the above thresholds

## Results Summary

### Processing Statistics
- **Successfully Processed:** 134 images (100%)
- **Failed:** 0 images (0%)

### Prediction Breakdown
- **GLAUCOMA POSITIVE:** 81 images (60.4%)
- **GLAUCOMA SUSPECT:** 29 images (21.6%)
- **NORMAL:** 24 images (17.9%)

### CDR Statistics
- **ACDR:** Mean = 0.5699, Std = 0.5012, Range = [0.001, 3.688]
- **VCDR:** Mean = 0.6216, Std = 0.2185, Range = [0.006, 1.000]
- **HCDR:** Mean = 0.6323, Std = 0.2231, Range = [0.013, 1.000]

### Clinical Insights
- 44 cases (32.8%) have VCDR > 0.7 (High Risk)
- 62 cases (46.3%) have ACDR > 0.5 (High Risk)
- 28 cases (20.9%) have VCDR > 0.8 (Very High Risk)
- 43 cases (32.1%) have ACDR > 0.7 (Very High Risk)

## Usage

### Extract Features from Raw Images
```bash
cd /Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Glaucoma
python3 extract_raw_images.py
```

### Analyze Extracted Features
```bash
python3 analyze_raw_results.py
```

### View CSV Data
```bash
# View first 20 rows
head -20 raw_glaucoma_images_features.csv

# Open in Excel/LibreOffice/Numbers
open raw_glaucoma_images_features.csv
```

## Top Risk Cases

### Highest ACDR Values (Most Severe)
1. Patient 533: ACDR = 3.688
2. Patient 72: ACDR = 3.052
3. Patient 385: ACDR = 2.230
4. Patient 382: ACDR = 1.523
5. Patient 515: ACDR = 1.517

### Highest VCDR Values (Most Severe)
1. Patient 138: VCDR = 1.000
2. Patient 225: VCDR = 1.000
3. Patient 337: VCDR = 1.000
4. Patient 390: VCDR = 1.000
5. Patient 508: VCDR = 1.000

## Important Notes

1. **Ground Truth**: These images are from the "glaucoma+ve" folder, indicating they are clinically confirmed glaucoma positive cases.

2. **Automated Detection**: The optic disc and cup detection is performed automatically using OpenCV-based image processing. Results may vary from manual segmentation.

3. **Clinical Validation**: The automated predictions should be validated against clinical diagnoses for accuracy assessment.

4. **Discrepancy Analysis**: 24 images (17.9%) were predicted as NORMAL despite being in the glaucoma-positive folder. These cases warrant manual review.

5. **Data Quality**: Some extreme ACDR values (>1.0) may indicate segmentation challenges and should be manually verified.

## Applications

The extracted features can be used for:
- **Machine Learning**: Training glaucoma detection models
- **Clinical Research**: Statistical analysis of glaucoma characteristics
- **Risk Stratification**: Identifying high-risk patients
- **Validation Studies**: Comparing automated vs. manual assessment
- **Longitudinal Analysis**: Tracking disease progression

## Technical Details

### Segmentation Method
- **Preprocessing**: CLAHE enhancement for contrast improvement
- **Disc Detection**: Brightness-based thresholding (optic disc is typically brightest)
- **Cup Detection**: Adaptive thresholding within disc region
- **Morphological Operations**: Opening and closing for noise reduction

### Clinical Thresholds
Based on established ophthalmology literature:
- Normal VCDR: < 0.5
- Glaucoma Suspect VCDR: 0.5 - 0.6
- Glaucoma Positive VCDR: > 0.6
- Severe Glaucoma VCDR: > 0.8

## References

- Cup-to-Disc Ratio thresholds based on clinical ophthalmology standards
- Chakshu Database: Indian fundus image database for glaucoma research
- OpenCV library for image processing and segmentation

## Author & Date

Created: February 17, 2026
Dataset: Chakshu Glaucoma Database
Image Source: Raw images (glaucoma+ve) folder (134 images)

---

For questions or issues, please review the Python scripts or contact the dataset maintainer.
