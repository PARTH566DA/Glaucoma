# Feature Extraction for Chakshu Glaucoma Dataset

This directory contains scripts for extracting glaucoma-related features from fundus images in the Chakshu dataset.

## Overview

The feature extraction pipeline computes the following features from fundus images:

### Geometric Features
- **Disc Area**: Total area of the optic disc (in pixels)
- **Cup Area**: Total area of the optic cup (in pixels)
- **Rim Area**: Area of the neuroretinal rim (Disc Area - Cup Area)
- **Cup Height**: Vertical extent of the cup (in pixels)
- **Cup Width**: Horizontal extent of the cup (in pixels)
- **Disc Height**: Vertical extent of the disc (in pixels)
- **Disc Width**: Horizontal extent of the disc (in pixels)

### Clinical Metrics
- **ACDR (Area Cup-to-Disc Ratio)**: Cup Area / Disc Area
- **VCDR (Vertical Cup-to-Disc Ratio)**: Cup Height / Disc Height
- **HCDR (Horizontal Cup-to-Disc Ratio)**: Cup Width / Disc Width
- **Glaucoma Decision**: Classification based on CDR thresholds
  - NORMAL: ACDR ≤ 0.4 and VCDR ≤ 0.6
  - GLAUCOMA SUSPECT: ACDR > 0.4 or VCDR > 0.6

## Scripts

### 1. `extract_features.py`
Main feature extraction script that:
- Loads fundus images and their corresponding disc/cup masks
- Computes all geometric and clinical features
- Saves results to CSV format
- Compares with reference CSV (if available)

**Usage:**
```bash
python3 extract_features.py
```

**Output:**
- `extracted_features_remidio_train.csv`: CSV file with all extracted features

### 2. `compare_features.py`
Comparison script that validates extracted features against reference data:
- Loads both extracted and reference CSVs
- Computes statistical comparisons (mean/max differences, correlations)
- Reports glaucoma decision agreement percentage

**Usage:**
```bash
python3 compare_features.py
```

### 3. `visualize_features.py`
Visualization script for feature analysis (to be created)

## Results

### Validation Results (Train/Remidio dataset)

The extracted features were compared with the reference CSV:

| Feature | Mean Abs Difference | Max Abs Difference | Correlation |
|---------|-------------------|-------------------|-------------|
| Disc Area | 5,609 px | 25,611 px | 0.996 |
| Cup Area | 6,164 px | 24,412 px | 0.987 |
| Rim Area | 2,700 px | 16,991 px | 0.973 |
| Cup Height | 16.3 px | 80 px | 0.973 |
| Cup Width | 18.9 px | 110 px | 0.960 |
| Disc Height | 10.1 px | 47 px | 0.989 |
| Disc Width | 8.5 px | 48 px | 0.991 |
| ACDR | 0.03 | 0.15 | 0.961 |
| VCDR | 0.03 | 0.14 | 0.933 |
| HCDR | 0.04 | 0.22 | 0.900 |

**Glaucoma Decision Agreement**: 734/810 (90.6%)

### Key Observations

1. **High Correlation**: All features show strong correlation (>0.90) with reference data
2. **Minor Variations**: Small differences likely due to:
   - Different mask processing methods (STAPLE vs Majority)
   - Rounding differences
   - Boundary pixel handling
3. **Good Clinical Agreement**: 90.6% agreement on glaucoma classification

## File Structure

```
20123135/
├── extract_features.py          # Main feature extraction script
├── compare_features.py           # Feature validation script
├── data_loader.py                # Data loading utilities
├── visualize_opencv.py           # Visualization utilities
├── extracted_features_remidio_train.csv  # Extracted features (810 images)
└── Train/
    ├── 1.0_Original_Fundus_Images/
    │   └── Remidio/              # Original fundus images
    ├── 5.0_OD_OC_Mean_Median_Majority_STAPLE/
    │   └── Remidio/
    │       ├── Disc/STAPLE/      # Disc masks
    │       └── Cup/STAPLE/       # Cup masks
    └── 6.0_Glaucoma_Decision/
        └── Majority/
            └── Remidio.csv       # Reference features
```

## Dependencies

```bash
pip3 install opencv-python numpy pandas
```

## Usage Examples

### Extract Features for All Images
```python
from extract_features import FeatureExtractor

extractor = FeatureExtractor()
df = extractor.extract_all_features(
    split="Train",
    device="Remidio",
    mask_type="STAPLE",
    output_csv="features.csv"
)
```

### Extract Features for Single Image
```python
from extract_features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features(
    filename="IMG_2431.JPG",
    split="Train",
    device="Remidio"
)

print(f"ACDR: {features['ACDR']:.3f}")
print(f"Decision: {features['Glaucoma Decision']}")
```

### Load and Analyze Features
```python
import pandas as pd

df = pd.read_csv("extracted_features_remidio_train.csv")

# Get statistics
print(df[['ACDR', 'VCDR', 'HCDR']].describe())

# Filter glaucoma suspects
suspects = df[df['Glaucoma Decision'] == 'GLAUCOMA  SUSUPECT']
print(f"Found {len(suspects)} glaucoma suspects")

# Analyze CDR distributions
print(f"Mean ACDR: {df['ACDR'].mean():.3f}")
print(f"Mean VCDR: {df['VCDR'].mean():.3f}")
```

## Clinical Interpretation

### CDR Thresholds
- **ACDR < 0.3**: Healthy (low risk)
- **0.3 ≤ ACDR ≤ 0.4**: Borderline (monitor)
- **ACDR > 0.4**: Glaucoma suspect (high risk)

- **VCDR < 0.5**: Healthy
- **0.5 ≤ VCDR ≤ 0.6**: Borderline
- **VCDR > 0.6**: Glaucoma suspect

### Feature Importance
1. **VCDR** (Vertical CDR): Most clinically significant
2. **ACDR** (Area CDR): Comprehensive measure
3. **HCDR** (Horizontal CDR): Additional validation
4. **Disc/Cup Areas**: Absolute measurements
5. **Rim Area**: Neuroretinal rim health indicator

## Next Steps

1. **Machine Learning**: Use extracted features for glaucoma classification
2. **Statistical Analysis**: Analyze CDR distributions across devices
3. **Visualization**: Create plots showing feature relationships
4. **Cross-validation**: Compare features across different mask types
5. **Test Set**: Extract features from Test split for model evaluation

## References

- Chakshu Dataset Documentation
- Clinical glaucoma assessment guidelines
- Cup-to-disc ratio measurement standards

## Contact

For questions or issues, refer to the main dataset README or documentation.
