import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    'Disc Area', 'Cup Area', 'Rim Area', 'Cup Height',
    'Cup Width', 'Disc Height', 'Disc Width', 'ACDR', 'VCDR', 'HCDR'
]


def normalize_name(name):
    """Normalize image naming across extracted and reference CSVs."""
    name = str(name)
    base = name.split('-')[0]
    base = base.split('.')[0]
    return base.lower()


def to_binary_label(series: pd.Series) -> np.ndarray:
    """Convert label text to binary target: glaucoma/suspect=1, normal=0."""
    return np.where(series.astype(str).str.contains('GLAUCOMA', case=False, na=False), 1, 0)


def merge_with_ground_truth(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Merge extracted feature rows with ground-truth labels using normalized image names."""
    if 'Images' not in features_df.columns or 'Images' not in labels_df.columns:
        raise ValueError("Both feature and label CSVs must contain an 'Images' column.")
    if 'Glaucoma Decision' not in labels_df.columns:
        raise ValueError("Ground-truth labels CSV must contain 'Glaucoma Decision' column.")

    features_df = features_df.copy()
    labels_df = labels_df.copy()
    features_df['normalized_name'] = features_df['Images'].apply(normalize_name)
    labels_df['normalized_name'] = labels_df['Images'].apply(normalize_name)
    labels_df = labels_df.rename(columns={'Glaucoma Decision': 'True_Label'})

    merged = pd.merge(
        features_df,
        labels_df[['normalized_name', 'True_Label']],
        on='normalized_name',
        how='inner'
    )

    if len(merged) == 0:
        raise ValueError("No matching image IDs between extracted features CSV and ground-truth labels CSV.")

    # Keep prepare_xy() interface by creating a temporary target column name it already supports.
    merged['Glaucoma_Prediction'] = merged['True_Label']
    return merged


def prepare_xy(df: pd.DataFrame, require_label: bool = True):
    """Prepare feature matrix and optional target from a dataframe."""
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    if not available_features:
        raise ValueError("No expected feature columns found in dataframe.")

    X = df[available_features].values

    y = None
    target_col = None
    # Prefer externally merged ground-truth labels when available.
    if 'True_Label' in df.columns:
        target_col = 'True_Label'
    elif 'Glaucoma_Prediction' in df.columns:
        target_col = 'Glaucoma_Prediction'
    elif 'Glaucoma Decision' in df.columns:
        target_col = 'Glaucoma Decision'

    if target_col is not None:
        y = to_binary_label(df[target_col])
    elif require_label:
        raise ValueError("No target label column found (expected 'True_Label', 'Glaucoma_Prediction', or 'Glaucoma Decision').")

    return X, y, available_features

def evaluate_extracted_features(csv_path: str, labels_csv: str = None):
    print("=" * 80)
    print("STRATIFIED K-FOLD EVALUATION FOR EXTRACTED FEATURES")
    print("=" * 80)

    # 1. Load the data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    # 2. Clean and prep the data
    # Drop rows where processing failed or features are missing
    if 'Processing_Status' in df.columns:
        df = df[df['Processing_Status'] == 'SUCCESS']

    available_features = [col for col in FEATURE_COLS if col in df.columns]
    if not available_features:
        raise ValueError("No expected feature columns found in dataframe.")

    required_cols = available_features + (["Images"] if "Images" in df.columns else [])
    df = df.dropna(subset=required_cols)

    if labels_csv:
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df.dropna(subset=['Images', 'Glaucoma Decision'])
        df = merge_with_ground_truth(df, labels_df)
        print(f"Using real ground-truth labels from: {labels_csv}")
        print(f"Matched samples for k-fold: {len(df)}")
    else:
        df = df.dropna()
        if 'Glaucoma Decision' in df.columns:
            print("Warning: Using 'Glaucoma Decision' from extracted CSV. If this column was rule-generated, this is not true clinical ground truth.")

    # 3. Define features and target
    X, y, available_features = prepare_xy(df, require_label=True)

    print(f"Dataset Size: {len(df)} patients")
    print(f"Features used: {len(available_features)} {available_features}")
    print(f"Class distribution: {sum(y==0)} Normal, {sum(y==1)} Glaucoma/Suspect\n")

    # 4. Setup Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize the classifier (Random Forest works great for tabular medical data)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    scaler = StandardScaler()

    # Metrics storage
    accuracies, precisions, recalls, f1s = [], [], [], []
    total_cm = np.zeros((2, 2))

    # 5. Run the Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        total_cm += confusion_matrix(y_test, y_pred, labels=[0, 1])

        print(f"Fold {fold}: Accuracy = {acc:.4f}")

    # 6. Print Final Results
    print("\n" + "=" * 80)
    print("FINAL CROSS-VALIDATION RESULTS (5-Fold)")
    print("=" * 80)
    print(f"Mean Accuracy:  {np.mean(accuracies):.4f} (± {np.std(accuracies):.4f})")
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall:    {np.mean(recalls):.4f}")
    print(f"Mean F1-Score:  {np.mean(f1s):.4f}")
    
    print("\nCumulative Confusion Matrix:")
    print(f"                 Predicted Normal | Predicted Glaucoma")
    print(f"Actual Normal   | {int(total_cm[0,0]):<15} | {int(total_cm[0,1])}")
    print(f"Actual Glaucoma | {int(total_cm[1,0]):<15} | {int(total_cm[1,1])}")
    
    # 7. Feature Importance
    clf.fit(scaler.fit_transform(X), y) # Train on full dataset for importances
    importances = pd.DataFrame({
        'Feature': available_features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(importances.head().to_string(index=False))


def evaluate_train_test(train_csv: str, test_features_csv: str, test_labels_csv: str, train_labels_csv: str = None):
    """Train on train CSV and evaluate on held-out test CSV with real test labels."""
    print("=" * 80)
    print("TRAIN/TEST EVALUATION WITH REAL TEST LABELS")
    print("=" * 80)

    train_df = pd.read_csv(train_csv).dropna()
    test_df = pd.read_csv(test_features_csv).dropna()
    labels_df = pd.read_csv(test_labels_csv).dropna()

    if train_labels_csv:
        train_labels_df = pd.read_csv(train_labels_csv).dropna(subset=['Images', 'Glaucoma Decision'])
        train_df = merge_with_ground_truth(train_df, train_labels_df)

    X_train, y_train, available_features = prepare_xy(train_df, require_label=True)
    _, _, _ = prepare_xy(test_df, require_label=False)

    if 'Images' not in test_df.columns or 'Images' not in labels_df.columns:
        raise ValueError("Both test feature CSV and test label CSV must contain an 'Images' column.")

    if 'Glaucoma Decision' not in labels_df.columns:
        raise ValueError("Test labels CSV must contain 'Glaucoma Decision' column.")

    test_df = test_df.copy()
    labels_df = labels_df.copy()
    test_df['normalized_name'] = test_df['Images'].apply(normalize_name)
    labels_df['normalized_name'] = labels_df['Images'].apply(normalize_name)
    labels_df = labels_df.rename(columns={'Glaucoma Decision': 'True_Label'})

    merged = pd.merge(
        test_df,
        labels_df[['normalized_name', 'True_Label']],
        on='normalized_name',
        how='inner'
    )

    if len(merged) == 0:
        raise ValueError("No matching image IDs between test features CSV and test labels CSV.")

    X_test_eval = merged[available_features].values
    y_test = to_binary_label(merged['True_Label'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_eval)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"Train samples: {len(train_df)}")
    print(f"Matched test samples: {len(merged)}")
    print(f"Features used: {len(available_features)} {available_features}")
    print(f"Test class distribution: {sum(y_test==0)} Normal, {sum(y_test==1)} Glaucoma/Suspect\n")

    print("=" * 80)
    print("HELD-OUT TEST RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(f"                 Predicted Normal | Predicted Glaucoma")
    print(f"Actual Normal   | {int(cm[0,0]):<15} | {int(cm[0,1])}")
    print(f"Actual Glaucoma | {int(cm[1,0]):<15} | {int(cm[1,1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate extracted glaucoma features")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["kfold", "test", "both"],
        help="Evaluation mode: train k-fold, held-out test, or both"
    )
    parser.add_argument(
        "--train-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/extracted_features_remidio_train.csv",
        help="Path to extracted train features CSV"
    )
    parser.add_argument(
        "--kfold-labels-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Train/6.0_Glaucoma_Decision/Majority/Remidio.csv",
        help="Path to original train labels CSV used as ground truth for k-fold"
    )
    parser.add_argument(
        "--test-features-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/extracted_features_remidio_test.csv",
        help="Path to extracted test features CSV"
    )
    parser.add_argument(
        "--test-labels-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Test/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Remidio_majority.csv",
        help="Path to real test labels CSV"
    )
    parser.add_argument(
        "--train-labels-csv",
        default=None,
        help="Optional path to train labels CSV (recommended for true train supervision)"
    )
    args = parser.parse_args()

    if args.mode in ["kfold", "both"]:
        try:
            evaluate_extracted_features(args.train_csv, args.kfold_labels_csv)
        except FileNotFoundError as e:
            print(f"\nK-fold ran without external labels (missing file): {e}")
            evaluate_extracted_features(args.train_csv, labels_csv=None)

    if args.mode in ["test", "both"]:
        try:
            evaluate_train_test(args.train_csv, args.test_features_csv, args.test_labels_csv, args.train_labels_csv)
        except FileNotFoundError as e:
            print(f"\nSkipping held-out test evaluation (missing file): {e}")
        except ValueError as e:
            print(f"\nSkipping held-out test evaluation: {e}")