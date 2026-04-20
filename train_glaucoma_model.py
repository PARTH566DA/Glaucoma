#!/usr/bin/env python3
"""
Train glaucoma model with proper label CSV and save payload for API inference.
Also computes anomaly detector on training feature distribution.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ACDR",
    "VCDR",
    "HCDR",
]


def normalize_name(name: str) -> str:
    name = str(name)
    base = name.split("-")[0]
    base = base.split(".")[0]
    return base.lower()


def to_multiclass_label(series: pd.Series) -> np.ndarray:
    labels = series.astype(str).str.upper()
    # 0 = Normal, 1 = Suspect, 2 = Glaucoma
    conditions = [
        labels.str.contains("SUSPECT|SUSUPECT"), # catches typoes in CSV like SUSUPECT
        labels.str.contains("GLAUCOMA") # Strict Glaucoma
    ]
    choices = [1, 2]
    return np.select(conditions, choices, default=0)


def merge_features_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if "Images" not in features_df.columns or "Images" not in labels_df.columns:
        raise ValueError("Both CSVs must have an 'Images' column.")
    if "Glaucoma Decision" not in labels_df.columns:
        raise ValueError("Labels CSV must contain 'Glaucoma Decision' column.")

    f = features_df.copy()
    l = labels_df.copy()

    if "Processing_Status" in f.columns:
        f = f[f["Processing_Status"] == "SUCCESS"]

    f["normalized_name"] = f["Images"].apply(normalize_name)
    l["normalized_name"] = l["Images"].apply(normalize_name)

    l = l.rename(columns={"Glaucoma Decision": "True_Label"})

    merged = pd.merge(
        f,
        l[["normalized_name", "True_Label"]],
        on="normalized_name",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No matched rows between features and labels.")

    return merged


def prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(cols) != len(FEATURE_COLS):
        missing = [c for c in FEATURE_COLS if c not in cols]
        raise ValueError(f"Missing feature columns: {missing}")

    X_df = df[cols].apply(pd.to_numeric, errors="coerce")
    mask = ~X_df.isna().any(axis=1)
    X = X_df[mask].values
    y = to_multiclass_label(df.loc[mask, "True_Label"])
    return X, y, cols


def cv_report(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, ps, rs, f1s = [], [], [], []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(X[tr])
        xte = scaler.transform(X[te])

        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )

        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(xtr, y[tr])
        yp = clf.predict(xte)

        accs.append(accuracy_score(y[te], yp))
        ps.append(precision_score(y[te], yp, zero_division=0, average='weighted'))
        rs.append(recall_score(y[te], yp, zero_division=0, average='weighted'))
        f1s.append(f1_score(y[te], yp, zero_division=0, average='weighted'))

    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_precision_mean": float(np.mean(ps)),
        "cv_recall_mean": float(np.mean(rs)),
        "cv_f1_mean": float(np.mean(f1s)),
    }


def train_and_save(train_features_csv: str, train_labels_csv: str, output_model: str) -> Dict:
    f = pd.read_csv(train_features_csv)
    l = pd.read_csv(train_labels_csv)

    merged = merge_features_labels(f, l)
    X, y, cols = prepare_xy(merged)

    report = cv_report(X, y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(Xs, y)

    # Train an unsupervised anomaly detector on normal feature distribution.
    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(Xs)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)

    payload = {
        "model": clf,
        "scaler": scaler,
        "feature_order": cols,
        "anomaly_detector": iso,
        "anomaly_threshold": -0.10,
        "label_schema": "mean_train_glaucoma_decision",
        "trained_from_features": train_features_csv,
        "trained_from_labels": train_labels_csv,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": report,
        "n_train": int(len(X)),
        "class_distribution": {
            "normal": int(np.sum(y == 0)),
            "glaucoma_or_suspect": int(np.sum(y == 1)),
        },
    }

    joblib.dump(payload, output_model)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train glaucoma model with Mean labels and save for API.")
    parser.add_argument(
        "--train-features-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/extracted_features_remidio_train.csv",
        help="Path to extracted train features CSV",
    )
    parser.add_argument(
        "--train-labels-csv",
        default="/Users/dau/Desktop/college/BTP/Dataset/Chakshu/20123135/Train/6.0_Glaucoma_Decision/Mean/Remidio.csv",
        help="Path to train labels CSV (must contain Glaucoma Decision)",
    )
    parser.add_argument(
        "--output-model",
        default="models/glaucoma_rf.joblib",
        help="Output model path",
    )

    args = parser.parse_args()

    payload = train_and_save(
        train_features_csv=args.train_features_csv,
        train_labels_csv=args.train_labels_csv,
        output_model=args.output_model,
    )

    print("=" * 80)
    print("MODEL TRAINED AND SAVED")
    print("=" * 80)
    print(f"Model path: {args.output_model}")
    print(f"Train samples: {payload['n_train']}")
    print(f"Class distribution: {payload['class_distribution']}")
    print(f"Metrics: {json.dumps(payload['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
