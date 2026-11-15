"""Train an E-commerce Customer Satisfaction prediction model.

Loads the customer-behavior dataset, performs preprocessing
(imputation, encoding, scaling), trains a RandomForestClassifier
to predict Satisfaction Level, evaluates it, and saves all the
artifacts needed by the Flask app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ecommerce_customer_behavior.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "Age",
    "Total Spend",
    "Items Purchased",
    "Average Rating",
    "Days Since Last Purchase",
]
CATEGORICAL_FEATURES = ["Gender", "City", "Membership Type", "Discount Applied"]
TARGET = "Satisfaction Level"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize the discount column to strings so LabelEncoder is happy
    df["Discount Applied"] = df["Discount Applied"].astype(str).str.upper().map(
        {"TRUE": "Yes", "FALSE": "No"}
    )
    # Drop rows with a missing target (a few rows in this file)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame):
    # Impute numerics with median, categoricals with mode
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    df[TARGET] = target_encoder.fit_transform(df[TARGET].astype(str))

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    scaler = StandardScaler()
    X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

    return X, y, encoders, target_encoder, scaler, feature_cols


def train_and_evaluate():
    print(f"Reading dataset from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    raw_choices = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }

    X, y, encoders, target_encoder, scaler, feature_cols = build_features(df.copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=target_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nAccuracy: {acc:.4f}")
    print(
        "\nClassification report:\n",
        classification_report(y_test, y_pred, target_names=target_encoder.classes_),
    )

    feature_importances = sorted(
        zip(feature_cols, model.feature_importances_.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    joblib.dump(target_encoder, MODEL_DIR / "target_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metadata = {
        "feature_columns": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_classes": target_encoder.classes_.tolist(),
        "category_choices": raw_choices,
        "metrics": {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
        },
        "feature_importances": feature_importances,
        "n_samples": int(len(df)),
        "class_distribution": (
            df[TARGET]
            .map(dict(enumerate(target_encoder.classes_)))
            .value_counts()
            .to_dict()
        ),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to: {MODEL_DIR}")
    print("Top features:")
    for name, importance in feature_importances:
        print(f"  {name:28s} {importance:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
