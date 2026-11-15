"""Flask app for E-commerce Customer Satisfaction prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "ecommerce_customer_behavior.csv"

app = Flask(__name__)


def _load_artifacts():
    model = joblib.load(MODEL_DIR / "model.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    target_encoder = joblib.load(MODEL_DIR / "target_encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, encoders, target_encoder, scaler, metadata


MODEL, ENCODERS, TARGET_ENCODER, SCALER, METADATA = _load_artifacts()

NUMERIC_FEATURES = METADATA["numeric_features"]
CATEGORICAL_FEATURES = METADATA["categorical_features"]
FEATURE_COLUMNS = METADATA["feature_columns"]
TARGET_CLASSES = METADATA["target_classes"]
CHOICES = METADATA["category_choices"]


def _safe_encode(encoder, value: str) -> int:
    """Encode a categorical value; fall back to the most common class if unseen."""
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return int(encoder.transform([encoder.classes_[0]])[0])


def _build_feature_row(payload: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERIC_FEATURES:
        try:
            row[col] = float(payload.get(col, 0) or 0)
        except (TypeError, ValueError):
            row[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        row[col] = _safe_encode(ENCODERS[col], payload.get(col, ""))

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df[NUMERIC_FEATURES] = SCALER.transform(df[NUMERIC_FEATURES])
    return df


def _recommendations(label: str, payload: dict) -> list[str]:
    tips: list[str] = []
    if label == "Unsatisfied":
        tips.append("Reach out with a personal apology and a 15-20% loyalty coupon.")
        tips.append("Investigate recent orders for delivery or quality issues.")
    elif label == "Neutral":
        tips.append("Send a curated product recommendation email this week.")
        tips.append("Offer free shipping on the next order to nudge re-engagement.")
    else:
        tips.append("Invite the customer to a referral / VIP program.")
        tips.append("Ask for a public review – their advocacy is high-value.")

    try:
        days = float(payload.get("Days Since Last Purchase", 0))
        if days > 30:
            tips.append(f"It has been {int(days)} days since last purchase – send a win-back email.")
    except (TypeError, ValueError):
        pass

    try:
        rating = float(payload.get("Average Rating", 0))
        if rating and rating < 3.5:
            tips.append("Average rating is below 3.5 – flag for the support team to review feedback.")
    except (TypeError, ValueError):
        pass

    return tips


@app.route("/")
def index():
    return render_template(
        "index.html",
        choices=CHOICES,
        numeric_features=NUMERIC_FEATURES,
        target_classes=TARGET_CLASSES,
        metadata=METADATA,
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.form.to_dict() if request.form else request.get_json(silent=True) or {}

    try:
        features = _build_feature_row(payload)
        proba = MODEL.predict_proba(features)[0]
        pred_idx = int(np.argmax(proba))
        label = TARGET_ENCODER.classes_[pred_idx]
        confidence = float(proba[pred_idx]) * 100
        proba_dict = {
            TARGET_ENCODER.classes_[i]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify(
            {
                "prediction": label,
                "confidence": round(confidence, 2),
                "probabilities": proba_dict,
                "recommendations": _recommendations(label, payload),
            }
        )

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(confidence, 2),
        probabilities=proba_dict,
        recommendations=_recommendations(label, payload),
        input_data=payload,
    )


@app.route("/dashboard")
def dashboard():
    df = pd.read_csv(DATA_PATH)
    df["Discount Applied"] = df["Discount Applied"].astype(str).str.upper().map(
        {"TRUE": "Yes", "FALSE": "No"}
    )
    df = df.dropna(subset=["Satisfaction Level"])

    stats = {
        "total_customers": int(len(df)),
        "avg_spend": round(float(df["Total Spend"].mean()), 2),
        "avg_rating": round(float(df["Average Rating"].mean()), 2),
        "avg_items": round(float(df["Items Purchased"].mean()), 2),
        "satisfaction_counts": df["Satisfaction Level"].value_counts().to_dict(),
        "membership_counts": df["Membership Type"].value_counts().to_dict(),
        "city_counts": df["City"].value_counts().to_dict(),
        "gender_counts": df["Gender"].value_counts().to_dict(),
        "spend_by_membership": df.groupby("Membership Type")["Total Spend"]
        .mean()
        .round(2)
        .to_dict(),
    }

    return render_template(
        "dashboard.html",
        stats=stats,
        metadata=METADATA,
    )


@app.route("/api/metadata")
def api_metadata():
    return jsonify(METADATA)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        features = _build_feature_row(payload)
        proba = MODEL.predict_proba(features)[0]
        pred_idx = int(np.argmax(proba))
        label = TARGET_ENCODER.classes_[pred_idx]
        return jsonify(
            {
                "prediction": label,
                "confidence": round(float(proba[pred_idx]) * 100, 2),
                "probabilities": {
                    TARGET_ENCODER.classes_[i]: round(float(p) * 100, 2)
                    for i, p in enumerate(proba)
                },
                "recommendations": _recommendations(label, payload),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
