# CustomerIQ — E-Commerce Customer Satisfaction Predictor

A Flask + scikit-learn web app that predicts whether an e-commerce customer will be
**Satisfied**, **Neutral**, or **Unsatisfied** based on their profile and shopping behavior,
and returns concrete recommendations for the next best action.

Trained on the bundled `E-commerce Customer Behavior` dataset (~348 customers).

## Features

- **Random Forest classifier** with full preprocessing pipeline (imputation, label encoding, scaling)
- **Modern, responsive UI** with a prediction form and inline AJAX predictions
- **Dataset dashboard** with KPIs, satisfaction breakdown, city/membership splits, and feature importances
- **REST API** (`POST /api/predict`) returning JSON predictions, probabilities, and recommendations
- **Smart recommendations engine** that personalizes follow-up actions based on the predicted class and inputs (e.g. days since last purchase, average rating)

## Project layout

```
ecommerce_prediction/
├── app.py                  # Flask app (routes, prediction, API)
├── train_model.py          # Trains the model and writes artifacts
├── requirements.txt
├── data/
│   └── ecommerce_customer_behavior.csv
├── models/                 # Saved model + encoders + metadata (generated)
├── static/
│   ├── css/style.css
│   └── js/script.js
└── templates/
    ├── base.html
    ├── index.html          # Prediction form
    ├── result.html         # Server-rendered result page
    └── dashboard.html      # Dataset insights
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (writes artifacts into ./models)
python train_model.py

# 3. Run the web app
python app.py
```

Then open <http://127.0.0.1:5000>.

## Endpoints

| Method | Path             | Description                                  |
| ------ | ---------------- | -------------------------------------------- |
| GET    | `/`              | Prediction form (HTML)                       |
| POST   | `/predict`       | Submits the form, renders the result page    |
| GET    | `/dashboard`     | Dataset insights & feature importances       |
| GET    | `/api/metadata`  | Model metadata (classes, choices, metrics)   |
| POST   | `/api/predict`   | JSON prediction endpoint                     |

### Example: JSON prediction

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 30,
    "Gender": "Female",
    "City": "New York",
    "Membership Type": "Gold",
    "Total Spend": 1200,
    "Items Purchased": 15,
    "Average Rating": 4.6,
    "Days Since Last Purchase": 20,
    "Discount Applied": "Yes"
  }'
```

Response:

```json
{
  "prediction": "Satisfied",
  "confidence": 100.0,
  "probabilities": {"Neutral": 0.0, "Satisfied": 100.0, "Unsatisfied": 0.0},
  "recommendations": [
    "Invite the customer to a referral / VIP program.",
    "Ask for a public review – their advocacy is high-value."
  ]
}
```

## Model

- **Algorithm:** `RandomForestClassifier` (300 trees, `random_state=42`)
- **Target:** `Satisfaction Level` (3 classes)
- **Features:** `Age`, `Total Spend`, `Items Purchased`, `Average Rating`, `Days Since Last Purchase`, `Gender`, `City`, `Membership Type`, `Discount Applied`
- **Preprocessing:** median/mode imputation → label-encoded categoricals → `StandardScaler` on numerics
- **Holdout test accuracy:** ~1.00 on this clean, small dataset. (See `models/metadata.json` for the full classification report and confusion matrix.)

> The dataset is small and quite structured, so accuracy is unusually high. With richer/noisier production data you'd want cross-validation and likely a regularized model or gradient boosting.

## Retraining

Drop a new CSV with the same columns into `data/ecommerce_customer_behavior.csv`, then run:

```bash
python train_model.py
```

Restart the Flask app to pick up the new artifacts.
