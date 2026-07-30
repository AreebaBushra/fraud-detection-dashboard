# Credit Card Fraud Detection Dashboard

A fraud monitoring dashboard powered by an XGBoost model trained on the
[mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset (284,807 anonymized European card transactions, 492 confirmed fraud
cases), with per-transaction explanations generated using SHAP.

## Why a dashboard, not a "type in your card details" form

The dataset's features (`V1`-`V28`) are the output of a PCA transformation
applied by the original data provider to protect real cardholder privacy.
That transformation was never publicly released, so there's no way to
convert a brand-new, live-entered transaction into those same features.
Because of that, this app is built the way real-world fraud detection
actually works: as an internal monitoring dashboard that reviews a stream
of transactions (here, pulled from the held-out test set) rather than a
customer-facing form.

## Model

- **Algorithm:** XGBoost Classifier
- **Class imbalance handling:** `scale_pos_weight` tuned to the ~577:1
  normal-to-fraud ratio in the training data
- **Performance on held-out test set:**
  - ROC-AUC: 0.977
  - Precision (fraud class): 0.844
  - Recall (fraud class): 0.827
- **Explainability:** SHAP `TreeExplainer`, top-5 contributing features
  precomputed per transaction and served with each prediction

## Project structure

```
fraud_dashboard/
├── app.py                  # FastAPI backend
├── requirements.txt
├── Procfile                 # Render start command
├── data/
│   └── transactions.json    # Precomputed sample transactions + SHAP explanations
├── model/
│   ├── xgb_fraud_model.pkl
│   ├── time_scaler.pkl
│   └── amount_scaler.pkl
├── static/
│   └── index.html            # Dashboard frontend (vanilla HTML/CSS/JS)
└── notebooks/
    ├── creditcardfrauddetection.ipynb   # EDA + model training
    └── shap_explainability.ipynb         # SHAP analysis
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://localhost:8000`.

## Deploying on Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   (already set in the `Procfile`, Render should detect it automatically)

## Notes on the "why" behind key decisions

- **Metric choice:** with fraud at just 0.17% of transactions, accuracy is
  meaningless (predicting "not fraud" always scores 99.8%). This project
  uses precision, recall, and ROC-AUC instead.
- **XGBoost over Random Forest:** chosen to show a distinct technique from
  prior projects, and because gradient boosting is the industry-standard
  approach for imbalanced tabular fraud problems.
- **SHAP over generic feature importance:** SHAP explains *individual*
  predictions, not just overall feature rankings, which is what a fraud
  analyst reviewing one flagged transaction actually needs.
